import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from .dataset import make_dataset, make_dataset_fixed_snr
from .snn_model import SNNModulator
from .profiler import hardware_summary, estimate_energy
from .signal_generator import SignalGenerator
from .encoding import iq_to_symbol_vector
from torch.optim.lr_scheduler import LambdaLR
from .quantize import quantize_model, prune_by_magnitude
from .channels import impair
import numpy as np
import random
import shutil
import json

def train_with_config(config: dict):
    mods = tuple(config.get("mods", ("BPSK","QPSK","8PSK","16QAM","16PSK","64QAM")))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    model = None
    opt = None
    best_val = 0.0
    stages = config.get("stages", [((20.0, 20.0), 15), ((15.0, 20.0), 15), ((10.0, 20.0), 10), ((5.0, 20.0), 10)])
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    for si, (rng, epochs) in enumerate(stages, start=1):
        X, y = make_dataset(
            mods=mods,
            use_matched=True,
            samples_per_class=int(config.get("samples_per_class", 1200)),
            symbols_per_sample=int(config.get("symbols_per_sample", 4096)),
            snr_range=rng,
            augment=bool(config.get("augment", True)),
            impair_prob=float(config.get("train_impair_prob", 0.0)),
            cfo=float(config.get("cfo", 0.0)),
            rayleigh=bool(config.get("rayleigh", False)),
            rayleigh_severity=float(config.get("rayleigh_severity", 1.0)),
            sps=int(config.get("sps", 16)),
        )
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        ds = TensorDataset(X_t, y_t)
        n = len(ds)
        n_val = int(0.2 * n)
        n_train = n - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        bs = int(config.get("batch_size", 128))
        dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=bs)
        if model is None:
            model = SNNModulator(input_dim=X.shape[1], num_classes=len(mods), hidden1=int(config.get("hidden1", 512)), hidden2=int(config.get("hidden2", 256)), drop_p=float(config.get("dropout", 0.1)))
            opt = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-5)))
            model.train()
            warm_steps = int(config.get("warmup_steps", 5))
            scheduler = LambdaLR(opt, lr_lambda=lambda e: min(1.0, (e + 1) / max(1, warm_steps)))
            early_best = None
            early_patience = int(config.get("early_patience", 5))
            early_wait = 0
        else:
            scheduler = None
        for epoch in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                out, spk = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            if scheduler is not None:
                scheduler.step()
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    out, _ = model(xb)
                    pred = out.argmax(dim=1)
                    correct += int((pred == yb).sum().item())
                    total += int(yb.numel())
            val_acc = correct / total if total > 0 else 0.0
            print(f"Stage {si} rng={rng} epoch {epoch+1}/{epochs} - val_acc={val_acc:.4f}")
            model.train()
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), os.path.join("reports", "checkpoint_best.pt"))
                if si == 1:
                    early_best = val_acc
                    early_wait = 0
            else:
                if si == 1:
                    early_wait += 1
                    if early_wait >= early_patience:
                        print("Early stopping Stage 1")
                        break
    # Load best checkpoint if available
    ckpt_path = os.path.join("reports", "checkpoint_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    with torch.no_grad():
        out, spk = model(X_t)
        pred = out.argmax(dim=1).cpu().numpy()
        acc = float(np.mean(pred == y))
        spikes = spk.sum().item()
    cm = np.zeros((len(mods), len(mods)), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[int(t), int(p)] += 1
    per_class = []
    for c in range(len(mods)):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        per_class.append((prec, rec, f1))
    macro_f1 = float(np.mean([f1 for _, _, f1 in per_class]))
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "results.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Total Spikes (hidden): {int(spikes)}\n")
        energy_nJ = estimate_energy(int(spikes)) * 1e9
        f.write(f"Estimated Energy per inference (nJ): {energy_nJ:.2f}\n")
        f.write(hardware_summary(int(spikes)))
        f.write("\nPer-class metrics (precision, recall, F1):\n")
        for i, (prec, rec, f1) in enumerate(per_class):
            f.write(f"{mods[i]}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}\n")
        f.write(f"Macro-F1: {macro_f1:.3f}\n")
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(mods)), list(mods))
    plt.yticks(range(len(mods)), list(mods))
    for i in range(len(mods)):
        for j in range(len(mods)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "confusion_matrix.png"))
    snr_vals = [0, 5, 10, 15, 20]
    acc_by_snr = []
    per_class_curves = [[] for _ in range(len(mods))]
    for snr in snr_vals:
        Xt, yt = make_dataset_fixed_snr(samples_per_class=128, symbols_per_sample=2048, snr_db=snr, mods=mods, use_matched=True)
        Xt_t = torch.from_numpy(Xt)
        with torch.no_grad():
            out_t, _ = model(Xt_t)
            pred_t = out_t.argmax(dim=1).cpu().numpy()
            acc_t = float(np.mean(pred_t == yt))
            acc_by_snr.append(acc_t)
            for c in range(len(mods)):
                mask = (yt == c)
                if mask.sum() > 0:
                    per_class_acc = float(np.mean(pred_t[mask] == yt[mask]))
                else:
                    per_class_acc = 0.0
                per_class_curves[c].append(per_class_acc)
    plt.figure(figsize=(6,4))
    plt.plot(snr_vals, acc_by_snr, marker="o")
    plt.title("Accuracy vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "snr_curve.png"))
    plt.figure(figsize=(6,4))
    for i, cls_curve in enumerate(per_class_curves):
        plt.plot(snr_vals, cls_curve, marker="o", label=mods[i])
    plt.title("Per-class Accuracy vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "snr_curve_per_class.png"))
    cfo_val = float(config.get("cfo", 0.0))
    use_rayleigh = bool(config.get("rayleigh", False))
    ray_sev = float(config.get("rayleigh_severity", 1.0))
    sps_cfg = int(config.get("sps", 16))
    run_imp = bool(config.get("impaired", False)) or (cfo_val != 0.0) or use_rayleigh
    if run_imp:
        snr_vals_imp = [0, 5, 10, 15, 20]
        acc_by_snr_imp = []
        per_class_imp = [[] for _ in range(len(mods))]
        gen_imp = SignalGenerator()
        for snr in snr_vals_imp:
            X_list = []
            y_list = []
            for label, mod in enumerate(mods):
                for _ in range(128):
                    if mod == "BPSK":
                        sig, _ = gen_imp.generate_bpsk(2048, snr_db=snr)
                    elif mod == "QPSK":
                        sig, _ = gen_imp.generate_qpsk(2048, snr_db=snr)
                    elif mod == "8PSK":
                        sig, _ = gen_imp.generate_8psk(2048, snr_db=snr)
                    elif mod == "16QAM":
                        sig, _ = gen_imp.generate_16qam(2048, snr_db=snr)
                    elif mod == "16PSK":
                        sig, _ = gen_imp.generate_16psk(2048, snr_db=snr)
                    elif mod == "64QAM":
                        sig, _ = gen_imp.generate_64qam(2048, snr_db=snr)
                    else:
                        continue
                    sig_i = impair(sig, sps=sps_cfg, cfo=cfo_val, rayleigh=use_rayleigh, rayleigh_severity=ray_sev)
                    feat = iq_to_symbol_vector(sig_i, sps=sps_cfg, n_symbols=256)
                    X_list.append(feat)
                    y_list.append(label)
            Xt_imp = np.stack(X_list, axis=0).astype(np.float32)
            yt_imp = np.array(y_list, dtype=np.int64)
            Xt_imp_t = torch.from_numpy(Xt_imp)
            with torch.no_grad():
                out_imp, _ = model(Xt_imp_t)
                pred_imp = out_imp.argmax(dim=1).cpu().numpy()
                acc_imp = float(np.mean(pred_imp == yt_imp))
                acc_by_snr_imp.append(acc_imp)
                for c in range(len(mods)):
                    mask = (yt_imp == c)
                    if mask.sum() > 0:
                        per_class_acc_imp = float(np.mean(pred_imp[mask] == yt_imp[mask]))
                    else:
                        per_class_acc_imp = 0.0
                    per_class_imp[c].append(per_class_acc_imp)
        plt.figure(figsize=(6,4))
        plt.plot(snr_vals_imp, acc_by_snr_imp, marker="o", label="Impaired")
        plt.plot(snr_vals, acc_by_snr, marker="s", label="Clean")
        plt.title("Accuracy vs SNR (Impaired vs Clean)")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("reports", "snr_curve_impaired.png"))
        plt.figure(figsize=(6,4))
        for i, cls_curve in enumerate(per_class_imp):
            plt.plot(snr_vals_imp, cls_curve, marker="o", label=mods[i])
        plt.title("Per-class Accuracy vs SNR (Impaired)")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("reports", "snr_curve_per_class_impaired.png"))
    # Constellation snapshots
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    gen = SignalGenerator()
    for snr_snap in [5, 15]:
        for mod in mods:
            if mod == "BPSK":
                sig, _ = gen.generate_bpsk(2000, snr_db=snr_snap)
            elif mod == "QPSK":
                sig, _ = gen.generate_qpsk(2000, snr_db=snr_snap)
            elif mod == "8PSK":
                sig, _ = gen.generate_8psk(2000, snr_db=snr_snap)
            elif mod == "16QAM":
                sig, _ = gen.generate_16qam(2000, snr_db=snr_snap)
            elif mod == "16PSK":
                sig, _ = gen.generate_16psk(2000, snr_db=snr_snap)
            elif mod == "64QAM":
                sig, _ = gen.generate_64qam(2000, snr_db=snr_snap)
            else:
                continue
            plt.figure(figsize=(5,5))
            plt.scatter(np.real(sig), np.imag(sig), s=3, alpha=0.5)
            plt.title(f"{mod} Constellation (SNR={snr_snap} dB)")
            plt.xlabel("I")
            plt.ylabel("Q")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join("reports","figures",f"constellation_{mod}_{snr_snap}dB.png"))
            plt.close()
    # SER curves (classical decisions)
    def demod_psk(symbols: np.ndarray, M: int):
        ang = np.angle(symbols)
        idx = np.mod(np.round((ang / (2*np.pi)) * M), M).astype(int)
        recon = np.exp(1j * (2*np.pi*idx / M))
        return idx, recon
    def demod_qam(symbols: np.ndarray, levels: np.ndarray, norm: float):
        i = np.real(symbols) * norm
        q = np.imag(symbols) * norm
        i_hat = levels[np.argmin(np.abs(i[:, None] - levels[None, :]), axis=1)]
        q_hat = levels[np.argmin(np.abs(q[:, None] - levels[None, :]), axis=1)]
        recon = (i_hat + 1j * q_hat) / norm
        return recon
    snr_vals_ser = [0,5,10,15,20]
    ser_curves = {m: [] for m in mods}
    for snr in snr_vals_ser:
        # BPSK
        sig, _ = gen.generate_bpsk(4000, snr_db=snr)
        _, recon = demod_psk(sig, 2)
        ser_curves["BPSK"].append(float(np.mean(np.abs(recon - sig) > 1e-6)))
        # QPSK
        sig, _ = gen.generate_qpsk(4000, snr_db=snr)
        _, recon = demod_psk(sig, 4)
        ser_curves["QPSK"].append(float(np.mean(np.abs(recon - sig) > 1e-6)))
        # 8PSK
        sig, _ = gen.generate_8psk(4000, snr_db=snr)
        _, recon = demod_psk(sig, 8)
        ser_curves["8PSK"].append(float(np.mean(np.abs(recon - sig) > 1e-6)))
        # 16QAM
        sig, _ = gen.generate_16qam(4000, snr_db=snr)
        recon = demod_qam(sig, levels=np.array([-3,-1,1,3], dtype=np.float32), norm=np.sqrt(10.0))
        ser_curves["16QAM"].append(float(np.mean(np.abs(recon - sig) > 1e-6)))
        # 16PSK
        sig, _ = gen.generate_16psk(4000, snr_db=snr)
        _, recon = demod_psk(sig, 16)
        ser_curves["16PSK"].append(float(np.mean(np.abs(recon - sig) > 1e-6)))
        # 64QAM
        sig, _ = gen.generate_64qam(4000, snr_db=snr)
        recon = demod_qam(sig, levels=np.array([-7,-5,-3,-1,1,3,5,7], dtype=np.float32), norm=np.sqrt(42.0))
        ser_curves["64QAM"].append(float(np.mean(np.abs(recon - sig) > 1e-6)))
    plt.figure(figsize=(6,4))
    for m in mods:
        plt.plot(snr_vals_ser, ser_curves[m], marker="o", label=m)
    plt.title("Classical SER vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("reports","figures","ser_curves.png"))
    avg_acc_classical = [1.0 - float(np.mean([ser_curves[m][i] for m in mods])) for i in range(len(snr_vals_ser))]
    plt.figure(figsize=(6,4))
    plt.plot(snr_vals, acc_by_snr, marker="o", label="SNN")
    plt.plot(snr_vals_ser, avg_acc_classical, marker="s", label="Classical Avg")
    plt.title("SNN Accuracy vs Classical Avg(1-SER)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("reports","figures","snn_vs_ser.png"))
    q_model = quantize_model(model, bits=8)
    prune_by_magnitude(q_model, pct=0.1)
    torch.save(q_model.state_dict(), os.path.join("reports", "snn_radio_quant_pruned.pt"))
    torch.save(model.state_dict(), os.path.join("reports", "snn_radio.pt"))
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Saved: reports/results.txt")
    print("Saved: reports/confusion_matrix.png")
    print("Saved: reports/snr_curve.png")
    print("Saved: reports/snr_curve_per_class.png")
    print("Saved: reports/figures/constellation_*_*.png")
    print("Saved: reports/figures/ser_curves.png")
    os.makedirs(os.path.join("reports","best","figures"), exist_ok=True)
    meta_path = os.path.join("reports","best","best.json")
    prev = {"best_acc": -1.0}
    if os.path.exists(meta_path):
        try:
            prev = json.load(open(meta_path))
        except Exception:
            prev = {"best_acc": -1.0}
    if acc > float(prev.get("best_acc", -1.0)):
        shutil.copy2(os.path.join("reports","results.txt"), os.path.join("reports","best","results.txt"))
        shutil.copy2(os.path.join("reports","confusion_matrix.png"), os.path.join("reports","best","confusion_matrix.png"))
        shutil.copy2(os.path.join("reports","snr_curve.png"), os.path.join("reports","best","snr_curve.png"))
        shutil.copy2(os.path.join("reports","snr_curve_per_class.png"), os.path.join("reports","best","snr_curve_per_class.png"))
        for name in ["ser_curves.png","snn_vs_ser.png"]:
            src = os.path.join("reports","figures",name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("reports","best","figures",name))
        shutil.copy2(os.path.join("reports","snn_radio.pt"), os.path.join("reports","best","snn_radio.pt"))
        try:
            shutil.copy2(os.path.join("reports","checkpoint_best.pt"), os.path.join("reports","best","checkpoint_best.pt"))
        except Exception:
            pass
        json.dump({"best_acc": acc, "macro_f1": macro_f1}, open(meta_path,"w"), indent=2)
        print("Updated best run under reports/best/")
    return acc, int(spikes)

def train_and_eval():
    cfg = {
        "mods": ("BPSK","QPSK","8PSK","16QAM","16PSK","64QAM"),
        "samples_per_class": 1200,
        "symbols_per_sample": 4096,
        "stages": [((20.0,20.0),15), ((15.0,20.0),15), ((10.0,20.0),10), ((5.0,20.0),10)],
        "batch_size": 128,
        "hidden1": 512,
        "hidden2": 256,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "warmup_steps": 5,
        "early_patience": 5,
        "label_smoothing": 0.0,
        "augment": False,
    }
    return train_with_config(cfg)
if __name__ == "__main__":
    train_and_eval()

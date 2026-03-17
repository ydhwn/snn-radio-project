import argparse
import os
from .train import train_and_eval, train_with_config
from .report_pack import build_presentation

def parse_stages(s: str):
    items = []
    for part in s.split(","):
        rng_str, ep_str = part.split(":")
        lo_str, hi_str = rng_str.split("-")
        items.append(((float(lo_str), float(hi_str)), int(ep_str)))
    return items

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--presentation", action="store_true")
    p.add_argument("--samples", type=int, default=1200)
    p.add_argument("--symbols", type=int, default=4096)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--hidden1", type=int, default=512)
    p.add_argument("--hidden2", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--stages", type=str, default="20-20:15,15-20:15,10-20:10,5-20:10")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--impaired", action="store_true")
    p.add_argument("--cfo", type=float, default=0.0)
    p.add_argument("--rayleigh", action="store_true")
    p.add_argument("--rayleigh_severity", type=float, default=1.0)
    p.add_argument("--sps", type=int, default=16)
    p.add_argument("--train_impair_prob", type=float, default=0.0)
    p.add_argument("--export", action="store_true", help="Export model to deploy/ folder")
    p.add_argument("--infer", type=str, help="Run inference on this model file")
    p.add_argument("--benchmark", type=str, help="Benchmark this model file against PyTorch baseline")
    args = p.parse_args()
    
    if args.export:
        from .export import export_model
        # Use best checkpoint if exists, else snn_radio.pt
        model_path = "reports/best/snn_radio.pt"
        if not os.path.exists(model_path):
            model_path = "reports/snn_radio.pt"
        print(f"Exporting {model_path}...")
        export_model(model_path)
        return

    if args.infer:
        from .inference import run_inference_demo
        ext = os.path.splitext(args.infer)[1]
        backend = "ts" if ext == ".ts" else "onnx"
        run_inference_demo(args.infer, backend)
        return

    if args.benchmark:
        from .inference import run_benchmark
        ext = os.path.splitext(args.benchmark)[1]
        backend = "ts" if ext == ".ts" else "onnx"
        run_benchmark(args.benchmark, backend)
        return

    if args.sweep:
        from .sweep import run_sweep
        cfg = {
            "samples_per_class": args.samples,
            "symbols_per_sample": args.symbols,
            "batch_size": args.batch,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.wd,
            "stages": parse_stages(args.stages),
            "impaired": args.impaired,
            "cfo": args.cfo,
            "rayleigh": args.rayleigh,
            "rayleigh_severity": args.rayleigh_severity,
            "sps": args.sps,
            "train_impair_prob": args.train_impair_prob,
        }
        best = run_sweep(cfg)
        print(best)
    else:
        cfg = {
            "samples_per_class": args.samples,
            "symbols_per_sample": args.symbols,
            "batch_size": args.batch,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.wd,
            "stages": parse_stages(args.stages),
            "impaired": args.impaired,
            "cfo": args.cfo,
            "rayleigh": args.rayleigh,
            "rayleigh_severity": args.rayleigh_severity,
            "sps": args.sps,
            "train_impair_prob": args.train_impair_prob,
        }
        acc, spikes = train_with_config(cfg)
        if args.presentation:
            path = build_presentation()
            print(path)

if __name__ == "__main__":
    main()

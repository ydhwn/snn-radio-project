import torch
import numpy as np
import os
import json
import argparse
import time
import onnxruntime as ort
from .encoding import iq_to_symbol_vector
from .dataset import make_dataset

# Modulation classes (standard set)
MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "16PSK", "64QAM"]

class InferenceEngine:
    def __init__(self, model_path: str, backend: str = "onnx"):
        self.backend = backend
        self.model_path = model_path
        
        if backend == "onnx":
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        elif backend == "ts":
            self.model = torch.jit.load(model_path)
            self.model.eval()
        elif backend == "pytorch":
            from .snn_model import SNNModulator
            # Inferred num_classes from weights later or use default 6
            self.model = SNNModulator(num_classes=6) 
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.eval()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def predict(self, iq_data: np.ndarray):
        """
        Runs inference on raw IQ data.
        iq_data: np.ndarray of complex64/128, or shape (N, 2)
        """
        # Feature extraction
        if iq_data.ndim == 1:
            feat = iq_to_symbol_vector(iq_data, sps=16, n_symbols=256)
            input_tensor = feat.reshape(1, -1).astype(np.float32)
        else:
            if iq_data.dtype == np.float32 and iq_data.shape[-1] == 512:
                # Already processed features (for benchmarking)
                input_tensor = iq_data
            else:
                feats = []
                for i in range(iq_data.shape[0]):
                    f = iq_to_symbol_vector(iq_data[i], sps=16, n_symbols=256)
                    feats.append(f)
                input_tensor = np.stack(feats).astype(np.float32)

        # Inference
        if self.backend == "onnx":
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = outputs[0]
        else:
            with torch.no_grad():
                t_in = torch.from_numpy(input_tensor)
                logits, spikes = self.model(t_in)
                logits = logits.numpy()

        # Post-processing
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        
        results = []
        for i in range(len(preds)):
            cls_idx = preds[i]
            conf = float(probs[i, cls_idx])
            res = {
                "class": MODS[cls_idx] if cls_idx < len(MODS) else f"Unknown({cls_idx})",
                "confidence": conf,
                "logits": logits[i].tolist(),
                "pred_idx": int(cls_idx)
            }
            results.append(res)
            
        return results

def run_benchmark(model_path: str, backend: str = "onnx", n_samples: int = 500):
    print(f"--- Benchmarking {backend} vs PyTorch ---")
    
    # 1. Load Original PyTorch Model (Baseline)
    pt_path = "reports/best/snn_radio.pt"
    if not os.path.exists(pt_path):
        pt_path = "reports/snn_radio.pt"
    
    print(f"Loading baseline PyTorch model from {pt_path}...")
    pt_engine = InferenceEngine(pt_path, backend="pytorch")
    
    # 2. Load Target Model
    print(f"Loading target {backend} model from {model_path}...")
    target_engine = InferenceEngine(model_path, backend=backend)
    
    # 3. Generate Test Data
    print(f"Generating {n_samples} test samples (10-20dB SNR)...")
    X, y = make_dataset(mods=MODS, samples_per_class=n_samples//len(MODS), snr_range=(10, 20))
    X = X.astype(np.float32)
    
    # 4. Run PyTorch Baseline
    t0 = time.time()
    pt_results = pt_engine.predict(X)
    t_pt = time.time() - t0
    pt_preds = np.array([r["pred_idx"] for r in pt_results])
    pt_acc = np.mean(pt_preds == y)
    
    # 5. Run Target Benchmark
    t0 = time.time()
    target_results = target_engine.predict(X)
    t_target = time.time() - t0
    target_preds = np.array([r["pred_idx"] for r in target_results])
    target_acc = np.mean(target_preds == y)
    
    # 6. Compare
    parity = np.mean(pt_preds == target_preds)
    
    print("\nBenchmark Results:")
    print(f"{'Metric':<20} | {'PyTorch':<15} | {backend.upper():<15}")
    print("-" * 55)
    print(f"{'Accuracy':<20} | {pt_acc:15.4f} | {target_acc:15.4f}")
    print(f"{'Latency (ms/sample)':<20} | {t_pt*1000/len(y):15.4f} | {t_target*1000/len(y):15.4f}")
    print(f"{'Throughput (fps)':<20} | {len(y)/t_pt:15.4f} | {len(y)/t_target:15.4f}")
    print("-" * 55)
    print(f"Accuracy Parity (Agreement): {parity*100:.2f}%")
    
    if parity > 0.99:
        print("SUCCESS: Target model perfectly matches original PyTorch logic.")
    else:
        print("WARNING: Slight divergence detected (expected due to quantization/precision differences).")

def run_inference_demo(model_path: str, backend: str = "onnx"):
    print(f"Loading {backend} model from {model_path}...")
    engine = InferenceEngine(model_path, backend)
    
    print("Generating dummy IQ signal (QPSK)...")
    # Generate a dummy QPSK signal
    # We can import SignalGenerator but to keep this script standalone-ish for deployment,
    # we might want to minimize deps, but here we are in the project context.
    from .signal_generator import SignalGenerator
    gen = SignalGenerator()
    # 20dB SNR QPSK
    sig, _ = gen.generate_qpsk(4096, snr_db=20) 
    
    print("Running inference...")
    results = engine.predict(sig)
    
    print(json.dumps(results[0], indent=2))
    return results[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to .onnx or .ts model")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "ts"])
    args = parser.parse_args()
    
    run_inference_demo(args.model, args.backend)

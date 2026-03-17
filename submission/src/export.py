import torch
import torch.nn as nn
import os
from .snn_model import SNNModulator

def export_model(model_path: str, output_dir: str = "deploy"):
    """
    Exports the trained SNN model to TorchScript and ONNX formats.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    # We need to instantiate the model with the same args. 
    # For simplicity, we'll try to infer or use defaults, but ideally config should be passed.
    # Assuming standard config for now or loading from checkpoint if it has config.
    # We'll use the default args from the class which match the training defaults roughly.
    # Note: If the saved model has different dimensions, load_state_dict will fail, so we might need to be careful.
    # But let's assume standard baseline 1200/4096 config which results in input_dim=512 (vec_len*2 = 256*2 = 512)
    
    # Check if we can determine input dim from the file? No.
    # We will assume the standard input dim 512 (256*2).
    # If the user changed symbols/vec_len, this might break.
    # For a robust industry export, we should save config with model.
    # But for now, we instantiate with defaults.
    
    try:
        model = SNNModulator(input_dim=512, hidden1=512, hidden2=256, num_classes=6) # 6 mods
        # The model architecture in train.py depends on config.
        # Let's try to load state dict.
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Adjust num_classes based on state_dict 'out.weight'
        if 'out.weight' in state_dict:
            out_features = state_dict['out.weight'].shape[0]
            if out_features != 6:
                print(f"Detected {out_features} classes, adjusting model...")
                model = SNNModulator(input_dim=512, hidden1=512, hidden2=256, num_classes=out_features)
        
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Dummy input for tracing/export
    dummy_input = torch.randn(1, 512)

    class SNNExportWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.beta1 = torch.tensor(float(getattr(base.lif1, "beta", 0.9)))
            self.beta2 = torch.tensor(float(getattr(base.lif2, "beta", 0.9)))
            self.th1 = torch.tensor(float(getattr(base.lif1, "threshold", 0.3)))
            self.th2 = torch.tensor(float(getattr(base.lif2, "threshold", 0.3)))
        def forward(self, x):
            h1 = self.base.dropout1(self.base.fc1(x))
            h1 = self.base.bn1(h1)
            mem1 = torch.zeros_like(h1)
            mem1 = self.beta1 * mem1 + h1
            spk1 = (mem1 - self.th1 > 0).to(h1.dtype)
            mem1 = torch.where(spk1 > 0, torch.zeros_like(mem1), mem1)
            h2 = self.base.dropout2(self.base.fc2(spk1))
            h2 = self.base.bn2(h2)
            mem2 = torch.zeros_like(h2)
            mem2 = self.beta2 * mem2 + h2
            spk2 = (mem2 - self.th2 > 0).to(h2.dtype)
            mem2 = torch.where(spk2 > 0, torch.zeros_like(mem2), mem2)
            out = self.base.out(spk2)
            return out, spk2
    
    wrapper = SNNExportWrapper(model).eval()
    xpar = torch.randn(8, 512)
    yb, _ = model(xpar)
    yw, _ = wrapper(xpar)
    d = (yb - yw).abs()
    print(f"Wrapper parity: max={d.max().item():.6f}, mean={d.mean().item():.6f}")

    # 1. TorchScript Export (Tracing)
    try:
        traced_script_module = torch.jit.trace(model, dummy_input, check_trace=False, strict=False)
        ts_path = os.path.join(output_dir, "model.ts")
        traced_script_module.save(ts_path)
        print(f"Exported TorchScript to {ts_path}")
    except Exception as e:
        try:
            traced_script_module = torch.jit.trace(wrapper, dummy_input, check_trace=False, strict=False)
            ts_path = os.path.join(output_dir, "model.ts")
            traced_script_module.save(ts_path)
            print(f"Exported TorchScript to {ts_path}")
        except Exception as ee:
            print(f"Failed to export TorchScript: {ee}")

    # 2. ONNX Export
    try:
        onnx_path = os.path.join(output_dir, "model.onnx")
        torch.onnx.export(wrapper,               # model being run
                          dummy_input,         # model input (or a tuple for multiple inputs)
                          onnx_path,           # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=18,    # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names = ['input'],   # the model's input names
                          output_names = ['output', 'spikes'], # the model's output names
                          dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'},
                                        'spikes' : {0 : 'batch_size'}})
        print(f"Exported ONNX to {onnx_path}")
    except Exception as e:
        print(f"Failed to export ONNX: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        export_model(sys.argv[1])
    else:
        print("Usage: python -m src.export <path_to_model.pt>")

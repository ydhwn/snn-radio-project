import torch

def quantize_tensor(t: torch.Tensor, bits: int = 8):
    if t.dtype not in (torch.float32, torch.float64, torch.float16):
        return t
    qlevels = 2 ** bits - 1
    maxv = torch.max(torch.abs(t))
    if maxv == 0:
        return t.clone()
    s = maxv / qlevels
    q = torch.round(t / s) * s
    return q

def quantize_model(model: torch.nn.Module, bits: int = 8):
    for p in model.parameters():
        with torch.no_grad():
            p.copy_(quantize_tensor(p, bits))
    return model

def prune_by_magnitude(model: torch.nn.Module, pct: float = 0.2):
    mags = torch.cat([p.detach().abs().flatten() for p in model.parameters() if p.requires_grad])
    thr = torch.quantile(mags, pct)
    for p in model.parameters():
        if p.requires_grad:
            with torch.no_grad():
                mask = p.abs() >= thr
                p.copy_(p * mask)
    return model

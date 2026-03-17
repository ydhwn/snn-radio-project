import json
from .train import train_with_config

def run_sweep(base_cfg: dict):
    grid = {
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-5, 5e-6],
        "dropout": [0.1, 0.2],
    }
    best_acc = -1.0
    best_cfg = None
    for lr in grid["lr"]:
        for wd in grid["weight_decay"]:
            for dp in grid["dropout"]:
                cfg = dict(base_cfg)
                cfg.update({"lr": lr, "weight_decay": wd, "dropout": dp})
                # Keep sweep light: reduce samples and epochs
                cfg["samples_per_class"] = min(cfg.get("samples_per_class", 1200), 400)
                cfg["stages"] = [((20.0,20.0), 5)]
                acc, _ = train_with_config(cfg)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = cfg
    out = {"best_acc": best_acc, "best_cfg": best_cfg}
    with open("reports/best_config.json", "w") as f:
        json.dump(out, f, indent=2)
    return out

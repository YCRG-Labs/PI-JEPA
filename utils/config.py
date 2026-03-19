import yaml
import os


class Config:
    def __init__(self, cfg_dict):
        self._cfg = cfg_dict

    def __getitem__(self, key):
        return self._cfg[key]

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def as_dict(self):
        return self._cfg


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = _apply_defaults(cfg)
    _validate(cfg)

    return Config(cfg)


def _apply_defaults(cfg):
    cfg.setdefault("experiment", {})
    cfg.setdefault("training", {})
    cfg.setdefault("model", {})
    cfg.setdefault("predictor", {})
    cfg.setdefault("decoder", {})
    cfg.setdefault("data", {})
    cfg.setdefault("ema", {})
    cfg.setdefault("masking", {})

    cfg["model"].setdefault("encoder", {})

    cfg["experiment"].setdefault("device", "cuda")
    cfg["experiment"].setdefault("precision", "fp32")

    cfg["training"].setdefault("epochs", 100)
    cfg["training"].setdefault("batch_size", 8)
    cfg["training"].setdefault("optim", {})
    cfg["training"]["optim"].setdefault("lr", 1e-4)
    cfg["training"]["optim"].setdefault("weight_decay", 1e-4)
    cfg["training"]["optim"].setdefault("betas", [0.9, 0.999])
    cfg["training"].setdefault("gradient", {})
    cfg["training"]["gradient"].setdefault("clip_norm", None)

    cfg["model"].setdefault("num_predictors", 1)

    cfg["ema"].setdefault("schedule", {})
    cfg["ema"]["schedule"].setdefault("tau_start", 0.996)
    cfg["ema"]["schedule"].setdefault("tau_end", 0.999)

    cfg["masking"].setdefault("context_ratio", 0.5)

    return cfg


def _validate(cfg):
    # REQUIRED FIELDS (UPDATED STRUCTURE)
    required = [
        ("model.encoder", "embed_dim"),
        ("model.encoder", "patch_size"),
        ("model.encoder", "image_size"),
        ("decoder", "image_size"),
        ("decoder", "patch_size"),
        ("data", "num_patches")
    ]

    for section, key in required:
        parts = section.split(".")
        ref = cfg

        for p in parts:
            if p not in ref:
                raise ValueError(f"Missing required config section: {section}")
            ref = ref[p]

        if key not in ref:
            raise ValueError(f"Missing required config: {section}.{key}")
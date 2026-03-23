import os
import torch
import random
import numpy as np


def save_checkpoint(
    path,
    model,
    decoder,
    optimizer,
    scaler,
    epoch,
    step,
    config,
    extra=None
):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    checkpoint = {
        "student_encoder": model.encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "predictors": [p.state_dict() for p in model.predictors],
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "config": config.as_dict() if hasattr(config, "as_dict") else config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "random": random.getstate()
        }
    }

    if extra is not None:
        checkpoint["extra"] = extra

    tmp_path = path + ".tmp"

    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


def load_checkpoint(
    path,
    model,
    decoder,
    optimizer=None,
    scaler=None,
    map_location="cpu"
):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.encoder.load_state_dict(checkpoint["student_encoder"])
    model.target_encoder.load_state_dict(checkpoint["target_encoder"])

    for p, state in zip(model.predictors, checkpoint["predictors"]):
        p.load_state_dict(state)

    decoder.load_state_dict(checkpoint["decoder"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    torch.set_rng_state(checkpoint["rng_state"]["torch"])

    if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])

    np.random.set_state(checkpoint["rng_state"]["numpy"])
    random.setstate(checkpoint["rng_state"]["random"])

    return checkpoint["epoch"], checkpoint["step"], checkpoint.get("extra", None)

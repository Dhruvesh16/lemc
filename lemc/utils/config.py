import os

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def run_name(cfg: dict, seed: int) -> str:
    """Checkpoint/log directory name for this (dataset, lemc on/off, variant,
    seed) combo. 'run_tag' (e.g. 'augmented') must be set in any config whose
    track_store_file differs from the plain per-dataset default, or two
    variants silently collide on the same checkpoint directory."""
    tag = f"_{cfg['run_tag']}" if cfg.get("run_tag") else ""
    lemc_part = "lemc" if cfg.get("use_lemc") else "nolemc"
    return f"{cfg['dataset']}_{lemc_part}{tag}_seed{seed}"


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    defaults = cfg.pop("defaults", [])
    config_dir = os.path.dirname(os.path.abspath(path))

    merged: dict = {}
    for d in defaults:
        default_path = os.path.join(config_dir, f"{d}.yaml")
        merged = _deep_merge(merged, load_config(default_path))
    return _deep_merge(merged, cfg)

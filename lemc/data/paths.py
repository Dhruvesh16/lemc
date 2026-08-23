import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRIKAR_ROOT = os.path.dirname(_REPO_ROOT)

BENCHMARK_REPO = os.environ.get(
    "LEMC_BENCHMARK_REPO",
    os.path.join(_SRIKAR_ROOT, "pedestrian-behaviour/models/PedestrianActionBenchmark"),
)
PIE_DATA_ROOT = os.environ.get(
    "LEMC_PIE_ROOT",
    os.path.join(_SRIKAR_ROOT, "pedestrian-behaviour/data/PIE"),
)
JAAD_DATA_ROOT = os.environ.get(
    "LEMC_JAAD_ROOT",
    os.path.join(_SRIKAR_ROOT, "pedestrian-behaviour/data/JAAD"),
)

TRACK_STORE_DIR = os.path.join(_REPO_ROOT, "data_cache")


def track_store_path(cfg: dict) -> str:
    """cfg may set 'track_store_file' to point at a variant (e.g. the
    detector-augmented store) -- defaults to the plain per-dataset store."""
    filename = cfg.get("track_store_file", f"{cfg['dataset']}_track_store.pkl")
    return os.path.join(TRACK_STORE_DIR, filename)

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

_DEFAULT_JAAD = os.path.join(_REPO_ROOT, "data_external", "JAAD")
if not os.path.isdir(os.path.join(_DEFAULT_JAAD, "annotations")):
    _DEFAULT_JAAD = os.path.join(_SRIKAR_ROOT, "pedestrian-behaviour/data/JAAD")
JAAD_DATA_ROOT = os.environ.get("LEMC_JAAD_ROOT", _DEFAULT_JAAD)

TRACK_STORE_DIR = os.path.join(_REPO_ROOT, "data_cache")


def track_store_path(cfg: dict) -> str:
    """cfg may set 'track_store_file' to point at a variant (e.g. the
    detector-augmented store) -- defaults to the plain per-dataset store."""
    filename = cfg.get("track_store_file", f"{cfg['dataset']}_track_store.pkl")
    return os.path.join(TRACK_STORE_DIR, filename)


def dataset_image_root(cfg: dict) -> str:
    if cfg.get("dataset") == "jaad":
        return os.path.join(JAAD_DATA_ROOT, "images")
    return os.path.join(PIE_DATA_ROOT, "images")


def frame_image_path(image_root: str, scene_id: str, frame: int, dataset: str) -> str:
    """Return on-disk path for a frame image (PIE: .jpg nested; JAAD: .png flat)."""
    if dataset == "jaad":
        return os.path.join(image_root, scene_id, f"{frame:05d}.png")
    set_id, video_id = scene_id.split("/", 1)
    return os.path.join(image_root, set_id, video_id, f"{frame:05d}.jpg")

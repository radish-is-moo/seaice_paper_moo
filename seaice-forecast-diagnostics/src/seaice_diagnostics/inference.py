"""Read-only ver2 notebook model loading and resumable forecast inference."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .protocol import (
    GRID_SHAPE,
    HORIZON_DAYS,
    atomic_savez_compressed,
    complete_initializations,
    validate_prediction_archive,
)


from .config import (
    REPO as PROJECT_ROOT,
    ARCHIVES as OUTPUT_ROOT,
    NOTEBOOKS as NOTEBOOK_ROOT,
    CACHE as CACHE_DIR,
    ERA5 as ERA5_SAMPLE,
    model_path,
)

VALID_DATES = pd.date_range("1979-01-01", "2025-12-31", freq="D")
HISTORY_LEN = 15
VARIABLES = ("t2m", "d2m", "u10", "v10", "msl", "sst", "sic", "sit")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    notebook: Path
    checkpoint: Path
    statistics: Path
    cell_index: int
    definition_names: tuple[str, ...]
    class_name: str


MODEL_SPECS = {
    "cnn": ModelSpec(
        key="cnn",
        display_name="CNN",
        notebook=NOTEBOOK_ROOT / "train_cnn.ipynb",
        checkpoint=model_path("cnn", "checkpoint"),
        statistics=model_path("cnn", "statistics"),
        cell_index=6,
        definition_names=("ResidualConvBlock", "SimpleDirectSICCNN"),
        class_name="SimpleDirectSICCNN",
    ),
    "unet": ModelSpec(
        key="unet",
        display_name="U-Net",
        notebook=NOTEBOOK_ROOT / "train_unet.ipynb",
        checkpoint=model_path("unet", "checkpoint"),
        statistics=model_path("unet", "statistics"),
        cell_index=6,
        definition_names=("UNetConvBlock", "UNetDown", "UNetUp", "SimpleDirectSICUNet"),
        class_name="SimpleDirectSICUNet",
    ),
    "gnn": ModelSpec(
        key="gnn",
        display_name="GNN",
        notebook=NOTEBOOK_ROOT / "train_gnn.ipynb",
        checkpoint=model_path("gnn", "checkpoint"),
        statistics=model_path("gnn", "statistics"),
        cell_index=6,
        definition_names=("DailyGridOnlyGNNEncoder", "DailyGridOnlyGNNForecastModel"),
        class_name="DailyGridOnlyGNNForecastModel",
    ),
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_definitions(source: str, names: set[str]) -> str:
    """Return only named top-level class/function definitions from source."""
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    missing = names.difference(getattr(node, "name", "") for node in selected)
    if missing:
        raise ValueError(f"Notebook definitions missing: {sorted(missing)}")
    return "\n\n".join(ast.unparse(node) for node in selected) + "\n"


def prediction_relative_path(model: str, target_year: int, init_date: object) -> Path:
    key = normalize_models([model])[0]
    return (
        Path("predictions")
        / key
        / str(int(target_year))
        / f"{pd.Timestamp(init_date).date().isoformat()}.npz"
    )


def reference_relative_path(target_year: int, init_date: object) -> Path:
    return (
        Path("references")
        / str(int(target_year))
        / f"{pd.Timestamp(init_date).date().isoformat()}.npz"
    )


def estimate_storage(n_initializations: int, n_models: int) -> dict[str, int | float]:
    field_bytes = (
        HORIZON_DAYS * GRID_SHAPE[0] * GRID_SHAPE[1] * np.dtype(np.float32).itemsize
    )
    prediction_bytes = int(n_initializations) * int(n_models) * field_bytes
    reference_bytes = int(n_initializations) * field_bytes
    total = prediction_bytes + reference_bytes
    return {
        "n_initializations": int(n_initializations),
        "n_models": int(n_models),
        "one_field_bytes": int(field_bytes),
        "prediction_bytes": int(prediction_bytes),
        "reference_bytes": int(reference_bytes),
        "total_raw_bytes": int(total),
        "total_raw_gib": float(total / 1024**3),
    }


def normalize_models(models: Iterable[str]) -> list[str]:
    values = [str(model).strip().lower().replace("-", "") for model in models]
    if values == ["all"] or "all" in values:
        return ["cnn", "unet", "gnn"]
    aliases = {
        "cnn": "cnn",
        "unet": "unet",
        "u_net": "unet",
        "gnn": "gnn",
        "graphsage": "gnn",
    }
    normalized: list[str] = []
    for value in values:
        if value not in aliases:
            raise ValueError(f"Unknown model: {value}")
        key = aliases[value]
        if key not in normalized:
            normalized.append(key)
    return normalized


class DailyCache:
    """Small LRU reader for the exact 15-day input and 30-day target cache."""

    def __init__(self, cache_dir: Path = CACHE_DIR, recent_size: int = 64):
        self.cache_dir = Path(cache_dir)
        self.recent_size = int(recent_size)
        self._recent: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    @staticmethod
    def key(date_like: object) -> str:
        return pd.Timestamp(date_like).strftime("%Y%m%d")

    def load_day(self, date_like: object) -> dict[str, np.ndarray]:
        key = self.key(date_like)
        if key in self._recent:
            value = self._recent.pop(key)
            self._recent[key] = value
            return value
        path = self.cache_dir / f"{key}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            value = {
                "daily": np.asarray(archive["daily"], dtype=np.float32),
                "sic_full": np.asarray(archive["sic_full"], dtype=np.float32),
            }
        if value["daily"].shape != (len(VARIABLES), *GRID_SHAPE):
            raise ValueError(
                f"Unexpected daily cache shape in {path}: {value['daily'].shape}"
            )
        self._recent[key] = value
        while len(self._recent) > self.recent_size:
            self._recent.popitem(last=False)
        return value

    def sample(
        self, init_date: object, mean: np.ndarray, std: np.ndarray
    ) -> dict[str, object]:
        init = pd.Timestamp(init_date).normalize()
        history_dates = pd.date_range(
            init - pd.Timedelta(days=HISTORY_LEN - 1), init, freq="D"
        )
        target_dates = pd.date_range(
            init + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D"
        )
        history = [self.load_day(date) for date in history_dates]
        targets = [self.load_day(date) for date in target_dates]
        daily = np.stack([day["daily"] for day in history], axis=0).astype(np.float32)
        daily = (daily - mean[None, :, None, None]) / std[None, :, None, None]
        x = np.nan_to_num(
            daily.reshape(HISTORY_LEN * len(VARIABLES), *GRID_SHAPE),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        current = np.asarray(history[-1]["sic_full"], dtype=np.float32)
        reference = np.stack([day["sic_full"] for day in targets], axis=0).astype(
            np.float32
        )
        return {
            "x": x,
            "current_sic": current,
            "reference": reference,
            "target_dates": target_dates,
        }


def build_grid_edges(num_lat: int = GRID_SHAPE[0], num_lon: int = GRID_SHAPE[1]):
    """Create the notebook's directed periodic eight-neighbour graph."""
    import torch

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    columns = np.arange(num_lon, dtype=np.int64)
    for row in range(num_lat):
        src = row * num_lon + columns
        for neighbour_row in (row - 1, row, row + 1):
            if neighbour_row < 0 or neighbour_row >= num_lat:
                continue
            for offset in (-1, 0, 1):
                if neighbour_row == row and offset == 0:
                    continue
                dst = neighbour_row * num_lon + ((columns + offset) % num_lon)
                sources.append(src)
                targets.append(dst)
    return torch.as_tensor(
        np.stack([np.concatenate(sources), np.concatenate(targets)]), dtype=torch.long
    )


class ModelRuntime:
    def __init__(self, spec: ModelSpec, device: str | None = None):
        import torch
        import torch.nn as nn

        self.spec = spec
        self.torch = torch
        torch.backends.cudnn.benchmark = True
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        for path in (spec.notebook, spec.checkpoint, spec.statistics):
            if not path.exists():
                raise FileNotFoundError(path)
        notebook = json.loads(spec.notebook.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][spec.cell_index]["source"])
        namespace: dict[str, object] = {"torch": torch, "nn": nn}
        if spec.key == "gnn":
            import torch.nn.functional as F
            from torch_geometric.nn import HeteroConv, Linear, SAGEConv

            namespace.update(
                {
                    "F": F,
                    "HeteroConv": HeteroConv,
                    "Linear": Linear,
                    "SAGEConv": SAGEConv,
                }
            )
        exec(
            compile(
                extract_definitions(source, set(spec.definition_names)),
                str(spec.notebook),
                "exec",
            ),
            namespace,
        )
        model_class = namespace[spec.class_name]
        if spec.key == "cnn":
            model = model_class(in_channels=120, horizon=30, hidden=32)
        elif spec.key == "unet":
            model = model_class(in_channels=120, horizon=30, base_channels=16)
        else:
            model = model_class(in_grid=120, horizon=30, hidden=64, num_layers=3)
        state = torch.load(spec.checkpoint, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        with np.load(spec.statistics, allow_pickle=False) as stats:
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.asarray(stats["std"], dtype=np.float32)
        if self.mean.shape != (8,) or self.std.shape != (8,):
            raise ValueError(
                f"Unexpected channel statistics shape for {spec.display_name}"
            )
        self.edge_index_dict = None
        if spec.key == "gnn":
            edge_path = OUTPUT_ROOT / "cache" / "grid_edges_101x1440.pt"
            edge_path.parent.mkdir(parents=True, exist_ok=True)
            if edge_path.exists():
                edge = torch.load(edge_path, map_location="cpu", weights_only=True)
            else:
                edge = build_grid_edges()
                part = Path(str(edge_path) + ".part")
                torch.save(edge, part)
                part.replace(edge_path)
            self.edge_index_dict = {("grid", "flows_to", "grid"): edge.to(self.device)}

    def predict(self, sample: dict[str, object]) -> np.ndarray:
        torch = self.torch
        x = np.asarray(sample["x"], dtype=np.float32)
        current = np.asarray(sample["current_sic"], dtype=np.float32)
        if x.shape != (120, *GRID_SHAPE):
            raise ValueError(
                f"Model input shape must be (120, 101, 1440), got {x.shape}"
            )
        with torch.inference_mode():
            if self.spec.key in {"cnn", "unet"}:
                x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
                current_tensor = (
                    torch.from_numpy(current).unsqueeze(0).unsqueeze(0).to(self.device)
                )
                output = self.model(x_tensor, current_tensor)["pred_full"].squeeze(0)
                prediction = output.detach().cpu().numpy()
            else:
                x_tensor = torch.from_numpy(x.reshape(120, -1).T).to(self.device)
                current_tensor = torch.from_numpy(current.reshape(-1, 1)).to(
                    self.device
                )
                delta = self.model(x_tensor, self.edge_index_dict)["delta_full"]
                output = torch.clamp(current_tensor + delta, 0.0, 1.0)
                prediction = (
                    output.detach().cpu().numpy().T.reshape(HORIZON_DAYS, *GRID_SHAPE)
                )
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (HORIZON_DAYS, *GRID_SHAPE):
            raise ValueError(
                f"Model output shape must be (30, 101, 1440), got {prediction.shape}"
            )
        return prediction


def _reference_is_valid(path: Path, init_date: object) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            reference = np.asarray(archive["reference"])
            stored_init = str(np.asarray(archive["init_date"]).item())
            leads = np.asarray(archive["lead_days"], dtype=int)
            targets = np.asarray(archive["target_dates"]).astype(str)
        return bool(
            reference.shape == (HORIZON_DAYS, *GRID_SHAPE)
            and reference.dtype.itemsize >= 4
            and reference.dtype.kind == "f"
            and stored_init == pd.Timestamp(init_date).date().isoformat()
            and np.array_equal(leads, np.arange(1, 31))
            and len(targets) == 30
        )
    except Exception:
        return False


def write_reference(path: Path, sample: dict[str, object], init_date: object) -> None:
    reference = np.asarray(sample["reference"], dtype=np.float32)
    targets = pd.DatetimeIndex(sample["target_dates"])
    atomic_savez_compressed(
        path,
        reference=reference,
        init_date=np.asarray(pd.Timestamp(init_date).date().isoformat()),
        target_dates=np.asarray(targets.strftime("%Y-%m-%d"), dtype=str),
        lead_days=np.arange(1, 31, dtype=np.int16),
        shape=np.asarray(reference.shape, dtype=np.int32),
        dtype=np.asarray(str(reference.dtype)),
        source=np.asarray(str(CACHE_DIR)),
    )


def run_inference(
    *,
    models: Iterable[str],
    years: Iterable[int],
    output_root: Path = OUTPUT_ROOT,
    force: bool = False,
    init_dates: Iterable[object] | None = None,
    device: str | None = None,
) -> dict[str, object]:
    selected_models = normalize_models(models)
    cases = complete_initializations(years)
    if init_dates is not None:
        requested = {pd.Timestamp(value).normalize() for value in init_dates}
        cases = cases[cases["init_date"].isin(requested)].reset_index(drop=True)
        missing = requested.difference(set(cases["init_date"]))
        if missing:
            raise ValueError(
                f"Requested init dates are outside complete windows: {sorted(missing)}"
            )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache = DailyCache()
    storage = estimate_storage(len(cases), len(selected_models))
    print(json.dumps({"storage_estimate": storage}, indent=2), flush=True)
    completed = skipped = 0
    for model_key in selected_models:
        runtime = ModelRuntime(MODEL_SPECS[model_key], device=device)
        print(
            f"[{runtime.spec.display_name}] checkpoint loaded on {runtime.device}",
            flush=True,
        )
        for case_index, case in enumerate(cases.itertuples(index=False), start=1):
            prediction_path = output_root / prediction_relative_path(
                model_key, case.target_year, case.init_date
            )
            existing = validate_prediction_archive(
                prediction_path,
                expected_model=runtime.spec.display_name,
                expected_init_date=case.init_date,
            )
            if existing["valid"] and not force:
                skipped += 1
                continue
            sample = cache.sample(case.init_date, runtime.mean, runtime.std)
            reference_path = output_root / reference_relative_path(
                case.target_year, case.init_date
            )
            if force or not _reference_is_valid(reference_path, case.init_date):
                write_reference(reference_path, sample, case.init_date)
            prediction = runtime.predict(sample)
            ocean = np.isfinite(np.asarray(sample["reference"], dtype=np.float32))
            if (
                not ocean.any()
                or float(prediction[ocean].min()) < 0.0
                or float(prediction[ocean].max()) > 1.0
            ):
                raise ValueError(
                    f"Predicted SIC range failed for {runtime.spec.display_name} {case.init_date.date()}"
                )
            target_dates = pd.DatetimeIndex(sample["target_dates"])
            atomic_savez_compressed(
                prediction_path,
                prediction=prediction,
                model=np.asarray(runtime.spec.display_name),
                model_key=np.asarray(runtime.spec.key),
                init_date=np.asarray(case.init_date.date().isoformat()),
                target_dates=np.asarray(target_dates.strftime("%Y-%m-%d"), dtype=str),
                lead_days=np.arange(1, 31, dtype=np.int16),
                shape=np.asarray(prediction.shape, dtype=np.int32),
                dtype=np.asarray(str(prediction.dtype)),
                notebook_sha256=np.asarray(sha256(runtime.spec.notebook)),
                checkpoint_sha256=np.asarray(sha256(runtime.spec.checkpoint)),
                statistics_sha256=np.asarray(sha256(runtime.spec.statistics)),
            )
            validation = validate_prediction_archive(
                prediction_path,
                expected_model=runtime.spec.display_name,
                expected_init_date=case.init_date,
            )
            if not validation["valid"]:
                raise ValueError(f"Written prediction failed validation: {validation}")
            completed += 1
            print(
                f"[{runtime.spec.display_name}] {case_index}/{len(cases)} init={case.init_date.date()} saved",
                flush=True,
            )
        del runtime
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return {
        "models": selected_models,
        "years": sorted({int(year) for year in years}),
        "n_cases": int(len(cases)),
        "completed": int(completed),
        "skipped": int(skipped),
        "storage_estimate": storage,
    }

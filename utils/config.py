"""Load and validate the VisionGuard YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from data.layout import SUPPORTED_CATEGORIES


class ConfigError(ValueError):
    """The project configuration cannot be used safely."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    num_workers: int
    batch_size: int


@dataclass(frozen=True)
class DataConfig:
    root: Path
    categories: tuple[str, ...]
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    weights: str
    layers: tuple[str, ...]
    neighborhood: int


@dataclass(frozen=True)
class MemoryConfig:
    coreset_ratio: float
    max_bank_size: int
    preselect_max: int
    patches_per_image: int
    query_chunk: int
    bank_chunk: int


@dataclass(frozen=True)
class ScoringConfig:
    score_neighbors: int
    heatmap_smooth_sigma: float


@dataclass(frozen=True)
class EvaluationConfig:
    max_pixels: int
    save_examples: int
    image_threshold: float | None = None


@dataclass(frozen=True)
class OutputConfig:
    directory: Path


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig
    memory: MemoryConfig
    scoring: ScoringConfig
    evaluation: EvaluationConfig
    output: OutputConfig


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a mapping.")
    return _parse(raw)


def _parse(raw: dict) -> ProjectConfig:
    root = project_root()
    runtime = _section(raw, "runtime")
    data = _section(raw, "data")
    model = _section(raw, "model")
    memory = _section(raw, "memory")
    scoring = _section(raw, "scoring")
    evaluation = _section(raw, "evaluation")
    output = _section(raw, "output")

    device = str(runtime.get("device", ""))
    if device != "cpu":
        raise ConfigError(
            "runtime.device must be 'cpu'. VisionGuard does not use CUDA."
        )
    workers = _positive_int(runtime, "num_workers", minimum=0)
    if workers != 0:
        raise ConfigError("runtime.num_workers must be 0 on this CPU setup.")
    batch_size = _positive_int(runtime, "batch_size", minimum=1)
    if batch_size not in (1, 2):
        raise ConfigError("runtime.batch_size must be 1 or 2.")

    categories = tuple(str(item) for item in _sequence(data, "categories"))
    if not categories:
        raise ConfigError("data.categories must list at least one category.")
    unknown = [item for item in categories if item not in SUPPORTED_CATEGORIES]
    if unknown:
        allowed = ", ".join(SUPPORTED_CATEGORIES)
        raise ConfigError(
            f"Unsupported categories {unknown}. Phase 1 allows: {allowed}."
        )
    if len(set(categories)) != len(categories):
        raise ConfigError("data.categories contains duplicates.")

    image_size = _positive_int(data, "image_size", minimum=32)
    if image_size % 32 != 0:
        raise ConfigError("data.image_size must be a multiple of 32.")
    mean = _float_triple(data, "mean")
    std = _float_triple(data, "std")
    if any(value == 0 for value in std):
        raise ConfigError("data.std values must be non-zero.")

    backbone = str(model.get("backbone", ""))
    if backbone != "resnet18":
        raise ConfigError("model.backbone must be resnet18 in phase 1.")
    weights = str(model.get("weights", ""))
    if weights not in ("imagenet", "none"):
        raise ConfigError("model.weights must be 'imagenet' or 'none'.")
    layers = tuple(str(item) for item in _sequence(model, "layers"))
    if not layers or any(layer not in ("layer2", "layer3") for layer in layers):
        raise ConfigError("model.layers must be a non-empty subset of layer2 and layer3.")
    if len(set(layers)) != len(layers):
        raise ConfigError("model.layers contains duplicates.")
    neighborhood = _positive_int(model, "neighborhood", minimum=1)
    if neighborhood % 2 == 0:
        raise ConfigError("model.neighborhood must be a positive odd integer.")

    ratio = float(memory.get("coreset_ratio", 0))
    if not 0 < ratio <= 1:
        raise ConfigError("memory.coreset_ratio must be in the interval (0, 1].")

    return ProjectConfig(
        seed=_positive_int(raw, "seed", minimum=0),
        runtime=RuntimeConfig(device=device, num_workers=workers, batch_size=batch_size),
        data=DataConfig(
            root=_resolve_dir(root, data.get("root"), "data.root"),
            categories=categories,
            image_size=image_size,
            mean=mean,
            std=std,
        ),
        model=ModelConfig(
            backbone=backbone,
            weights=weights,
            layers=layers,
            neighborhood=neighborhood,
        ),
        memory=MemoryConfig(
            coreset_ratio=ratio,
            max_bank_size=_positive_int(memory, "max_bank_size", minimum=1),
            preselect_max=_positive_int(memory, "preselect_max", minimum=1),
            patches_per_image=_positive_int(memory, "patches_per_image", minimum=1),
            query_chunk=_positive_int(memory, "query_chunk", minimum=1),
            bank_chunk=_positive_int(memory, "bank_chunk", minimum=1),
        ),
        scoring=ScoringConfig(
            score_neighbors=_positive_int(scoring, "score_neighbors", minimum=1),
            heatmap_smooth_sigma=_non_negative_float(scoring, "heatmap_smooth_sigma"),
        ),
        evaluation=EvaluationConfig(
            max_pixels=_positive_int(evaluation, "max_pixels", minimum=1000),
            save_examples=_positive_int(evaluation, "save_examples", minimum=0),
            image_threshold=_optional_non_negative(evaluation, "image_threshold"),
        ),
        output=OutputConfig(
            directory=_resolve_dir(root, output.get("directory"), "output.directory")
        ),
    )


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid config section: {name}")
    return value


def _sequence(section: dict, name: str) -> list:
    value = section.get(name)
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list.")
    return value


def _positive_int(section: dict, name: str, minimum: int) -> int:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}.")
    return value


def _optional_non_negative(section: dict, name: str) -> float | None:
    if name not in section or section[name] is None:
        return None
    return _non_negative_float(section, name)


def _non_negative_float(section: dict, name: str) -> float:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigError(f"{name} must be a number >= 0.")
    return float(value)


def _float_triple(section: dict, name: str) -> tuple[float, float, float]:
    value = section.get(name)
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"{name} must be a list of three numbers.")
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must contain numbers.") from exc
    return numbers  # type: ignore[return-value]


def _resolve_dir(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a path string.")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def config_to_dict(config: ProjectConfig) -> dict:
    """JSON-ready copy of the settings used for a run."""
    return {
        "seed": config.seed,
        "runtime": {
            "device": config.runtime.device,
            "num_workers": config.runtime.num_workers,
            "batch_size": config.runtime.batch_size,
        },
        "data": {
            "root": str(config.data.root),
            "categories": list(config.data.categories),
            "image_size": config.data.image_size,
            "mean": list(config.data.mean),
            "std": list(config.data.std),
        },
        "model": {
            "backbone": config.model.backbone,
            "weights": config.model.weights,
            "layers": list(config.model.layers),
            "neighborhood": config.model.neighborhood,
        },
        "memory": {
            "coreset_ratio": config.memory.coreset_ratio,
            "max_bank_size": config.memory.max_bank_size,
            "preselect_max": config.memory.preselect_max,
            "patches_per_image": config.memory.patches_per_image,
            "query_chunk": config.memory.query_chunk,
            "bank_chunk": config.memory.bank_chunk,
        },
        "scoring": {
            "score_neighbors": config.scoring.score_neighbors,
            "heatmap_smooth_sigma": config.scoring.heatmap_smooth_sigma,
        },
        "evaluation": {
            "max_pixels": config.evaluation.max_pixels,
            "save_examples": config.evaluation.save_examples,
            "image_threshold": config.evaluation.image_threshold,
        },
        "output": {"directory": str(config.output.directory)},
    }


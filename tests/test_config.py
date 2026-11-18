"""Configuration safety checks."""

from pathlib import Path

import pytest

from utils.config import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "default.yaml"


def test_default_config_is_cpu_bounded():
    config = load_config(DEFAULT)
    assert config.runtime.device == "cpu"
    assert config.runtime.num_workers == 0
    assert config.runtime.batch_size == 1
    assert config.data.image_size == 128
    assert config.data.categories == ("bottle", "hazelnut")
    assert config.model.backbone == "resnet18"
    assert config.model.layers == ("layer2", "layer3")
    assert config.memory.preselect_max >= 1


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("device: cpu", "device: cuda"),
        ("num_workers: 0", "num_workers: 2"),
        ("batch_size: 1", "batch_size: 8"),
        ("    - bottle", "    - cable"),
    ],
)
def test_default_config_rejects_unsafe_edits(tmp_path, old, new):
    path = tmp_path / "edited.yaml"
    path.write_text(DEFAULT.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)

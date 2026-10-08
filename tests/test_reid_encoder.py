"""Scenarios from specs/reid-model: Custom_ReID loads both weight formats and embeds every box."""

import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from ultralytics.utils.ops import xywh2xyxy

from safetrajai.reid import encoders
from safetrajai.reid.encoders import Custom_ReID
from safetrajai.reid.training.model import ReIDModel, create_backbone, save_weights

DETS = np.array([[100.0, 80.0, 40.0, 60.0, 0], [300.0, 200.0, 120.0, 50.0, 1], [50.0, 400.0, 30.0, 30.0, 2]])


class LegacyNet(nn.Module):
    def forward(self, images, camids):
        return images.mean(dim=(2, 3)).repeat(1, 3)


def frame() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def new_weight(tmp_path):
    model = ReIDModel(create_backbone("resnet18", 64, pretrained=False), num_classes=5)
    path = tmp_path / "class_reid_weights" / "run" / "model.pt"
    save_weights(path, model, "resnet18", 64, "synthetic")
    return path


def test_one_l2_feature_per_box(new_weight) -> None:
    encoder = Custom_ReID(str(new_weight))
    feats = encoder(frame(), DETS)
    assert len(feats) == 3
    for feat in feats:
        assert feat.shape == (512,)
        assert np.linalg.norm(feat) == pytest.approx(1.0, abs=1e-5)


def test_new_format_does_not_need_pickle_classes(new_weight, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "utils.model", None)
    monkeypatch.setitem(sys.modules, "safetrajai.reid.model", None)
    checkpoint = torch.load(new_weight, weights_only=True)
    assert checkpoint["manifest_version"] == "synthetic"
    assert not any(k.startswith("classifier.") for k in checkpoint["state_dict"])
    Custom_ReID(str(new_weight))


def record_crops(monkeypatch) -> list:
    calls = []
    real = encoders.save_one_box

    def recording(xyxy, im, **kwargs):
        crop = real(xyxy, im, **kwargs)
        calls.append((xyxy.clone(), kwargs, crop))
        return crop

    monkeypatch.setattr(encoders, "save_one_box", recording)
    return calls


def test_crop_is_tight_box(new_weight, monkeypatch) -> None:
    calls = record_crops(monkeypatch)
    image = frame()
    Custom_ReID(str(new_weight))(image, DETS)
    expected = xywh2xyxy(torch.from_numpy(DETS[:, :4]))
    assert len(calls) == 3
    for (xyxy, kwargs, _), want in zip(calls, expected):
        assert torch.equal(xyxy, want)
        assert kwargs["gain"] == 1.0 and kwargs["pad"] == 0
    assert np.array_equal(calls[0][2], image[50:110, 80:120])


def test_legacy_crop_keeps_gain_and_pad(new_weight, monkeypatch) -> None:
    calls = record_crops(monkeypatch)
    Custom_ReID(str(new_weight), legacy_crop=True)(frame(), DETS)
    assert len(calls) == 3
    for _, kwargs, _ in calls:
        assert kwargs["gain"] == 1.02 and kwargs["pad"] == 10
    height, width = calls[0][2].shape[:2]
    assert height > 60 and width > 40


def test_legacy_full_model_pickle_still_loads(tmp_path) -> None:
    path = tmp_path / "class_reid_weights" / "legacy" / "best_full_model.pt"
    path.parent.mkdir(parents=True)
    torch.save(LegacyNet(), path)
    encoder = Custom_ReID(str(path))
    assert encoder.config is None
    assert isinstance(encoder.model, LegacyNet)
    feats = encoder(frame(), DETS)
    assert len(feats) == 1 and feats[0].shape == (9,)

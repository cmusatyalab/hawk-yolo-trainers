# SPDX-FileCopyrightText: 2022-2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for config modules."""

from __future__ import annotations

import pytest


class TestYOLOConfig:
    def test_default_values(self) -> None:
        from hawk_yolo_trainers.yolo.config import YOLOTrainerConfig, YOLOModelConfig

        yolo_config = YOLOTrainerConfig()
        assert yolo_config.train_batch_size == 16
        assert yolo_config.image_size == 640
        assert yolo_config.initial_model_epochs == 30

        yolo_model_config = YOLOModelConfig()
        assert yolo_model_config.input_size == 480
        assert yolo_model_config.test_batch_size == 32


class TestYOLORadarConfig:
    def test_default_values(self) -> None:
        from hawk_yolo_trainers.yolo_radar.config import (
            YOLOTrainerConfig,
            YOLOModelConfig,
        )

        radar_config = YOLOTrainerConfig()
        assert radar_config.train_batch_size == 16
        assert radar_config.image_size == 256
        assert radar_config.initial_model_epochs == 30

        radar_model_config = YOLOModelConfig()
        assert radar_model_config.input_size == 480
        assert radar_model_config.test_batch_size == 32

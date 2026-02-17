# SPDX-FileCopyrightText: 2022-2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for trainer modules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestYOLOTrainer:
    def test_trainer_creation(self) -> None:
        from hawk_yolo_trainers.yolo.config import YOLOTrainerConfig
        from hawk_yolo_trainers.yolo.trainer import YOLOTrainer

        config = YOLOTrainerConfig()
        context = MagicMock()

        trainer = YOLOTrainer(config, context)
        assert trainer is not None


class TestYOLOModel:
    def test_model_creation(self) -> None:
        from hawk_yolo_trainers.yolo.config import YOLOModelConfig
        from hawk_yolo_trainers.yolo.model import YOLOModel

        config = YOLOModelConfig()
        context = MagicMock()
        context.class_list = MagicMock()
        context.retriever = MagicMock()

        with pytest.raises(AssertionError):
            model = YOLOModel(config, context, MagicMock(), 0)


class TestYOLOTrainerRadar:
    def test_trainer_creation(self) -> None:
        from hawk_yolo_trainers.yolo_radar.config import YOLOTrainerConfig
        from hawk_yolo_trainers.yolo_radar.trainer import YOLOTrainerRadar

        config = YOLOTrainerConfig()
        context = MagicMock()

        trainer = YOLOTrainerRadar(config, context)
        assert trainer is not None


class TestYOLOModelRadar:
    def test_model_creation(self) -> None:
        from hawk_yolo_trainers.yolo_radar.config import YOLOModelConfig
        from hawk_yolo_trainers.yolo_radar.model import YOLOModelRadar

        config = YOLOModelConfig()
        context = MagicMock()
        context.class_list = MagicMock()
        context.retriever = MagicMock()

        with pytest.raises(AssertionError):
            model = YOLOModelRadar(config, context, MagicMock(), 0)

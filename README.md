# hawk-yolo-trainers

YOLO and YOLO-Radar trainers for Hawk - GPL-3.0-only

## Overview

This package provides YOLOv5-based training capabilities (both standard and radar variants) for the Hawk computer vision system. It depends on `cmuhawk` (the original Hawk package) and implements pluggable trainer interfaces.

## Components

### YOLO Trainer
Standard YOLOv5 object detection trainer.

### YOLO-Radar Trainer  
YOLOv5 variant with radar-specific functionality.

## Installation

```bash
pip install hawk-yolo-trainers
```

Or install from source:

```bash
cd hawk-yolo-trainers
pip install .
```

## Usage

After installation, both trainers will be automatically registered with Hawk's plugin system.

### YOLO Configuration

```yaml
model_trainer:
  name: yolo
  config:
    train_batch_size: 16
    image_size: 640
```

### YOLO-Radar Configuration

```yaml
model_trainer:
  name: yolo_radar
  config:
    train_batch_size: 16
    image_size: 256
```

## Dependencies

- `cmuhawk>=0.1.0` - Original Hawk package
- `torch>=2.2,<3.0`
- `torchvision>=0.17,<0.18`
- Other dependencies listed in `pyproject.toml`

## License

This package is licensed under GPL-3.0-only due to its dependency on YOLOv5 (Ultralytics).

## Development

```bash
uv sync
pytest
```

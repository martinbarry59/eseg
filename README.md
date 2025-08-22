# surreal-events

Event-based depth / segmentation utilities and models (experimental).  

## Features
- ConvLSTM-based depth estimation model for event streams
- MobileNetV2 feature encoder with UNet-like decoder
- Event voxelization and augmentation utilities
- Real-time camera viewers (Metavision / DAVIS) with overlay visualization
- Mixed perceptual + edge loss utilities (LPIPS + Sobel)

## Installation
```bash
pip install eseg
```
(Once published to PyPI.)

For development:
```bash
git clone https://github.com/youruser/eseg.git
cd eseg
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
pip install -e .[dev,viewer]
```

## Quick Start
```python
import torch
from eseg.models import ConvLSTM
# TODO: usage example after final API stabilizes
```

## Live Stream
```bash
python -m eseg.live_stream
```

## Testing
```bash
pytest
```

## License
MIT. See `LICENSE`.

## Disclaimer
Research code; APIs may change before 1.0.0.

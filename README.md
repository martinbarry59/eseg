# Human Instance Segmentation using Evn

![](pedestrians.gif)


## Features
- ConvLSTM-based depth estimation model for event streams
- MobileNetV2 feature encoder with UNet-like decoder
- Event voxelization and augmentation utilities
- Real-time camera viewers (Metavision / DAVIS) with overlay visualization
- Mixed perceptual + edge loss utilities (LPIPS + Sobel)

## Requirements
We implemented the dataviewers on Both [dvprocessing](https://dv-processing.inivation.com/master/index.html) and [metavisionSDK](https://docs.prophesee.ai/stable/get_started/get_started_python.html)


Please install [metavisionSDK](https://docs.prophesee.ai/stable/get_started/get_started_python.html) for Prophesee live camera. 

⚠️ Warning DO NOT forget to set metavisionsdk in your python path especially for windows!

And / Or

[dvprocessing](https://dv-processing.inivation.com/master/index.html) for Davis Cameras

For hdf5 file we use [metavisionSDK](https://docs.prophesee.ai/stable/get_started/get_started_python.html)  aedat files can be processed using [dvprocessing](https://dv-processing.inivation.com/master/index.html)

Pleas, before installing eseg install a GPU enabled pytorch here: https://pytorch.org/get-started/locally/


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
⚠️ Warning if you work on a virtualenvironment you will need to copy your global sdk library to your local environment
```
cp -r path/to/your/metavisionsdk/metavision_* <path/to/your/virtualenv/python<yourversion>/site-packages/
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

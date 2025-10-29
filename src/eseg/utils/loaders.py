import torch
from typing import Optional
from eseg.models.ConvLSTM import EConvlstm
from eseg.models.EfficientConvLSTM import EfficientConvLSTM 
from eseg.config import checkpoint_path
import sys
import os
import urllib.request
PRETRAINED_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/martinbarry59/eseg/main/src/checkpoints/CONVLSTM.pth"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _download_checkpoint(url: str, dest: str) -> bool:
    """Download a checkpoint file with a simple progress indicator.

    Returns True if successful, False otherwise.
    """
    # try:
    print(f"Downloading pretrained checkpoint from {url} to {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out_file:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8192
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                print(
                    f"\rDownloading checkpoint: {percent:5.1f}% ({downloaded}/{total} bytes)",
                    end="",
                )
    print("\nDownload complete.")
    return True
def load_model(model_type: str = "full", checkpoint_path: str = "./checkpoints") -> torch.nn.Module:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    
    with torch.no_grad():

        network = "CONVLSTM"
        checkpoint_file: Optional[str] = None

        if checkpoint_path:
            checkpoint_file = f"{checkpoint_path}/{network + ("_LIGHT" if model_type == "light" else "_FULL")}.pth"
            
            model = EConvlstm(model_type=network, skip_lstm=True) if model_type == "full" else EfficientConvLSTM(model_type=network)
            print(f"Loading checkpoint from {checkpoint_file}")
            try:
                model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            except Exception:
                print("Checkpoint not found or failed to load.")
                # Ask user if they want to download the pretrained weights
                answer = input("Download pretrained checkpoint now? [y/N]: ").strip().lower()
                if answer == "y":
                    if _download_checkpoint(PRETRAINED_CHECKPOINT_URL, checkpoint_file):
                        try:
                            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
                            print("Pretrained weights loaded.")
                        except Exception as e:
                            print(
                                f"Downloaded file could not be loaded: {e}. Continuing uninitialized."
                            )
                    else:
                        print("Download failed. Continuing with randomly initialized weights.")
                else:
                    print("Continuing without pretrained weights.")
        else:
            model = EConvlstm(model_type=network, skip_lstm=True)
    return model.to(device)


def load_metavision(verbose=False, continue_on_fail=False):
    metavision = None
    try:
        # sys.path.append("/usr/lib/python3/dist-packages")
        import metavision_sdk_stream as metavision  # type: ignore
        print("Metavision SDK successfully loaded.")
    except Exception as e:
        if verbose:
            print(
                "Metavision SDK not found. Cannot read .hdf5 files or connect to Prophesee cameras."
            )
            print(
                "if Metavision is installed, ensure  it is installed locally "
            )
            print(e)
        if not continue_on_fail:
            sys.exit(1)
    return metavision


def load_dv_processing(verbose=False, continue_on_fail=False):
    dv = None
    try:
        import dv_processing as dv  # type: ignore
    except Exception as e:
        if verbose:
            print("dv_processing not found. Cannot read Raw or aedat files.")
            print(e)
        if not continue_on_fail:
            sys.exit(1)
    return dv
def load_h5py(verbose=False, continue_on_fail=False):
    h5py = None
    try:
        import h5py  # type: ignore
    except Exception as e:
        if verbose:
            print("h5py not found. Cannot read .h5 files.")
            print(e)
        if not continue_on_fail:
            sys.exit(1)
    return h5py
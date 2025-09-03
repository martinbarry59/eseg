"""Live streaming entry point for event camera inference.

Attempts to connect to a Prophesee (Metavision) camera first, then falls
back to a DAVIS device using `dv_processing`. Loads a ConvLSTM-based
model from checkpoint and visualizes predicted depth maps in real time.
"""

from eseg.models.ConvLSTM import EConvlstm
from eseg.config import checkpoint_path

import cv2
import torch

import sys
from typing import Optional
from eseg.utils.functions import _download_checkpoint
import argparse

# Mock URL for pretrained model (replace with real one later)
PRETRAINED_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/martinbarry59/eseg/main/src/checkpoints/CONVLSTM.pth"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    with torch.no_grad():

        network = "CONVLSTM"
        checkpoint_file: Optional[str] = None

        if checkpoint_path:
            checkpoint_file = f"{checkpoint_path}/CONVLSTM.pth"
            checkpoint_file = f"{checkpoint_path}/model_epoch_3_CONVLSTM.pth"
            model = EConvlstm(model_type=network, skip_lstm=True)
            print(f"Loading checkpoint from {checkpoint_file}")
            try:
                model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            except Exception:
                print("Checkpoint not found or failed to load.")
                # Ask user if they want to download the pretrained weights
                answer = input(
                    "Download pretrained checkpoint now? [y/N]: "
                ).strip().lower()
                if answer == "y":
                    if _download_checkpoint(PRETRAINED_CHECKPOINT_URL, checkpoint_file):
                        try:
                            model.load_state_dict(
                                torch.load(checkpoint_file, map_location=device)
                            )
                            print("Pretrained weights loaded.")
                        except Exception as e:
                            print(f"Downloaded file could not be loaded: {e}. Continuing uninitialized.")
                    else:
                        print("Download failed. Continuing with randomly initialized weights.")
                else:
                    print("Continuing without pretrained weights.")
        else:
            model = EConvlstm(model_type=network, skip_lstm=True)
    return model
model = load_model()
# Attempt dynamic camera backend selection


def parse_args():

    parser = argparse.ArgumentParser(
        description="Live event stream depth inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-event-file",
        default=None,
        help="Path to input event file (RAW or HDF5). If omitted, a live camera is used.",
    )
    parser.add_argument(
        "--slice-time-ms",
        type=int,
        default=100,
        help="Time window for each event slice in milliseconds.",
    )
    parser.add_argument(
        "-f",
        "--filter-size-ms",
        type=int,
        default=20,
        help="Size of the temporal noise filter in milliseconds.",
    )
    parser.add_argument(
        "-s",
        "--save-output-video",
        type=str,
        default=None,
        help="Save output video to file, gif or mp4.",
        
    )
    ## test if save path is valid MP4 or GIF
    args = parser.parse_args()
    if args.save_output_video is not None:
        if not (args.save_output_video.endswith('.mp4') or args.save_output_video.endswith('.gif')):
            parser.error("Output video file must be of type .mp4 or .gif")
            sys.exit(1)
    return args
def load_metavision(verbose=False, continue_on_fail=False):
    metavision = None
    try:
        sys.path.append("/usr/lib/python3/dist-packages")
        import metavision_sdk_stream as metavision   # type: ignore
    except Exception as e:
        if verbose:
            print("Metavision SDK not found. Cannot read .hdf5 files or connect to Prophesee cameras.")
            print("if Metavision is installed, ensure  it is installed locally or in /usr/lib/python3/dist-packages")
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
def from_file(input_event_file):
    if  input_event_file.endswith('.hdf5'):
        metavision = load_metavision(verbose=True, continue_on_fail=False)
        try:
            camera = metavision.Camera.from_file(input_event_file)
            camera_type = 'Prophesee'
        except Exception as e:
            print(f"Failed to load events from file: {input_event_file}")
            print(e)
            sys.exit(1)
    else:
        dv = load_dv_processing(verbose=True, continue_on_fail=False)
        try:
            camera = dv.io.MonoCameraRecording(input_event_file)
            camera_type = 'DAVIS'
        except Exception as e:
            print(f"Failed to load events from file: {input_event_file}")
            print(e)
            sys.exit(1)
    return camera, camera_type
def from_camera():
    print("No input file provided. Trying to open a camera.")
    metavision = load_metavision(verbose=True, continue_on_fail=True)
    camera = None
    if metavision is not None:
        try:
            camera = metavision.Camera.from_first_available()
            camera_type = 'Prophesee'
        except Exception as e:
            print(e)
            print("Failed to find a Prophesee camera. Moving to dv_processing.")

    if camera is None:
        print("No Prophesee camera found. Trying to open a DAVIS camera using dv_processing.")
        dv = load_dv_processing(verbose=True, continue_on_fail=False)
        if dv is not None:
            try:
                camera = dv.io.camera.open()
                camera_type = 'DAVIS'
            except Exception as e:
                print(e)
                print("Failed to find a DAVIS camera.")
                sys.exit(1)
    return camera, camera_type
def run(input_event_file = None,slice_time_ms: int = 100, filter_size_ms: int = 20, save_video: str = None):
    if input_event_file:
        camera, camera_type = from_file(input_event_file)
    else:
        camera, camera_type = from_camera()
    if camera_type == 'Prophesee':
        from eseg.utils.dataviewers import dataviewerprophesee as dataviewer
    else:
        from eseg.utils.dataviewers import dataviewerdavis as dataviewer
    viewer = dataviewer(camera, slice_time_ms=slice_time_ms, filter_size_ms=filter_size_ms, video_save_path=save_video)

    model.to(device)
    viewer.setModel(model)
    viewer.run()
    # if viewer.se is not None:
    #     print("Writing video to file... please wait")
    #     viewer.video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_event_file=args.input_event_file,
        slice_time_ms=args.slice_time_ms,
        filter_size_ms=args.filter_size_ms,
        save_video=args.save_output_video
    )

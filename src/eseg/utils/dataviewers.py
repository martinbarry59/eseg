"""Realtime event camera visualization utilities.

Supports DAVIS (Inivation / Prophesee compatible) and Prophesee specific
interfaces to stream events, convert them to tensors, run a model, and
overlay predictions for display.
"""

import torch
import cv2
from datetime import timedelta
import numpy as np
import sys
from typing import Optional, List
import imageio
import time
# NOTE: System-specific SDK imports (Metavision) follow. These may raise
# ImportError on systems without the vendor libraries installed.




class dataviewer:
    """Base viewer handling event accumulation and model inference.

    Subclasses define how to acquire events from device-specific SDKs.
    """

    def __init__(self, camera, video_save_path: Optional[str] = None, verbose: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width, self.height = None, None
        self.events: Optional[torch.Tensor] = None
        self.instant_events = None
        self.window_name = "Event Frame"
        self.window = cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.model = None
        self.slicer = None
        self.filter = None
        self.reader = None
        self.inference_times  = {"preprocess": [], "inference": [], "postprocess": []}
        self.video_save_path = video_save_path
        self.img = None
        self.predictions = None
        self.verbose = verbose
        if self.video_save_path is not None:
            ## Mp4 of gif depending on file extension
            if self.video_save_path.endswith(".mp4"):
                self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = None
                self.save_type = "mp4"
            elif self.video_save_path.endswith(".gif"):
                self.video_writer = imageio.get_writer(
                    self.video_save_path, fps=20, mode="I", loop=0
                )
                self.save_type = "gif"
        self.camera = camera

    def setModel(self, model):
        self.model = model
        self.model.width = self.width
        self.model.height = self.height
        self.model.eval()
        
    
        self.model.to(self.device)
        
        # Warm up the model with a dummy input to optimize CUDA kernels
        # self._warmup_model()
    def estimateInferenceTime(self) -> float:
        for key in self.inference_times:
            times = self.inference_times[key][10:]
            if len(times) > 0:
                avg_time = sum(times) / len(times)
                print(f"Average {key} time: {avg_time*1000:.2f} ms over {len(times)} runs, max {max(times)*1000:.2f} ms, min {min(times)*1000:.2f} ms, std {np.std(times)*1000:.2f} ms")
            if key == "inference":
                for kkey in self.model.model_inference_times:
                    times = self.model.model_inference_times[kkey][10:]
                    if len(times) > 0:
                        avg_time = sum(times) / len(times)
                        print(f"  - {key} sub-step {kkey}: {avg_time*1000:.2f} ms over {len(times)} runs, max {max(times)*1000:.2f} ms, min {min(times)*1000:.2f} ms, std {np.std(times)*1000:.2f} ms")
    def extractEvents(self, events, reversex: bool = False,normalize_p = True) -> torch.Tensor:
        """Convert structured event arrays to tensor (t,x,y,p) - optimized version."""
        # Pre-allocate tensor for better memory efficiency
        n_events = len(events)
        if n_events == 0:
            return torch.zeros(0, 4, device=self.device)
            
        events_tensor = torch.empty(n_events, 4, device=self.device, dtype=torch.float32)
        
        # Direct tensor operations for speed
        xs = self.width - events["x"] - 1 if reversex else events["x"]
        ys = events["y"]
        if not normalize_p:
            ps = events["polarity"] if reversex else events["p"]
        else:
            ps = 2 * events["polarity"] - 1 if reversex else 2 * events["p"] - 1
        ts = events["timestamp"] if reversex else events["t"]
        
        # Normalize timestamps in-place
        ts_min = ts.min()
        ts = ts - ts_min
        
        # Fill tensor directly
        events_tensor[:, 0] = torch.from_numpy(ts.copy()).to(self.device)
        events_tensor[:, 1] = torch.from_numpy(xs.astype(np.float32, copy=False)).to(self.device)
        events_tensor[:, 2] = torch.from_numpy(ys.astype(np.float32, copy=False)).to(self.device)
        events_tensor[:, 3] = torch.from_numpy(ps.copy().astype(np.float32)).to(self.device)
        
        return events_tensor

    def predict(self):
        with torch.no_grad():
            # Clone the events tensor to avoid CUDA graphs overwriting issues
            events_input = self.events.unsqueeze(0)
            seq_events = [events_input]  # Single frame list
            ## auto-cast for inference if model supports it
            predictions, _, seq_events = self.model(seq_events)

        return predictions, seq_events

    def mergePredictions(self, img, predictions):
        pred = cv2.applyColorMap((predictions * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.5, pred, 0.5, 0)
        img = cv2.resize(img, (640, 320), interpolation=cv2.INTER_LINEAR)
        return img

    def showImage(self, img):
        if self.video_save_path is not None:
            if self.save_type == "gif":
                self.video_writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                if self.video_writer is None:
                    height, width, _ = img.shape
                    self.video_writer = cv2.VideoWriter(
                        self.video_save_path, self.fourcc, 20.0, (width, height)
                    )
                self.video_writer.write(img)
        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            self.saveVideo()
            if self.verbose:
                self.estimateInferenceTime()
            sys.exit(0)
    def retrieveEvents(self, events):
        self.instant_events = events

    def processEvents(self, events, reversex: bool = False, normalize_p: bool = True):
        start_time = time.time()
        events_tensor = self.extractEvents(events.numpy().copy(), reversex=reversex, normalize_p=normalize_p) if not isinstance(events, np.ndarray) == True else self.extractEvents(events, reversex=reversex, normalize_p=normalize_p)
        self.events = events_tensor
        preprocess_time = time.time() - start_time
        self.inference_times["preprocess"].append(preprocess_time)
        start_time = time.time()
        predictions, seq_events = self.predict()
        inference_time = time.time() - start_time
        self.inference_times["inference"].append(inference_time)
        start_time = time.time()
        img = torch.sum(seq_events[0][0], dim=0)
        if self.img is None:
            self.img = img.cpu().pin_memory()
        else:
            self.img.copy_(img, non_blocking=True)
        if self.predictions is None:
            self.predictions = predictions[0,0].cpu().pin_memory()
        else:
            self.predictions.copy_(predictions[0,0], non_blocking=True)
        
        
        self.img[self.img != 0] = 255
        rgb_img = cv2.cvtColor(self.img.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        merged_img = self.mergePredictions(rgb_img, self.predictions.numpy())
        postprocess_time = time.time() - start_time
        self.inference_times["postprocess"].append(postprocess_time)
        self.showImage(merged_img)
        
    def run(self):  # interface method
        raise NotImplementedError

    def step(self, slice):
        raise NotImplementedError
    def saveVideo(self):
        if self.video_save_path is not None:
            if self.save_type == "gif":
                
                self.video_writer.close()
            else:
                self.video_writer.release()
    

class dataviewerdavis(dataviewer):
    """Viewer for DAVIS / Inivation style cameras using dv_processing."""

    def __init__(
        self,
        camera,
        slice_time_ms: int = 100,
        filter_size_ms: int = 20,
        video_save_path: Optional[str] = None,
        verbose: bool = False,
    ):
        print("Using dv_processing for event processing")
        import dv_processing as dv

        super().__init__(camera, video_save_path=video_save_path, verbose=verbose)
        self.width, self.height = self.camera.getEventResolution()
        self.slicer = dv.EventStreamSlicer()
        self.filter = dv.noise.BackgroundActivityNoiseFilter(
            (self.width, self.height),
            backgroundActivityDuration=timedelta(milliseconds=filter_size_ms),
        )
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=slice_time_ms), self.retrieveEvents)

    def run(self):
    
        while self.camera.isRunning():
            with torch.no_grad():
                self.instant_events = None
                events = self.camera.getNextEventBatch()
                self.step(events)

    def step(self, slice):
        if slice is None or len(slice) == 0:
            return
        self.slicer.accept(slice)
        if self.instant_events is None or len(self.instant_events) == 0:
            return
        self.filter.accept(self.instant_events)
        filtered_events = self.filter.generateEvents()
        if filtered_events is None or len(filtered_events) == 0:
            return
        self.processEvents(filtered_events, reversex=True)


class dataviewerprophesee(dataviewer):
    """Viewer for Prophesee devices using Metavision SDK."""

    def __init__(
        self,
        camera,
        slice_time_ms: int = 100,
        filter_size_ms: int = 20,
        video_save_path: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(camera, video_save_path=video_save_path, verbose=verbose)
        print("Using metavision_sdk_stream for event processing")
        from metavision_sdk_stream import CameraStreamSlicer, SliceCondition  # type: ignore
        from metavision_sdk_cv import ActivityNoiseFilterAlgorithm  # type: ignore
        from metavision_sdk_base import EventCDBuffer  # type: ignore
        self.buffer = EventCDBuffer()
        self.width, self.height = self.camera.width(), self.camera.height()

        slice_condition = SliceCondition.make_n_us(int(slice_time_ms * 1000))
        self.slicer = CameraStreamSlicer(self.camera.move(), slice_condition=slice_condition)
        self.activity_filter = ActivityNoiseFilterAlgorithm(
            self.width, self.height, filter_size_ms * 1000
        )

    def run(self):
        for slice in self.slicer:
            with torch.no_grad():
                self.step(slice)

    def step(self, slice):
        self.activity_filter.process_events(slice.events, self.buffer)
        self.processEvents(self.buffer, reversex=False)
class dataviewerh5py(dataviewer):
    """Viewer for H5PY event files using h5py."""

    def __init__(
        self,
        camera,
        slice_time_ms: int = 100,
        filter_size_ms: int = 20,
        video_save_path: Optional[str] = None,
        verbose: bool = False,
    ):
        import h5py

        super().__init__(camera, video_save_path=video_save_path, verbose=verbose)
        print("Using h5py for event processing")
        
        self.events_data = torch.Tensor(camera["vids"][:,:4])
        self.events_data = np.array(
            list(
                zip(
                    self.events_data[:, 0].numpy(),
                    self.events_data[:, 1].numpy(),
                    self.events_data[:, 2].numpy(),
                    self.events_data[:, 3].numpy(),
                )
            ),
            dtype=[("t", "f4"), ("x", "u2"), ("y", "u2"), ("p", "i1")],
        )
        self.width = camera["width"][()] if "width" in camera else 346
        self.height =  camera["height"][()] if "height" in camera else 260
        self.num_events = self.events_data.shape[0]
        self.slice_time = slice_time_ms / 1000.0
        self.current_time = 0
        
    def run(self):
        while len(self.events_data) > 0:
            with torch.no_grad():
                self.step()

    def step(self):

        times = self.events_data["t"]
        indices = (times > self.current_time) * (times <= (self.current_time + self.slice_time))
        events = self.events_data[indices]
        ## give name to columns
        ## pop indices from events
        self.events_data = self.events_data[~indices]
        self.current_time += self.slice_time
        if events is None or len(events) == 0:
            return

        
        self.processEvents(events, reversex=False, normalize_p=False)
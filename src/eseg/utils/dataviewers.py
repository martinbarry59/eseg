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

# NOTE: System-specific SDK imports (Metavision) follow. These may raise
# ImportError on systems without the vendor libraries installed.




class dataviewer:
    """Base viewer handling event accumulation and model inference.

    Subclasses define how to acquire events from device-specific SDKs.
    """

    def __init__(self, camera, video_save_path: Optional[str] = None):
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
        self.video_save_path = video_save_path

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

    def extractEvents(self, events, reversex: bool = False) -> torch.Tensor:
        """Convert structured event arrays to tensor (t,x,y,p)."""
        xs = self.width - events["x"] - 1 if reversex else events["x"]
        ys = events["y"]
        ps = 2 * events["polarity"] - 1 if reversex else 2 * events["p"] - 1
        ts = events["timestamp"] if reversex else events["t"]
        ts = ts - ts.min()
        events_tensor = torch.stack(
        (
            torch.tensor(ts.copy()),
            torch.tensor(xs.astype(np.int32, copy=False)).float(),
            torch.tensor(ys.astype(np.int32, copy=False)).float(),
            torch.tensor(ps.copy()).float(),
        ),
        dim=1,
)
        return events_tensor.to(self.device)

    def predict(self):
        with torch.no_grad():

            seq_events = self.events.unsqueeze(0).unsqueeze(0).to(self.device)
            predictions, _, seq_events = self.model(seq_events)

        return predictions, seq_events

    def mergePredictions(self, img, predictions):
        pred = predictions[0, 0].detach().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
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
            sys.exit(0)
    def retrieveEvents(self, events):
        self.instant_events = events

    def processEvents(self, events, reversex: bool = False):
        events_tensor = self.extractEvents(events.numpy().copy(), reversex=reversex)
        self.events = events_tensor
        predictions, seq_events = self.predict()
        self.predictions = predictions.clone()
        img = np.sum(seq_events[0][0].detach().cpu().numpy(), axis=0).astype(np.uint8)
        img[img != 0] = 255
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        merged_img = self.mergePredictions(img, predictions)
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
    ):
        print("Using dv_processing for event processing")
        import dv_processing as dv

        super().__init__(camera, video_save_path=video_save_path)
        self.width, self.height = self.camera.getEventResolution()
        self.slicer = dv.EventStreamSlicer()
        self.filter = dv.noise.BackgroundActivityNoiseFilter(
            (self.width, self.height),
            backgroundActivityDuration=timedelta(milliseconds=filter_size_ms),
        )
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=slice_time_ms), self.retrieveEvents)

    def run(self):
        while self.camera.isRunning():
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
    ):
        super().__init__(camera, video_save_path=video_save_path)
        print("Using metavision_sdk_stream for event processing")
        from metavision_sdk_stream import CameraStreamSlicer, SliceCondition  # type: ignore
        from metavision_sdk_cv import ActivityNoiseFilterAlgorithm  # type: ignore
        from metavision_sdk_base import EventCDBuffer  # type: ignore
        self.buffer = EventCDBuffer()
        self.width, self.height = self.camera.width(), self.camera.height()

        slice_condition = SliceCondition.make_n_us(slice_time_ms * 1000)
        self.slicer = CameraStreamSlicer(self.camera.move(), slice_condition=slice_condition)
        self.activity_filter = ActivityNoiseFilterAlgorithm(
            self.width, self.height, filter_size_ms * 1000
        )

    def run(self):
        for slice in self.slicer:
            self.step(slice)

    def step(self, slice):
        self.activity_filter.process_events(slice.events, self.buffer)
        self.processEvents(self.buffer, reversex=False)

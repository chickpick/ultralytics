# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial
from pathlib import Path
import numpy as np
import torch

from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start_OLD(predictor: object, persist: bool = False) -> None:
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool): Whether to persist the trackers if they already exist.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.

    Examples:
        Initialize trackers for a predictor object:
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes.
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool): Whether to persist the trackers if they already exist.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.

    Examples:
        Initialize trackers for a predictor object:
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    trackers = []
    # Change this to initialize 4 trackers
    for _ in range(4):  # Instead of using predictor.dataset.bs
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)

    # Set the trackers and video paths
    predictor.trackers = trackers
    predictor.vid_path = [None] * 4  # Adjusted to match the number of trackers

def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.
    
    This version excludes class 1 from being tracked but keeps its detections.
    """
    path, im0s = predictor.batch[:2]
    
    is_obb = predictor.args.task == "obb"  # Check if the task is oriented bounding boxes (OBBs)
    is_stream = predictor.dataset.mode == "stream"  # Check if the dataset is in streaming mode

    trackers = [predictor.trackers[cam_idx] for cam_idx in range(4)]  # One tracker per camera

    for i in range(len(im0s)):
        # Get the tracker corresponding to the current camera
        tracker = trackers[i]
        # Get the tracker for the current image
        #tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name  # Define the video path for saving results
        
        # Reset the tracker if the video path has changed
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        # Get detections (either boxes or OBBs)
        det = predictor.results[i].obb if is_obb else predictor.results[i].boxes

        # Skip processing if there are no detections
        if len(det) == 0:
            continue

        # Separate detections by class (assuming class 1 is for wings)
        cls_labels = det.cls.cpu().numpy()  # Get class labels as a NumPy array
        mask_chick = cls_labels != 1  # Mask for chicks (not class 1)
        mask_wing = cls_labels == 1    # Mask for wings (class 1)

        # Apply the masks to filter the detections
        det_chick_track = det[mask_chick]  # Detections for chicks
        det_wings = det[mask_wing]          # Detections for wings

        tracks = tracker.update(det_chick_track.cpu().numpy(), im0s[i])  # Update tracker

        # Debugging prints for shapes and contents
        # print(f"Class 1 Wings (shape: {det_wings.shape}):\n{det_wings}\n")
        # print(f"Class 0 Chicks to send to tracking (shape: {det_chick_track.shape}):\n{det_chick_track}\n")

        if len(tracks) == 0:
            continue
        
        # Debugging print for tracking results
        # print(f"Chick after Tracking shape: {tracks.shape}\nOutput of Tracking: {tracks}\n")

        # Process wing detections
        if det_wings.shape[0] > 0:  # Check if there are any wings detected
            det_wings_data = det_wings.data  # Get the raw data tensor for wings
            wings_device = det_wings_data.device  # Get the device for wings data

            # Create a tensor of zeros to add a new column for the updated wings detections
            zero_column = torch.zeros(det_wings.shape[0], 1, device=wings_device)

            # Concatenate the zero column at the appropriate position in the wings detections
            updated_wings = torch.cat((det_wings_data[:, :4], zero_column, det_wings_data[:, 4:]), dim=1)

            # Debugging print for updated wings shape
            # print(f"Wings shape after update: {updated_wings.shape}")

            # Convert tracks to the same device as updated_wings
            tracks_device = torch.as_tensor(tracks[:, :-1], device=wings_device)

            # Combine chick and wing detections
            combined_detections = torch.cat((tracks_device, updated_wings), dim=0)
            # Debugging print for combined detections
            # print(f"Combined Detections Shape: {combined_detections.shape}")
        else:
            combined_detections = torch.as_tensor(tracks[:, :-1])  # Use just chick detections if no wings are detected

        # Update the predictor results with the combined detections
        update_args = {"obb" if is_obb else "boxes": combined_detections}
        predictor.results[i].update(**update_args)

def on_predict_postprocess_end_old(predictor: object, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    path, im0s = predictor.batch[:2]

    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (predictor.results[i].obb if is_obb else predictor.results[i].boxes).cpu().numpy()
        if len(det) == 0:
            continue
        tracks = tracker.update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)

def register_tracker(model: object, persist: bool) -> None:
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))

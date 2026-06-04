# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.custom_sort3D.kalman import CustomSORT3DKalmanBoxTracker
from trackers.core.sort.utils import (
    get_alive_trackers,
    # get_iou_matrix,
    get_iou_matrix_3d,
    get_distance_matrix_3d
)
from agros.main_post_processing import Detections3D

class CustomSORT3DTracker(BaseTracker):
    """In SORT, object tracking begins with high-confidence detections fed into a
    Kalman filter framework assuming uniform motion for state prediction across frames.
    Association occurs via IoU-based costs in the Hungarian algorithm, enforcing a
    threshold to filter weak matches and initialize new identities. Tracks persist only
    with consistent associations, terminating quickly to avoid erroneous propagation.
    This detection-driven approach underscores the importance of upstream detector
    performance in achieving competitive multi-object tracking results. Over time, SORT
    has become a cornerstone for evaluating motion-based improvements in the field.

    SORT's standout strength is its real-time capability, processing hundreds of frames
    per second while maintaining accuracy comparable to more complex offline methods. It
    performs well in controlled environments with reliable detections, minimizing
    computational demands. However, without mechanisms for re-identification, it incurs
    frequent identity switches during object reappearances post-occlusion. The linear
    motion assumption limits effectiveness in non-linear paths, such as those in sports
    or wildlife tracking. Ultimately, SORT's efficiency is offset by its sensitivity to
    environmental complexities, necessitating hybrid extensions for broader
    applicability.

    Args:
        lost_track_buffer: `int` specifying number of frames to buffer when a
            track is lost. Increasing this value enhances occlusion handling but
            may increase ID switching for similar objects.
        frame_rate: `float` specifying video frame rate in frames per second.
            Used to scale the lost track buffer for consistent tracking across
            different frame rates.
        track_activation_threshold: `float` specifying minimum detection
            confidence to create new tracks. Higher values reduce false
            positives but may miss low-confidence objects.
        minimum_consecutive_frames: `int` specifying number of consecutive
            frames before a track is considered valid. Before reaching this
            threshold, tracks are assigned `tracker_id` of `-1`.
        minimum_iou_threshold: `float` specifying IoU threshold for associating
            detections to existing tracks. Higher values require more overlap.
    """

    tracker_id = "sort"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold

        # Active trackers
        self.trackers: list[CustomSORT3DKalmanBoxTracker] = []

    def _get_associated_indices(
        self, iou_matrix: np.ndarray, detection_boxes: np.ndarray, maximize: bool=True
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        Associate detections to trackers based on IOU

        Args:
            iou_matrix: IOU cost matrix.
            detection_boxes: Detected bounding boxes in the form [x1, y1, x2, y2].

        Returns:
            Matched indices, unmatched trackers, unmatched detections.
        """
        matched_indices = []
        unmatched_trackers = set(range(len(self.trackers)))
        unmatched_detections = set(range(len(detection_boxes)))

        if len(self.trackers) > 0 and len(detection_boxes) > 0:
            # Find optimal assignment using scipy.optimize.linear_sum_assignment.
            # Note that it uses a a modified Jonker-Volgenant algorithm with no
            # initialization instead of the Hungarian algorithm as mentioned in the
            # SORT paper.
            row_indices, col_indices = linear_sum_assignment(iou_matrix, maximize=maximize)
            y_min = detection_boxes[:,1].min()
            y_max = detection_boxes[:,1].max()

            for row, col in zip(row_indices, col_indices):
                if maximize:
                    if iou_matrix[row, col] >= self.minimum_iou_threshold:
                        matched_indices.append((row, col))
                        unmatched_trackers.remove(row)
                        unmatched_detections.remove(col)
                else:
                    if iou_matrix[row, col] <= self.minimum_iou_threshold:
                        matched_indices.append((row, col))
                        unmatched_trackers.remove(row)
                        unmatched_detections.remove(col)
                
                ## plant specific
                # After linear_sum_assignment, relax unmatched tolerance at boundaries
                # boundary_y_margin = 0.50  # plants within this of the row ends may legitimately vanish
                # dy = iou_matrix[row, col]
                # det_y_pos = detection_boxes[col, 1]
                # at_boundary = (det_y_pos < y_min + boundary_y_margin or 
                #             det_y_pos > y_max - boundary_y_margin)
                # threshold = self.minimum_iou_threshold * 2 if at_boundary else self.minimum_iou_threshold
                # if dy <= threshold:
                #     matched_indices.append((row, col))
                #     unmatched_trackers.remove(row)
                #     unmatched_detections.remove(col)
        return matched_indices, unmatched_trackers, unmatched_detections

    def _spawn_new_trackers(
        self,
        confidences: np.ndarray | None,
        detection_boxes: np.ndarray,
        unmatched_detections: set[int],
    ) -> None:
        for detection_idx in unmatched_detections:
            if (
                confidences is None
                or detection_idx >= len(confidences)
                or confidences[detection_idx] >= self.track_activation_threshold
            ):
                self.trackers.append(
                    CustomSORT3DKalmanBoxTracker(detection_boxes[detection_idx])
                )

    def update(self, detections: Detections3D, u: np.ndarray | None) -> Detections3D:
        """Update tracker state with new detections and return tracked objects.
        Performs Kalman filter prediction, IoU-based association, and initializes
        new tracks for unmatched high-confidence detections.

        Args:
            detections: `Detections3D` containing bounding boxes with shape
                `(N, 4)` in `(x_min, y_min, x_max, y_max)` format and optional
                confidence scores.

        Returns:
            `Detections3D` with `tracker_id` assigned for each detection.
                Unmatched or immature tracks have `tracker_id` of `-1`.
        """
        if len(self.trackers) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        detection_boxes = (
            detections.xyzxyz if len(detections) > 0 else np.array([]).reshape(0, 6)
        )

        for tracker in self.trackers:
            tracker.predict(u=u)

        # iou_matrix = get_iou_matrix_3d(self.trackers, detection_boxes)
        # matched_indices, _, unmatched_detections = self._get_associated_indices(
        #     iou_matrix, detection_boxes, 
        # )

        
        cost_matrix = get_distance_matrix_3d(self.trackers, detection_boxes)
        y_trackers = np.array([t.state[1]for t in self.trackers])
        y_det = detection_boxes[:,1]
        margin = 0.1
        y_track_cor, y_det_cor = 0, 0
        if y_trackers.size>0:
            if y_trackers.max() - margin > y_det.max():
                ## do not match this tracker since it is moved to the next row
                cost_matrix[y_trackers.argmin(),:] = self.minimum_iou_threshold+1
                y_track_cor = y_track_cor-1

            # if y_det.max() - margin > y_trackers.max():
                ## this is not a problem because likely y_det is shifted
            # if y_trackers.min() + margin < y_det.min():
                ## this is not a problem because likely y_tracker is shifted
            if y_det.min() + margin < y_trackers.min():
                ## Likely a new plant has emerged, reduce matching cost
                cost_matrix[:,y_det.argmin()] = self.minimum_iou_threshold+1
                y_det_cor = y_det_cor - 1

        print(cost_matrix.shape, cost_matrix.shape[0] + y_track_cor, cost_matrix.shape[1] + y_det_cor)

        # cost_matrix = get_row_matching_cost(self.trackers, detection_boxes)
        matched_indices, _, unmatched_detections = self._get_associated_indices(
            cost_matrix, detection_boxes, maximize=False
        )


        # Update matched trackers and record the det_idx -> tracker mapping
        matched_tracker_for_det: dict[int, CustomSORT3DKalmanBoxTracker] = {}
        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])
            matched_tracker_for_det[col] = self.trackers[row]

        self._spawn_new_trackers(
            detections.confidence, detection_boxes, unmatched_detections
        )

        self.trackers = get_alive_trackers(
            self.trackers,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update,
        )

        # Build tracker_ids from the recorded mapping (no deepcopy, no re-IoU)
        tracker_ids = np.full(len(detection_boxes), -1, dtype=int)
        for det_idx, tracker in matched_tracker_for_det.items():
            if tracker.number_of_successful_updates >= self.minimum_consecutive_frames:
                if tracker.tracker_id == -1:
                    tracker.tracker_id = CustomSORT3DKalmanBoxTracker.get_next_tracker_id()
                tracker_ids[det_idx] = tracker.tracker_id

        detections.tracker_id = tracker_ids
        return detections

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.trackers = []
        CustomSORT3DKalmanBoxTracker.count_id = 0


def get_row_matching_cost(trackers, detection_boxes, max_cost=1e6):
    """
    Cost = |y_center_tracker - y_center_detection|
    Add a large penalty for reversing order (cross-matches),
    which prevents ID switches from non-monotonic assignments.
    """
    N = len(trackers)
    M = len(detection_boxes)
    cost = np.full((N, M), max_cost)
    
    # Sort both by Y to enforce order-preserving matching
    tracker_y = np.array([t.state[1] for t in trackers]).reshape(-1)  # y center
    det_y = detection_boxes[:, 1] #(detection_boxes[:, 1] + detection_boxes[:, 4]) / 2.0
    
    tracker_order = np.argsort(tracker_y)
    det_order = np.argsort(det_y)
    
    # Only allow matches within a Y distance threshold; 
    # penalise order-reversing matches heavily
    y_threshold = 0.30  # metres — tune to ~half your inter-plant spacing
    for i, ti in enumerate(tracker_order):
        for j, dj in enumerate(det_order):
            dy = abs(tracker_y[ti] - det_y[dj])
            if dy < y_threshold:
                if abs(i-j)>4:
                    print("weird combination!!!")
                order_penalty = abs(i - j) * 0.05  # small penalty for skipping neighbours
                cost[ti, dj] = dy + order_penalty
    return cost
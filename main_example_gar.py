from pathlib import Path

import cv2
import natsort
import numpy as np
import pandas as pd
import supervision as sv
from supervision.detection.core import Detections

from trackers import CustomSORTTracker
from trackers.utils.match_template import estimate_shift_match_template

tracker = CustomSORTTracker(
    lost_track_buffer=30,
    frame_rate=30,
    track_activation_threshold=0.25,
    minimum_consecutive_frames=3,
    minimum_iou_threshold=0.2 # for demo low
)

main_folder = Path("test/data/belt_tulip/")
main_folder = Path("/home/agro/w-drive-vision/GARdata/new_format/3710485404_bollennpplkoen/datasets/videos/frames_and_detections/")

input_files = natsort.natsorted(main_folder.glob("*.json"))

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()

# https://supervision.roboflow.com/0.27.0/notebooks/count-objects-crossing-the-line/#process-video
START = sv.Point(550, 432)
END = sv.Point(550, 0)

line_zone = sv.LineZone(start=START, end=END
                        , minimum_crossing_threshold=2,
                        triggering_anchors=[sv.Position.CENTER])

line_zone_annotator = sv.LineZoneAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=1,
    text_orient_to_line=True)

prev_img = None
for i, annot_name in enumerate(input_files):

    detections = Detections.from_darwin(
                    json_name=annot_name,
                    with_masks=True,
                    classes=["Potato"],
                    skip_unknown_classes=True,
                    with_track_ids=False,
                    # with_ellipse_as=False,
                )

    frame_bgr = cv2.imread(str(annot_name.parent / (annot_name.stem + ".png")))
    next_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if i==0:
        tmp_shift = 0
        shift = tmp_shift
    else:
        # if df.iloc[i-1]["pixelshift"]*-1!=0:
        #     print("debugging")
        shift, res = estimate_shift_match_template(bottom_incoming_image=next_img,
                                            top_stitched_image=prev_img,
                                            starting_roi_xyxy=[300, 40, 800, 375],
                                            max_shift=110,
                                            movement="x-axis", # or x-axis
                                            direction= "decrease" # decrease
        )
        # shift = 0

        print("calculated shift", shift, annot_name.name)

    detections = tracker.update(detections, u=np.array([shift, 0, shift, 0]))

    annotated_frame = box_annotator.annotate(frame_bgr, detections)

    # disable dbecause was time consuming
    # annotated_frame = mask_annotator.annotate(annotated_frame, detections[0 ])
    # Convert tracker_id to string and handle empty tracker_id
    if detections.tracker_id is None or len(detections.tracker_id) == 0:
        # Handle empty tracker or None
        detections.tracker_id = ["Unknown"] * len(detections)
    else:
        detections.tracker_id = [
            str(t_id) if t_id is not None else "Unknown"
            for t_id in detections.tracker_id
        ]
    annotated_frame = label_annotator.annotate(annotated_frame,
                                               detections,
                                               labels=detections.tracker_id)

    if shift!=0:
        line_zone.trigger(detections)
    line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

    cv2.imshow("CustomFilter", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # # Initialize video writer on first frame
    # if i == 0:
    #     height, width = annotated_frame.shape[:2]
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter('output/default.mp4', fourcc, 20.0,
    #           (width, height))

    # # Write the frame to video
    # out.write(annotated_frame)

    # # Release video writer after the loop (add after the loop ends)
    # if i == len(input_files) - 1:
    #     out.release()

    prev_img = next_img.copy()

print("finished")
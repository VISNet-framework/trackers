from pathlib import Path

import cv2
from trackers.utils.match_template import estimate_shift_match_template


def test_match_template():
    match(movement="x-axis", direction="increase")
    match(movement="x-axis", direction="decrease")

    match(movement="y-axis", direction="increase")
    match(movement="y-axis", direction="decrease")


def match(movement, direction):

    indir = Path(__file__).parent / "../../data/belt_tulip"
    name_1 = indir / "437.png"
    name_2 = indir / "438.png"

    def read_data():
        bgr_img1 = cv2.imread(str(name_1), cv2.IMREAD_GRAYSCALE)
        bgr_img2 = cv2.imread(str(name_2), cv2.IMREAD_GRAYSCALE)

        # # Rotate both images counter clockwise 90 degrees
        bgr_img1 = cv2.rotate(bgr_img1, cv2.ROTATE_90_CLOCKWISE)
        bgr_img2 = cv2.rotate(bgr_img2, cv2.ROTATE_90_CLOCKWISE)
        return bgr_img1, bgr_img2

    gt = -46
    if direction=="increase":
        gt = 46
        bgr_img2, bgr_img1 = read_data()
    else:
        bgr_img1, bgr_img2 = read_data()


    starting_roi_xyxy = [40, 300,375, 800]
    if movement=="x-axis":
        starting_roi_xyxy = [300, 40, 800, 375]

        bgr_img1 = cv2.rotate(bgr_img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bgr_img2 = cv2.rotate(bgr_img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    shift_y, _ = estimate_shift_match_template(bottom_incoming_image=bgr_img2,
                                           top_stitched_image=bgr_img1,
                                           starting_roi_xyxy=starting_roi_xyxy,
                                           max_shift=110,
                                           movement=movement,
                                           direction=direction
    )
    assert shift_y==gt

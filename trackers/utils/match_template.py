"""
Script to estimate shift. Copied from:
https://github.com/roel-klein/stitchem
"""

import warnings

import cv2
import numpy as np
import numpy.typing as npt


def estimate_vertical_shift_match_template(
        bottom_incoming_image : npt.NDArray,
        top_stitched_image: npt.NDArray,
        starting_roi_xyxy: list[int],
        max_shift : int,
        method = cv2.TM_CCOEFF):

    pixelshift, res = estimate_shift_match_template(bottom_incoming_image,
                                         top_stitched_image,
                                         starting_roi_xyxy,
                                         max_shift,
                                         method)
    ## TODO fix in future to prevent the use of -1
    return pixelshift*-1, res

def estimate_shift_match_template(
        bottom_incoming_image : npt.NDArray,
        top_stitched_image: npt.NDArray,
        starting_roi_xyxy: list[int],
        max_shift : int,
        method = cv2.TM_CCOEFF,
        movement: str = "y-axis", # or x-axis
        direction: str = "decrease", # decrease
        mask = None,
    ) -> tuple[int, npt.NDArray]:
    """
    Estimates the shift in pixels from top_stitched_image to bottom_incoming_image.
    Arguments:
        - bottom_incoming_image : shifted image to align, grayscale np.uint8
        - top_stitched_image : starting image to align with, grayscale np.uint8
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for align.
        - max_shift : maximum vertical shift in pixels
    Returns:
        pixelshift : Estimated number pixels of that should be added from incoming image
        errors : For each pixelshift, the sum of the MAE.
    """
    roi_x1 = starting_roi_xyxy[0]
    roi_y1 = starting_roi_xyxy[1]
    roi_x2 = starting_roi_xyxy[2]
    roi_y2 = starting_roi_xyxy[3]

    if movement=="y-axis":
        height = bottom_incoming_image.shape[0]
        max_shift_x = 0
        max_shift_y = min(max_shift, height)

        if roi_y1 < max_shift_y:
            warnings.warn(
                f"The region of interest y-position {roi_y1} is smaller than "
                f"maximum shift {max_shift}. Adjusting y-position to "
                f"{max_shift}, to make it possible to check previous pixel "
                f"lines. Adjusting either max_shift or still_box_xyxy is "
                f"recommended to prevent possible runtimeerrors.",
                stacklevel=2,
            )
            # further shift starting roi y-position if necessary
            # to make sure we can look at the max_shift pixels before it
            roi_y2 += max_shift_y - roi_y1
            roi_y1 += max_shift_y - roi_y1
            if roi_y2 > height:
                raise RuntimeError("Bounding box exceeded image border, reduce max "
                                   "shift or roi box size")

        # convert roi indices to be negative, relative to image end
        # this way we can use the same roi indices for a longer stitched image
        # and a shorter incoming image
        roi_y1 -= bottom_incoming_image.shape[0]
        roi_y2 -= bottom_incoming_image.shape[0]

    elif movement=="x-axis":
        width = bottom_incoming_image.shape[1]
        max_shift_y = 0
        max_shift_x = min(max_shift, width)

        if roi_x1 < max_shift_x:
            warnings.warn(
                f"The region of interest x-position {roi_x1} is smaller than "
                f"maximum shift {max_shift}. Adjusting x-position to "
                f"{max_shift}, to make it possible to check previous pixel "
                f"lines. Adjusting either max_shift or still_box_xyxy is "
                f"recommended to prevent possible runtimeerrors.",
                stacklevel=2,
            )
            # further shift starting roi y-position if necessary
            # to make sure we can look at the max_shift pixels before it
            roi_x2 += max_shift_x - roi_x1
            roi_x1 += max_shift_x - roi_x1
            if roi_x2 > width:
                raise RuntimeError("Bounding box exceeded image border, reduce max "
                                   "shift or roi box size")

        # convert roi indices to be negative, relative to image end
        # this way we can use the same roi indices for a longer stitched image
        # and a shorter incoming image
        roi_x1 -= bottom_incoming_image.shape[1]
        roi_x2 -= bottom_incoming_image.shape[1]

    else:
        KeyError()

    if direction=="decrease":
        # Apply template Matching
        search = bottom_incoming_image[roi_y1-max_shift_y:roi_y2,
                                       roi_x1-max_shift_x:roi_x2]
        index_order = -1
    elif direction=="increase":
        search = bottom_incoming_image[roi_y1:roi_y2+max_shift_y,
                                       roi_x1:roi_x2+max_shift_x]
        index_order = 1


    template = top_stitched_image[roi_y1:roi_y2, roi_x1:roi_x2]
    if mask is not None:
        # mask is for template!
        res = cv2.matchTemplate(search, template, method,mask=mask[roi_y1:roi_y2, roi_x1:roi_x2])
    else:
        res = cv2.matchTemplate(search, template, method,mask=None)

    if movement=="x-axis":
        res = res.T

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, lower is better (error metric)
    # for other, higher is better(similarity metric)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        pixelshift = int(np.argmin(res[::index_order, 0])) * 1*index_order
    else:
        pixelshift = int(np.argmax(res[::index_order, 0])) * 1*index_order

    # import matplotlib.pyplot as plt
    # cv2.imwrite("search.png", search)
    # cv2.imwrite("searchT.png", template)
    # plt.plot(res[::index_order, 0])
    # plt.show()
    return pixelshift, res


def estimate_vertical_shift_match_template_twostage(
        bottom_incoming_image : npt.NDArray,
        top_stitched_image: npt.NDArray,
        starting_roi_xyxy: list[int],
        max_shift : int,
        first_stage_decimation : int = 4,
        method = cv2.TM_SQDIFF,
    ) -> tuple[int, npt.NDArray]:
    """
    Estimates the vertical shift in pixels from top_stitched_image to incoming image.
    Arguments:
        - bottom_incoming_image : shifted image to align, grayscale np.uint8
        - top_stitched_image : starting image to align with
        - starting_roi_xyxy: bounding box of the bottom_stitched_image for alignment
        - max_shift : maximum vertical shift in pixels
        - first_stage_decimation : first-stage subsampling factor in vertical direction
    Returns:
        pixelshift : Estimated number pixels of to be added from incoming image.
        errors : For each pixelshift, the sum of the MAE.
    """
    height = bottom_incoming_image.shape[0]
    max_shift = min(max_shift, height)

    roi_x1 = starting_roi_xyxy[0]
    roi_y1 = starting_roi_xyxy[1]
    roi_x2 = starting_roi_xyxy[2]
    roi_y2 = starting_roi_xyxy[3]

    if roi_y1 < max_shift:
        warnings.warn(
            f"The region of interest y-position {roi_y1} is smaller than "
            f"maximum shift {max_shift}. Adjusting y-position to "
            f"{max_shift}, to make it possible to check previous pixel "
            f"lines. Adjusting either max_shift or still_box_xyxy is "
            f"recommended to prevent possible runtimeerrors.",
            stacklevel=2,
        )
        # further shift starting roi y-position if necessary
        # to make sure we can look at the max_shift pixels before it
        roi_y2 += max_shift - roi_y1
        roi_y1 += max_shift - roi_y1
        if roi_y2 > height:
            raise RuntimeError("Bounding box exceeded image border, reduce max shift or"
                               "roi box size")

    # convert roi indices to be negative, relative to image end
    # this way we can use the same roi indices for a longer stitched image
    # and a shorter incoming image
    roi_y1 -= bottom_incoming_image.shape[0]
    roi_y2 -= bottom_incoming_image.shape[0]

    # Stage 1 : Coarse
    search = bottom_incoming_image[roi_y1-max_shift:roi_y2:first_stage_decimation,
                                   roi_x1:roi_x2]
    template = top_stitched_image[roi_y1:roi_y2:first_stage_decimation,
                                  roi_x1:roi_x2]

    res = cv2.matchTemplate(search,template,method)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, lower is better (error metric)
    # for other, higher is better(similarity metric)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        pixelshift = int(np.argmin(res[::-1, 0]) * first_stage_decimation)
    else:
        pixelshift = int(np.argmax(res[::-1, 0]) * first_stage_decimation)

    # Stage 2 : Pixel-precise
    new_y1 = max(roi_y1 - pixelshift - first_stage_decimation, -bottom_incoming_image.shape[0]) + 1  # noqa: E501
    new_y2 = min(roi_y2 - pixelshift + first_stage_decimation, roi_y2)
    search = bottom_incoming_image[new_y1:new_y2, roi_x1:roi_x2]
    template = top_stitched_image[roi_y1:roi_y2, roi_x1:roi_x2]

    res = cv2.matchTemplate(search,template,method)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, lower is better (error metric)
    # for other, higher is better(similarity metric)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        pixelshift = int(np.argmin(res[::-1, 0]))
    else:
        pixelshift = int(np.argmax(res[::-1, 0]))
    return pixelshift, res

def estimate_vertical_shift_orb(
        bottom_incoming_image : npt.NDArray,
        top_stitched_image: npt.NDArray,
        starting_roi_xyxy: list[int],
        max_shift : int,
        movement: str = "y-axis", # or x-axis
        direction: str = "decrease", # decrease
        n_features: int = 30,
        k1d1_precomputed : tuple[cv2.KeyPoint, npt.NDArray] | None = None,

    ) -> tuple[float, tuple[npt.NDArray[np.float64], float]]:
    """
    Estimates the vertical shift in pixels from still_image to shift_image using orb feature matching.
    Note that the maximum shift is implicitly the y-range of the starting region of interest. 

    Arguments:
        - bottom_incoming_image : shifted image to align, BGR np.uint8
        - top_stitched_image : starting image to align with, BGR np.uint8
        - starting_roi_xyxy: bounding box of the bottom_stitched_image to use for alignment
        - upsample_factor : fft upsampling for sub-pixel precision.
                default=1 (no upsampling).
        
    Returns:
        pixelshift : Estimated vertical shift in pixels necessary to align shift_image with still_image 
    """
    roi_x1 = starting_roi_xyxy[0]
    roi_y1 = starting_roi_xyxy[1]
    roi_x2 = starting_roi_xyxy[2]
    roi_y2 = starting_roi_xyxy[3]

    if movement=="y-axis":
        height = bottom_incoming_image.shape[0]
        max_shift_x = 0
        max_shift_y = min(max_shift, height)

        if roi_y1 < max_shift_y:
            warnings.warn(
                f"The region of interest y-position {roi_y1} is smaller than "
                f"maximum shift {max_shift}. Adjusting y-position to "
                f"{max_shift}, to make it possible to check previous pixel "
                f"lines. Adjusting either max_shift or still_box_xyxy is "
                f"recommended to prevent possible runtimeerrors.",
                stacklevel=2,
            )
            # further shift starting roi y-position if necessary
            # to make sure we can look at the max_shift pixels before it
            roi_y2 += max_shift_y - roi_y1
            roi_y1 += max_shift_y - roi_y1
            if roi_y2 > height:
                raise RuntimeError("Bounding box exceeded image border, reduce max "
                                   "shift or roi box size")

        # convert roi indices to be negative, relative to image end
        # this way we can use the same roi indices for a longer stitched image
        # and a shorter incoming image
        roi_y1 -= bottom_incoming_image.shape[0]
        roi_y2 -= bottom_incoming_image.shape[0]

    elif movement=="x-axis":
        width = bottom_incoming_image.shape[1]
        max_shift_y = 0
        max_shift_x = min(max_shift, width)

        if roi_x1 < max_shift_x:
            warnings.warn(
                f"The region of interest x-position {roi_x1} is smaller than "
                f"maximum shift {max_shift}. Adjusting x-position to "
                f"{max_shift}, to make it possible to check previous pixel "
                f"lines. Adjusting either max_shift or still_box_xyxy is "
                f"recommended to prevent possible runtimeerrors.",
                stacklevel=2,
            )
            # further shift starting roi y-position if necessary
            # to make sure we can look at the max_shift pixels before it
            roi_x2 += max_shift_x - roi_x1
            roi_x1 += max_shift_x - roi_x1
            if roi_x2 > width:
                raise RuntimeError("Bounding box exceeded image border, reduce max "
                                   "shift or roi box size")

        # convert roi indices to be negative, relative to image end
        # this way we can use the same roi indices for a longer stitched image
        # and a shorter incoming image
        roi_x1 -= bottom_incoming_image.shape[1]
        roi_x2 -= bottom_incoming_image.shape[1]

    else:
        KeyError()

    if direction=="decrease":
        # Apply template Matching
        search = bottom_incoming_image[roi_y1-max_shift_y:roi_y2,
                                       roi_x1-max_shift_x:roi_x2]
        index_order = -1
    elif direction=="increase":
        search = bottom_incoming_image[roi_y1:roi_y2+max_shift_y,
                                       roi_x1:roi_x2+max_shift_x]
        index_order = 1


    template = top_stitched_image[roi_y1:roi_y2, roi_x1:roi_x2]

    orb = cv2.ORB_create(n_features) # type: ignore
    if k1d1_precomputed is None:
        k1, d1 = orb.detectAndCompute(template, None)
    else:
        k1, d1 = k1d1_precomputed
    k2, d2 = orb.detectAndCompute(search, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)


    # we don't need matches to be sorted
    # matches = sorted(matches, key=lambda m: m.distance)

    if not matches:
        warnings.warn("No orb matches found, using default pixelshift of zero.", stacklevel=2)
        return 0.0, (k2, d2)

    dy_values = []
    dx_values = []
    for m in matches:
        p1 = np.array(k1[m.queryIdx].pt)
        p2 = np.array(k2[m.trainIdx].pt)
        dx = p1[0] - p2[0]   # horizontal movement (pixels)
        dy = p1[1] - p2[1]   # vertical movement (pixels)
        dx_values.append(dx)
        dy_values.append(dy)

    # Median is robust to mismatches
    if movement=="x-axis":
        return np.median(dx_values), (k2, d2)
    
    # Draw matches for visualization
    # img_matches = cv2.drawMatches(template, k1, search, k2, matches, None)
    # img_matches = cv2.resize(img_matches, dsize=None, fx=0.25, fy=0.25)
    # cv2.putText(img_matches, f"Shift: {np.median(dy_values)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    # plt.title(f"Shift: {np.median(dy_values)}")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    return np.median(dy_values), (k2, d2)
    
    
    # # if pixelshift <= 0:
    # #     warnings.warn(f"ORB found negative pixelshift {pixelshift}", stacklevel=2)
    # return pixelshift, (k2, d2)

if __name__=="__main__":
    from pathlib import Path

    indir = Path("data/belt_tulip")
    name_1 = indir / "437.png"
    name_2 = indir / "438.png"

    movement="x-axis"
    direction="increase"

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

    shift_y, res = estimate_shift_match_template(bottom_incoming_image=bgr_img2,
                                           top_stitched_image=bgr_img1,
                                           starting_roi_xyxy=starting_roi_xyxy,
                                           max_shift=110,
                                           movement=movement,
                                           direction=direction
    )

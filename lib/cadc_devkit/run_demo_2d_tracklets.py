import cv2
import json

import numpy as np

from typing import Tuple
from pathlib import Path
from matplotlib import pyplot as plt
from load_calibration import  load_calibration


MOVE_FORWARD = True


def vis_gt_on_cam_2d(date: str, sequence: str, camera: str, frame: int, base_dir: str = None) -> Tuple[np.ndarray, str]:

    if base_dir:
        BASE = Path(base_dir)
    else:
        BASE = Path.home() / 'datasets' / 'CADCD'

    image_path = str(BASE / date / sequence / "labeled" / f"image_0{camera}" / "data" / f"{format(frame, '010')}.png")
    calib_path = str(BASE / date / "calib")

    distorted = "raw" in image_path

    annotations_file = BASE / date / sequence / "2d_annotations.json"

    # load calibration dictionary
    calib = load_calibration(calib_path)

    # Projection matrix from camera to image frame
    T_IMG_CAM = np.eye(4)
    T_IMG_CAM[0:3,0:3] = np.array(calib['CAM0' + camera]['camera_matrix']['data']).reshape(-1, 3)
    T_IMG_CAM = T_IMG_CAM[0:3,0:4] # remove last row

    dist_coeffs = np.array(calib['CAM0' + camera]['distortion_coefficients']['data'])

    # Load 2d annotations
    with open(annotations_file) as f:
        annotations_data = json.load(f)

    img = cv2.imread(image_path)

    # Add each box to image
    for camera_response in annotations_data[frame]['camera_responses']:

        if camera_response['camera_used'] != int(camera):
            continue

        for annotation in camera_response['annotations']:

            left = int(annotation['left'])
            top = int(annotation['top'])
            width = int(annotation['width'])
            height = int(annotation['height'])

            if distorted:

                cv2.rectangle(img, (left,top), (left + width,top + height), (0, 255, 0), thickness=3)

            else:

                pts_uv = np.array([[[left,top]],[[left + width,top + height]]], dtype=np.float32)
                new_pts = cv2.undistortPoints(pts_uv, T_IMG_CAM[0:3,0:3], dist_coeffs, P=T_IMG_CAM[0:3,0:3])

                cv2.rectangle(img,
                              (new_pts[0][0][0], new_pts[0][0][1]),
                              (new_pts[1][0][0], new_pts[1][0][1]), (0, 255, 0),
                              thickness=3)

    relative_path = '/'.join(image_path.split('/')[-6:])

    return img, relative_path


if __name__ == '__main__':

    for cam_id in range(8):

        image, title = vis_gt_on_cam_2d(date='2019_02_27', sequence='0010', camera=str(cam_id), frame=26)

        plt.title(title)
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])  # to hide tick values on x and y axis
        plt.show()
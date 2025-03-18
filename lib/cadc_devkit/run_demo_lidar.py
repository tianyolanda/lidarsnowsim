import sys
import cv2

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from lidar_utils import lidar_utils
from load_calibration import load_calibration


def lidar(date: str, sequence: str, camera: str, frame: int, base_dir: str=None, move_forward: bool=False) -> None:

    if base_dir:
        BASE = Path(base_dir)
    else:
        BASE = Path.home() / 'datasets' / 'CADCD'

    path_type = 'labeled'
    distorted = path_type == 'raw'

    assert path_type in ['labeled', 'raw'], f'unknown path_type "{path_type}"'

    image_path = str(BASE / date / sequence / path_type / f"image_0{camera}" / "data" / f"{format(frame, '010')}.png")
    lidar_path = str(BASE / date / sequence / path_type / "lidar_points" / "data" / f"{format(frame, '010')}.bin")
    calib_path = str(BASE / date / "calib")

    # load calibration dictionary
    calib = load_calibration(calib_path)
  
    # Projection matrix from camera to image frame
    t_img_cam = np.eye(4)
    t_img_cam[0:3, 0:3] = np.array(calib['CAM0' + camera]['camera_matrix']['data']).reshape(-1, 3)
    t_img_cam = t_img_cam[0:3, 0:4] # remove last row
  
    t_cam_lidar = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + camera]))
  
    dist_coeffs = np.array(calib['CAM0' + camera]['distortion_coefficients']['data'])

    lidar_utils_obj = lidar_utils(t_cam_lidar)

    while True:

        try:

            print(f'{date} - {sequence} - {frame}')

            # read image
            img = cv2.imread(image_path)

            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Project points onto image
            img = lidar_utils_obj.project_points(img, lidar_path, t_img_cam, t_cam_lidar, dist_coeffs, distorted)
            # cv2.imwrite("test.png", img)

            image = Image.fromarray(img, 'RGB')

            plt.imshow(image)
            plt.xticks([]), plt.yticks([])  # to hide tick values on x and y axis
            plt.show()
            plt.clf()                       # will make the plot window empty
            plt.close()

            if move_forward:

                frame += 1

                image_path = str(BASE / date / sequence / path_type / f"image_0{camera}" / "data" /
                                 f"{format(frame, '010')}.png")
                lidar_path = str(BASE / date / sequence / path_type / "lidar_points" / "data" /
                                 f"{format(frame, '010')}.bin")

                img = cv2.imread(image_path)

        except FileNotFoundError:

            print('end of sequence reached')
            sys.exit(0)



if __name__ == '__main__':

    lidar(date='2019_02_27', sequence='0043', camera='0', frame=0, move_forward = True)
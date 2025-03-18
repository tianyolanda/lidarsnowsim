import cv2

import numpy as np


class lidar_utils:

    def __init__(self, t_cam_lidar):

        self.t_cam_lidar = t_cam_lidar


    def project_points(self, img, lidar_path, t_img_cam, t_cam_lidar, dist_coeffs, distorted):

        self.t_cam_lidar = t_cam_lidar

        scan_data = np.fromfile(lidar_path, dtype=np.float32)

        # 2D array where each row contains a point [x, y, z, intensity]
        lidar = scan_data.reshape((-1, 4))

        # Get height and width of the image
        h, w = img.shape[:2]

        projected_points = []

        rows = lidar.shape[0]
        # print(lidar[0,:])
        # print(lidar[1,:])
        # print(lidar[1,0:3])

        for i in range(rows):
            # print(lidar[i,:])
            p = np.array([0.0, 0.0, 0.0, 1.0])
            p[0:3] = lidar[i,0:3]
            # print("p",p)
            projected_p =  np.matmul(self.t_cam_lidar, p.transpose())
            if projected_p[2] < 2: # arbitrary cut off
                continue
            projected_points.append([projected_p[0], projected_p[1], projected_p[2]])

        #print("projected_points", projected_points)

        # Send [x, y, z] and Transform
        projected_points_np = np.array(projected_points)
        image_points = project(projected_points_np, t_img_cam, dist_coeffs, distorted)
        # print("image_points")
        # print(image_points)

        radius = 0

        rows = projected_points_np.shape[0]

        NUM_COLOURS = 7
        rainbow = [
            [0, 0, 255], # Red
            [0, 127, 255], # Orange
            [0, 255, 255], # Yellow
            [0, 255, 0], # Green
            [255, 0, 0], # Blue
            [130, 0, 75], # Indigo
            [211, 0, 148] # Violet
        ]

        for i in range(rows):

            colour = int(NUM_COLOURS*(projected_points_np[i][2]/70))
            x = int(image_points[i][0])
            y = int(image_points[i][1])
            if x < 0 or x > w - 1 or y < 0 or y > h - 1:
                continue
            if colour > NUM_COLOURS-1:
                continue

            cv2.circle(img, (x,y), radius, rainbow[colour], thickness=2, lineType=8, shift=0)

        return img


def project(p_in, t_img_cam, dist_coeffs, distorted):

    p_out = []
    rows = p_in.shape[0]

    for i in range(rows):
        # print("p_in[i]", p_in[i])
        point = np.array([0.0, 0.0, 0.0, 1.0])
        # print("p_in[i][0]",p_in[i][0])
        point[0:3] = p_in[i]
        # print("p[0]",p[0])
        if distorted:
            rvec = tvec = np.zeros(3)
            # print(p[0:3])
            # print(t_img_cam[0:3,0:3])
            # print(np.array(calib['CAM02']['distortion_coefficients']['data']))
            image_points, jac = cv2.projectPoints(np.array([point[0:3]]), rvec, tvec, t_img_cam[0:3,0:3], dist_coeffs)
            p_out.append([image_points[0,0,0], image_points[0,0,1]])
            # print("image_points", image_points[0,0])
        else:
            curr = np.matmul(t_img_cam, point.transpose()).transpose()
            # print("curr",curr)
            done = [curr[0] / curr[2], curr[1] / curr[2]]
            p_out.append(done)
            # print("p_out append", done)

    return p_out

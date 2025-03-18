import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from typing import Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation as R

RED, GREEN, BLUE, YELLOW = [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]

COLORMAP = {'Car': GREEN,
            'Cyclist': YELLOW,
            'Pedestrian': RED}

MOVE_FORWARD = True
DISPLAY_LIDAR = False
DISPLAY_CUBOID_CENTER = False
MIN_CUBOID_DIST = 40.0


def bev(date: str, sequence: str, frame: int, left: int=50, right: int=50, front: int=50, back: int=50,
        linewidth: int=20, pointsize: int=50, dpi: int=10, base_dir: str=None, plot_center: bool=False,
        plot_partly: bool=False, use_intensity: bool=False) -> Tuple[np.ndarray, str]:

    if base_dir:
        BASE = Path(base_dir)
    else:
        BASE = Path.home() / 'datasets' / 'CADCD'

    lidar_path = str(BASE / date / sequence / "labeled" / "lidar_points" / "data" / f"{format(frame, '010')}.bin")

    annotations_file = BASE / date / sequence / "3d_ann.json"

    #limit the viewing range
    side_range = [-left, right]
    fwd_range = [-front, back]

    w, h = left + right, front + back

    scan_data = np.fromfile(lidar_path, dtype= np.float32)  # single row of all the lidar values
    lidar = scan_data.reshape((-1, 4))                      # 2D array where each row contains [x, y, z, intensity]

    lidar_x = lidar[:, 0]
    lidar_y = lidar[:, 1]
    lidar_z = lidar[:, 2]
    lidar_i = lidar[:, 3]

    lidar_x_trunc = []
    lidar_y_trunc = []
    lidar_z_trunc = []
    lidar_i_trunc = []

    for i in range(len(lidar_x)):

        if fwd_range[0] < lidar_x[i] < fwd_range[1] and side_range[0] < lidar_y[i] < side_range[1]:

            lidar_x_trunc.append(lidar_x[i])
            lidar_y_trunc.append(lidar_y[i])
            lidar_z_trunc.append(lidar_z[i])
            lidar_i_trunc.append(lidar_i[i])

    # to use for the plot
    x_img = [i * -1 for i in lidar_y_trunc] # the negative lidar y axis is the positive img x axis
    y_img = lidar_x_trunc                   # the lidar x axis is the img y axis

    if use_intensity:
        pixel_values = lidar_i_trunc
    else:
        pixel_values = lidar_z_trunc

    # shift values such that 0,0 is the minimum
    x_img = [i - side_range[0] for i in x_img]
    y_img = [i - fwd_range[0] for i in y_img]

    '''
    tracklets 
    '''

    # Load 3d annotations
    with open(annotations_file) as f:
        annotations_data = json.load(f)

    # Add each cuboid to image
    '''
    Rotations in 3 dimensions can be represented by a sequece of 3 rotations around a sequence of axes. 
    In theory, any three axes spanning the 3D Euclidean space are enough. 
    In practice the axes of rotation are chosen to be the basis vectors.
    The three rotations can either be in a global frame of reference (extrinsic) 
    or in a body centred frame of refernce (intrinsic), which is attached to, and moves with, the object under rotation
    '''

    # PLOT THE IMAGE
    cmap = "jet"    # color map to use
    x_max = side_range[1] - side_range[0]
    y_max = fwd_range[1] - fwd_range[0]
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)

    # the coordinates in the tracklet json are lidar coords
    x_trunc = []
    y_trunc = []
    x_1 = []
    x_2 =[]
    x_3 = []
    x_4 =[]
    y_1 =[]
    y_2 =[]
    y_3 = []
    y_4 =[]

    colors = []

    for cuboid in annotations_data[frame]['cuboids']:

        T_Lidar_Cuboid = np.eye(4)                                                              # identify matrix
        T_Lidar_Cuboid[0:3, 0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_dcm()     # rotate the identity matrix
        T_Lidar_Cuboid[0][3] = cuboid['position']['x']                                          # center of the tracklet, from cuboid to lidar
        T_Lidar_Cuboid[1][3] = cuboid['position']['y']
        T_Lidar_Cuboid[2][3] = cuboid['position']['z']

        # make sure the cuboid is within the range we want to see
        if not plot_partly and ((not fwd_range[0] < cuboid['position']['x'] < fwd_range[1]) or
                                (not side_range[0] < cuboid['position']['y'] < side_range[1])):
            continue

        x_trunc.append(cuboid['position']['x'])
        y_trunc.append(cuboid['position']['y'])

        width = cuboid['dimensions']['x']
        length = cuboid['dimensions']['y']
        height = cuboid['dimensions']['z']

        label = cuboid['label']
        color = COLORMAP.get(label, BLUE)
        colors.append(color)

        # the top view of the tracklet in the "cuboid frame". The cuboid frame is a cuboid with origin (0, 0, 0)
        # we are making a cuboid that has the dimensions of the tracklet but is located at the origin
        front_top_right = np.array(
            [[1, 0, 0, length / 2], [0, 1, 0, width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]])

        front_top_left = np.array(
            [[1, 0, 0, length / 2], [0, 1, 0, -width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]])


        back_top_right = np.array(
            [[1, 0, 0, -length / 2], [0, 1, 0, width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]])

        back_top_left = np.array(
            [[1, 0, 0, -length / 2], [0, 1, 0, -width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]])

        # Project to lidar
        f_t_r =  np.matmul(T_Lidar_Cuboid, front_top_right)
        f_t_l  = np.matmul(T_Lidar_Cuboid, front_top_left)
        b_t_r  = np.matmul(T_Lidar_Cuboid, back_top_right)
        b_t_l = np.matmul(T_Lidar_Cuboid, back_top_left)

        x_1.append(f_t_r[0][3])
        y_1.append(f_t_r[1][3])

        x_2.append(f_t_l[0][3])
        y_2.append(f_t_l[1][3])

        x_3.append(b_t_r[0][3])
        y_3.append(b_t_r[1][3])

        x_4.append(b_t_l[0][3])
        y_4.append(b_t_l[1][3])

    # to use for the plot
    x_img_tracklets = [i * -1 for i in y_trunc] # the negative lidar y axis is the positive img x axis
    y_img_tracklets = x_trunc                   # the lidar x axis is the img y axis

    x_img_1 = [i * -1 for i in y_1]
    y_img_1 = x_1

    x_img_2 = [i * -1 for i in y_2]
    y_img_2 = x_2

    x_img_3 = [i * -1 for i in y_3]
    y_img_3 = x_3

    x_img_4 = [i * -1 for i in y_4]
    y_img_4 = x_4

    # shift values such that 0, 0 is the minimum
    x_img_tracklets = [i -side_range[0] for i in x_img_tracklets]
    y_img_tracklets = [i -fwd_range[0] for i in y_img_tracklets]

    x_img_1 = [i -side_range[0] for i in x_img_1]
    y_img_1 = [i - fwd_range[0] for i in y_img_1]

    x_img_2 = [i - side_range[0] for i in x_img_2]
    y_img_2 = [i - fwd_range[0] for i in y_img_2]

    x_img_3 = [i - side_range[0] for i in x_img_3]
    y_img_3 = [i - fwd_range[0] for i in y_img_3]

    x_img_4 = [i - side_range[0] for i in x_img_4]
    y_img_4 = [i - fwd_range[0] for i in y_img_4]

    # plot the tracklets
    for i in range(len(x_img_1)):
        poly = np.array([[x_img_1[i], y_img_1[i]], [x_img_2[i], y_img_2[i]],
                         [x_img_4[i], y_img_4[i]], [x_img_3[i], y_img_3[i]]])
        polys = patches.Polygon(poly, closed=True, fill=False, edgecolor=colors[i], linewidth=linewidth)
        ax.add_patch(polys)

        if plot_center:
            ax.scatter(x_img_tracklets, y_img_tracklets, marker ='o', color=colors[i], linewidths=linewidth)

    ax.scatter(x_img, y_img, s=pointsize, c=pixel_values, alpha=1.0, cmap=cmap)
    ax.axis('scaled')           # {equal, scaled}
    ax.xaxis.set_visible(False) # do not draw axis tick marks
    ax.yaxis.set_visible(False) # do not draw axis tick marks

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.xlim([0, x_max])
    plt.ylim([0, y_max])

    # If we haven't already shown or saved the plot, then we need to draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    relative_path = '/'.join(lidar_path.split('/')[-6:])

    return img, relative_path


if __name__ == '__main__':

    image, title = bev(date='2019_02_27', sequence='0043', frame=0)

    image = Image.fromarray(image, 'RGB')
    image.show()
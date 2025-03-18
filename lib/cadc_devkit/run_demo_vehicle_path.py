import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

import load_novatel_data, convert_novatel_to_pose


def show_path(date: str, sequence: str, base_dir: str=None) -> None:

    if base_dir:
        BASE = Path(base_dir)
    else:
        BASE = Path.home() / 'datasets' / 'CADCD'

    novatel_path = BASE / date / sequence / 'labeled' / 'novatel' / 'data'

    novatel = load_novatel_data.load_novatel_data(novatel_path)
    poses = convert_novatel_to_pose.convert_novatel_to_pose(novatel)

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_title('Vehicle path')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')

    length = 1
    A = np.matrix([[0, 0, 0, 1],
                   [length, 0, 0, 1],
                   [0, 0, 0, 1],
                   [0, length, 0, 1],
                   [0, 0, 0, 1],
                   [0, 0, length, 1]]).transpose()

    for pose in tqdm(poses):
        B = np.matmul(pose, A)
        ax.plot([B[0,0], B[0,1]], [B[1,0], B[1,1]],[B[2,0],B[2,1]], 'r-') # x: red
        ax.plot([B[0,2], B[0,3]], [B[1,2], B[1,3]],[B[2,2],B[2,3]], 'g-') # y: green
        ax.plot([B[0,4], B[0,5]], [B[1,4], B[1,5]],[B[2,4],B[2,5]], 'b-') # z: blue

    # Equal axis doesn't seem to work so set an arbitrary limit to the z axis
    ax.set_zlim3d(-10,10)

    plt.show()

if __name__ == '__main__':

    show_path(date='2019_02_27', sequence='0043')
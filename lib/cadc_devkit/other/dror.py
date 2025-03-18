__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import os
import pcl
import pickle
import logging
import argparse

import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import datetime


CADC_ROOT = Path().home() / 'datasets' / 'CADCD'
DENSE_ROOT = Path().home() / 'datasets' / 'DENSE' / 'SeeingThroughFog'


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default='DENSE', choices=['CADC', 'DENSE'],
                        help='specify the dataset to be processed')
    parser.add_argument('--alpha', type=float, default=0.08,
                        help='specify the horizontal angular resolution of the lidar')
    parser.add_argument('--crop', type=bool, default=True,
                        help='specify if the pointcloud should be cropped')
    parser.add_argument('--pkl', type=str, default='dense_infos_all.pkl',
                        help='specify the pkl file to be processed (only relevant for DENSE)')
    parser.add_argument('--sensor', type=str, default='hdl64', choices=['vlp32', 'hdl64'],
                        help='specify if the sensor type')
    parser.add_argument('--signal', type=str, default='strongest', choices=['strongest', 'last'],
                        help='specify if the signal type')

    args = parser.parse_args()

    if args.dataset == 'CADC':      # remove irrelevant arguments
        del args.pkl
        del args.sensor
        del args.signal
    else:                           # make sure correct sensor split is used
        if args.sensor == 'vlp32' and 'vlp32' not in args.pkl:
            args.pkl = args.pkl.replace('.pkl', '_vlp32.pkl')

    return args


def load_cadc_pointcloud(date: str, sequence: str, frame: str) -> np.ndarray:

    filename = CADC_ROOT / date / sequence / 'labeled' / 'lidar_points' / 'data' / frame

    pc = np.fromfile(filename, dtype=np.float32)
    pc = pc.reshape((-1, 4))

    pc[:, 3] = np.round(pc[:, 3] * 255)

    return pc


def load_dense_pointcloud(file: str, sensor: str, signal: str) -> np.ndarray:

    filename = DENSE_ROOT / f'lidar_{sensor}_{signal}' / f'{file}.bin'

    pc = np.fromfile(filename, dtype=np.float32)
    pc = pc.reshape((-1, 5))

    return pc


def get_cube_mask(pc: np.ndarray,
                  x_min: float = 3, x_max: float = 13,
                  y_min: float = -1, y_max: float = 1,
                  z_min: float = -1, z_max: float = 1) -> np.ndarray:

    x_mask = np.logical_and(x_min <= pc[:, 0], pc[:, 0] <= x_max)
    y_mask = np.logical_and(y_min <= pc[:, 1], pc[:, 1] <= y_max)
    z_mask = np.logical_and(z_min <= pc[:, 2], pc[:, 2] <= z_max)

    cube_mask = np.logical_and(x_mask, y_mask, z_mask)

    return cube_mask


def process_cadc(args: argparse.Namespace):

    log = args.logger

    dates = sorted(os.listdir(CADC_ROOT))

    stats_dict = {}
    lookup_dict = {}

    for date in dates:

        stats_dict[date] = {}
        lookup_dict[date] = {}

        sequences = sorted(os.listdir(CADC_ROOT / date))

        for sequence in sequences:

            # skip calib folder
            if sequence == 'calib':
                continue

            lookup_dict[date][sequence] = {}

            frames = sorted(os.listdir(CADC_ROOT / date / sequence / 'labeled' / 'lidar_points' / 'data'))
            pbar_frames = tqdm(range(len(frames)), desc=f'{date}/{sequence}')

            first_n_snow = -1
            min_n_snow = np.inf
            max_n_snow = 0
            avg_n_snow = 0

            min_n_cube = np.inf
            max_n_cube = 0
            avg_n_cube = 0

            for f in pbar_frames:

                frame = frames[f]

                pc = load_cadc_pointcloud(date, sequence, frame)

                cube_mask = get_cube_mask(pc)

                if args.crop:
                    pc = pc[cube_mask]

                keep_mask = dynamic_radius_outlier_filter(pc)

                snow_indices = (keep_mask == 0).nonzero()[0]#.astype(np.int16)
                lookup_dict[date][sequence][frame.replace('.bin', '')] = snow_indices

                # snow statistics

                n_snow = (keep_mask == 0).sum()

                if f == 0:
                    first_n_snow = n_snow

                if n_snow > max_n_snow:
                    max_n_snow = n_snow

                if n_snow < min_n_snow:
                    min_n_snow = n_snow

                avg_n_snow = (avg_n_snow * f + n_snow) / (f+1)

                # cube statistics

                n_cube = cube_mask.sum()

                if n_cube > max_n_cube:
                    max_n_cube = n_cube

                if n_cube < min_n_cube:
                    min_n_cube = n_cube

                avg_n_cube = (avg_n_cube * f + n_cube) / (f + 1)

                pbar_frames.set_postfix({'cube': f'{int(n_cube)}',
                                         'snow': f'{int(n_snow)}'})

            log.info(f'{date}/{sequence}   1st_snow: {int(first_n_snow):>4}, '
                                         f'min_snow: {int(min_n_snow):>4}, '
                                         f'avg_snow: {int(avg_n_snow):>4}, '
                                         f'max_snow: {int(max_n_snow):>4}, '
                                         f'min_cube: {int(min_n_cube):>4}, '
                                         f'avg_cube: {int(avg_n_cube):>4}, '
                                         f'max_cube: {int(max_n_cube):>4}')

            stats_dict[date][sequence] = {'1st': int(first_n_snow),
                                          'min': int(min_n_snow),
                                          'max': int(max_n_snow),
                                          'avg': int(avg_n_snow)}

    suffix = '_crop' if args.crop else ''

    stats_save_path = Path(__file__).parent.resolve() / 'data' / f'cadc_stats{suffix}.pkl'

    with open(stats_save_path, 'wb') as f:
        pickle.dump(stats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    lookup_save_path = Path(__file__).parent.resolve() / 'data' / f'cadc_dror{suffix}.pkl'

    with open(lookup_save_path, 'wb') as f:
        pickle.dump(lookup_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_dense(args: argparse.Namespace):

    pcdet_path = Path(__file__).parent.resolve().parent.parent.parent.parent

    info_path = pcdet_path / 'data' / 'dense' / 'test_not_in_use' / args.pkl

    if not info_path.exists():
        info_path = str(info_path).replace('test_not_in_use', 'test_in_use')

    sensor = args.sensor
    signal = args.signal
    alpha = f'alpha_{args.alpha}'

    variant = 'crop' if args.crop else 'full'
    split = args.pkl.replace('.pkl', '').replace('dense_infos_', '').replace('_vlp32', '')

    dense_infos = []

    with open(info_path, 'rb') as i:
        infos = pickle.load(i)
        dense_infos.extend(infos)

    save_folder = Path.home() / 'Downloads' / 'DROR' / alpha / split / sensor / signal / variant
    save_folder.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(len(dense_infos)), desc='_'.join(str(save_folder).split('/')[-4:]))

    min_n_snow = np.inf
    max_n_snow = 0
    avg_n_snow = 0

    min_n_cube = np.inf
    max_n_cube = 0
    avg_n_cube = 0

    for i in pbar:

        info = dense_infos[i]

        file = dict(info)['point_cloud']['lidar_idx']
        save_path = save_folder / f'{file}.pkl'

        if save_path.exists():
            continue

        try:
            pc = load_dense_pointcloud(file=file, sensor=sensor, signal=signal)
        except FileNotFoundError:
            continue

        cube_mask = get_cube_mask(pc)

        if args.crop:
            pc = pc[cube_mask]

        if len(pc) == 0:
            snow_indices = []
            n_snow = 0
        else:
            keep_mask = dynamic_radius_outlier_filter(pc, alpha=args.alpha)
            snow_indices = (keep_mask == 0).nonzero()[0]#.astype(np.int16)
            n_snow = (keep_mask == 0).sum()

        with open(save_path, 'wb') as f:
            pickle.dump(snow_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

        # snow statistics

        if n_snow > max_n_snow:
            max_n_snow = n_snow

        if n_snow < min_n_snow:
            min_n_snow = n_snow

        avg_n_snow = (avg_n_snow * i + n_snow) / (i + 1)

        # cube statistics

        n_cube = cube_mask.sum()

        if n_cube > max_n_cube:
            max_n_cube = n_cube

        if n_cube < min_n_cube:
            min_n_cube = n_cube

        avg_n_cube = (avg_n_cube * i + n_cube) / (i + 1)

        pbar.set_postfix({'cube': f'{int(n_cube)}',
                          'snow': f'{int(n_snow)}'})


# adapted from https://github.com/mpitropov/cadc_devkit/blob/master/other/filter_pointcloud.py#L13-L50
def dynamic_radius_outlier_filter(pc: np.ndarray, alpha: float = 0.16, beta: float = 3.0,
                                  k_min: int = 3, sr_min: float = 0.04) -> np.ndarray:
    """
    :param pc:      pointcloud
    :param alpha:   horizontal angular resolution of the lidar
    :param beta:    multiplication factor
    :param k_min:   minimum number of neighbors
    :param sr_min:  minumum search radius

    :return:        mask [False = snow, True = no snow]
    """

    pc = pcl.PointCloud(pc[:, :3])

    num_points = pc.size

    # initialize mask with False
    mask = np.zeros(num_points, dtype=bool)

    k = k_min + 1

    kd_tree = pc.make_kdtree_flann()

    for i in range(num_points):

        x = pc[i][0]
        y = pc[i][1]

        r = np.linalg.norm([x, y], axis=0)

        sr = alpha * beta * np.pi / 180 * r

        if sr < sr_min:
            sr = sr_min

        [_, sqdist] = kd_tree.nearest_k_search_for_point(pc, i, k)

        neighbors = -1      # start at -1 since it will always be its own neighbour

        for val in sqdist:
            if np.sqrt(val) < sr:
                neighbors += 1

        if neighbors >= k_min:
            mask[i] = True  # no snow -> keep

    return mask


if __name__ == '__main__':

    arguments = parse_args()

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY_H:M:S
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    this_file_location = Path(__file__).parent.resolve()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)8s  %(message)s', "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=this_file_location / 'logs' / f'{arguments.dataset}_{dt_string}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console)

    arguments.logger = logger

    for c in [True, False]:

        arguments.crop = c

        if arguments.dataset == 'DENSE':

            for se in ['hdl64', 'vlp32']:

                arguments.sensor = se

                for si in ['strongest', 'last']:

                    arguments.signal = si

                    for key, value in arguments.__dict__.items():
                        if value is not None and key != 'logger':
                            logger.info(f'{key:<10}: {value}')

                    process_dense(args=arguments)

        else:

            for key, value in arguments.__dict__.items():
                if value is not None:
                    logger.info(f'{key:<10}: {value}')

            process_cadc(args=arguments)

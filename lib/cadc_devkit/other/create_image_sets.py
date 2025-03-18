__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import pickle
import numpy as np
from tqdm import tqdm
from os import listdir
from typing import Dict
from pathlib import Path
from operator import itemgetter
from os.path import isfile, join
from matplotlib import pyplot as plt

# DROR snow intensity thresholds
DROR_LEVELS = {'none':    ( 0,   9),
               'light':   (10,  79)}

DROR = Path(__file__).parent.parent.parent.parent.absolute()



def create_dror_subsets(filename: str, alpha: float = 0.45) -> None:

    sensor = 'vlp32' if 'vlp32' in filename else 'hdl64'
    signal = 'last' if 'last' in filename else 'strongest'

    path = DROR / f'alpha_{alpha}' / 'all' / sensor / signal / 'crop'

    with open(filename) as f:
        lines = f.readlines()

    filepath = Path(filename)

    title = filepath.stem

    pkl_files = [path / f'{"_".join(line.rstrip().split(","))}.pkl' for line in lines]

    tuple_dict = {'none': [],
                  'light': [],
                  'heavy': []}

    tbar = tqdm(pkl_files, desc=title)

    for pkl_file in tbar:

        p = pkl_file.stem.split('_')
        line = f'{p[0]}_{p[1]},{p[2]}'

        with open(pkl_file, 'rb') as f:
            snow_indices = pickle.load(f)

        num_snow = len(snow_indices)

        heavy = True  # because we can not create a range with stop=np.inf

        for key, value in DROR_LEVELS.items():

            if num_snow in range(value[0], value[1] + 1):

                tuple_dict[key].append((num_snow, line))
                heavy = False
                break

        if heavy:
            tuple_dict['heavy'].append((num_snow, line))

    intensity_dict = {}

    for key, tuple_list in tuple_dict.items():

        sorted_tuple_list = sorted(tuple_list, key=itemgetter(0))

        intensity_dict[key] = [tuple_item[1] for tuple_item in sorted_tuple_list]

    create_histogram(intensity_dict, title, len(pkl_files))

    save_image_sets(intensity_dict, filepath, alpha)


def save_image_sets(dictionary: Dict, filepath: Path, alpha: float) -> None:

    for key, lines in dictionary.items():

        if len(lines) > 0:

            savepath = str(filepath).replace('.txt', f'_dror_alpha_{alpha}_{key}.txt')

            with open(savepath, 'w') as f:
                for line in lines:
                    f.write(f'{line}\n')


def create_histogram(dictionary: Dict, title: str, total: int, show: bool = False) -> None:

    plt.rcdefaults()
    fig, ax = plt.subplots()

    intensities = tuple(dictionary.keys())
    x_pos = np.arange(len(intensities))
    occurances = [len(v) for v in dictionary.values()]

    rectangles = ax.bar(x_pos, occurances, align='center')
    autolabel(ax, rectangles)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(intensities)
    ax.set_ylabel('# frames')
    ax.set_title(f'{title} ({total})')

    savedir = Path(__file__).parent.resolve() / 'histograms'
    savedir.mkdir(parents=True, exist_ok=True)

    plt.savefig(savedir / f'{title}.png')

    if show:
        plt.show()


def autolabel(ax, rects):
    """
    Attach a text label to each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height, height, ha='center', va='bottom')


if __name__ == '__main__':

    image_sets = Path().home() / 'repositories' / 'PCDet' / 'data' / 'dense' / 'ImageSets'

    files = sorted([join(image_sets, f) for f in listdir(image_sets) if isfile(join(image_sets, f))])

    for file in files:
        if 'test_snow' in file and 'dror' not in file:
            create_dror_subsets(file)

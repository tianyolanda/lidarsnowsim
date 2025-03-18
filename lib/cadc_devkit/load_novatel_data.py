import os

from pathlib import Path

def load_novatel_data(novatel_path: Path):

    files = os.listdir(novatel_path)
    novatel = []

    for file in sorted(files):

        with open(novatel_path / file) as fp:
            novatel.append(fp.readline().split(' '))

    return novatel
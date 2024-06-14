"""Create leave-one-out average of other subject's data."""

import h5py
import numpy as np
from constants import SUBS, TRS
from tqdm import tqdm
from util.path import Path


def compute(narrative: str, **kwargs):

    # read all subjects
    m = 81924
    n = len(SUBS[narrative])
    Y_bold = np.zeros((n, TRS[narrative], m), dtype=np.float32)
    for i, sub in enumerate(tqdm(SUBS[narrative], desc="read")):
        boldpath = Path(
            root="data/derivatives/clean",
            datatype="func",
            sub=f"{sub:03d}",
            task=narrative,
            space="fsaverage6",
            ext=".h5",
        )

        with h5py.File(boldpath, "r") as f:
            Y_bold[i] = f["bold"][...]

    # write out subjects
    for i, sub in enumerate(tqdm(SUBS[narrative], desc="write")):
        others = list(range(n))
        others.remove(i)

        boldpath = Path(
            root="data/derivatives/leaveout",
            datatype="func",
            sub=f"{sub:03d}",
            task=narrative,
            space="fsaverage6",
            ext=".h5",
        )
        boldpath.mkdirs()

        boldO = Y_bold[others].mean(0)
        with h5py.File(boldpath, "w") as f:
            f.create_dataset(name="bold", data=boldO)

    # write group average
    boldpath = Path(
        root="data/derivatives/group",
        datatype="func",
        sub="000",
        task=narrative,
        space="fsaverage6",
        ext=".h5",
    )
    boldpath.mkdirs()
    with h5py.File(boldpath, "w") as f:
        f.create_dataset(name="bold", data=Y_bold.mean(0))


def main(narratives: str, **kwargs):
    for narrative in narratives:
        compute(narrative, **kwargs)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--narratives", type=str, nargs="+", default=["black", "forgot"]
    )

    main(**vars(parser.parse_args()))

"""Run confound regression on fMRI-prepped BOLD signal."""

import warnings

import h5py
import numpy as np
import pandas as pd
from constants import CONFOUND_REGRESSORS, NARRATIVE_SLICE, SUBS, TR
from tqdm import tqdm
from util import subject
from util.path import Path


def get_bold(sub: int, narrative: str) -> np.ndarray:
    boldpath = Path(
        root="data/derivatives/fmriprep/",
        datatype="func",
        sub=f"{sub:03d}",
        task=narrative,
        space="fsaverage6",
        hemi="L",
        ext=".func.gii",
    )
    boldpath.update(sub=f"{sub:03d}")
    paths = [boldpath, boldpath.copy().update(hemi="R")]

    confpath = boldpath.copy()
    del confpath["hemi"]
    del confpath["space"]
    confpath.update(desc="confounds", suffix="regressors", ext=".tsv")

    # TODO make these arguments?
    confdata = pd.read_csv(confpath, sep="\t", usecols=CONFOUND_REGRESSORS)
    confdata.bfill(inplace=True)

    masker = subject.GiftiMasker(
        t_r=TR[narrative],
        ensure_finite=True,
        standardize="zscore_sample",
        standardize_confounds=True,
    )
    Y_bold = masker.fit_transform(paths, confounds=confdata.to_numpy())

    Y_bold = Y_bold[NARRATIVE_SLICE[narrative]]

    return Y_bold


def main(narratives: str, **kwargs):
    for narrative in narratives:
        for sub_id in tqdm(SUBS[narrative], desc=narrative):

            boldpath = Path(
                root="data/derivatives/clean",
                datatype="func",
                sub=f"{sub_id:03d}",
                task=narrative,
                space="fsaverage6",
                ext=".h5",
            )
            boldpath.mkdirs()

            Y_bold = get_bold(sub_id, narrative)

            with h5py.File(boldpath, "w") as f:
                f.create_dataset(name="bold", data=Y_bold)


if __name__ == "__main__":
    from argparse import ArgumentParser

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--narratives", type=str, nargs="+", default=["black", "forgot"]
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    main(**vars(parser.parse_args()))

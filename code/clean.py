"""Run confound regression on fMRI-prepped BOLD signal."""

import warnings

import h5py
import numpy as np
import pandas as pd
from constants import CONFOUND_MODEL, NARRATIVE_SLICE, SUBS, TR, TRS
from extract_confounds import extract_confounds, load_confounds
from tqdm import tqdm
from util import subject
from util.path import Path


def get_nuisance_regressors(narrative: str):

    filename = f"data/stimuli/whisperx/narrative-{narrative}.csv"
    df = pd.read_csv(filename, index_col=0)

    word_onsets = np.zeros(TRS[narrative], dtype=np.float32)
    word_rates = np.zeros(TRS[narrative], dtype=np.float32)
    for tr in range(TRS[narrative]):
        subdf = df[df.TR == tr]
        if len(subdf):
            word_onsets[tr] = 1
            word_rates[tr] = len(subdf)

    return word_onsets, word_rates


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

    confounds_fn = confpath.fpath
    confounds_df, confounds_meta = load_confounds(confounds_fn)
    confounds = extract_confounds(confounds_df, confounds_meta, CONFOUND_MODEL)

    word_onsets, word_rates = get_nuisance_regressors(narrative)
    task_confounds = np.zeros((len(confounds), 2))
    task_confounds[NARRATIVE_SLICE[narrative], 0] = word_onsets
    task_confounds[NARRATIVE_SLICE[narrative], 1] = word_rates

    all_confounds = np.hstack((confounds.to_numpy(), task_confounds))

    masker = subject.GiftiMasker(
        t_r=TR[narrative],
        detrend=True,
        ensure_finite=True,
        standardize="zscore_sample",
        standardize_confounds=True,
    )
    Y_bold = masker.fit_transform(paths, confounds=all_confounds)

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

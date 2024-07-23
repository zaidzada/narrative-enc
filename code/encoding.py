"""Story encoding"""

from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
from constants import SUBS, TR, TRS
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import correlation_score_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from util.atlas import Atlas
from util.path import Path
from voxelwise_tutorials.delayer import Delayer


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


def _get_llm_embs(narrative: str, modelname: str, layer: int):

    filename = f"data/features/{modelname}/narrative-{narrative}.pkl"
    df = pd.read_pickle(filename)
    df["start"] = df["start"].ffill()
    df["TR"] = df["start"].divide(TR[narrative]).apply(np.floor).apply(int)

    filename = f"data/features/{modelname}/narrative-{narrative}.h5"
    with h5py.File(filename, "r") as f:
        states = f["activations"][layer + 1]
    df["embedding"] = [e for e in states]

    n_features = df.iloc[0].embedding.size
    embeddings = np.zeros((TRS[narrative], n_features), dtype=np.float32)
    for tr in range(TRS[narrative]):
        subdf = df[df.TR == tr]
        if len(subdf):
            embeddings[tr] = subdf.embedding.mean(0)

    return embeddings


def _get_pickle_feature(feature_name: str, narrative: str):
    filename = f"data/features/{feature_name}/narrative-{narrative}.pkl"
    df = pd.read_pickle(filename)
    df["TR"] = df["start"].divide(TR[narrative]).apply(np.floor).apply(int)
    n_features = df.iloc[0].embedding.size

    embeddings = np.zeros((TRS[narrative], n_features), dtype=np.float32)
    for tr in range(TRS[narrative]):
        subdf = df[df.TR == tr]
        if len(subdf):
            embeddings[tr] = subdf.embedding.mean(0)

    return embeddings


def get_feature(feature_name: str, narrative: str, **kwargs):
    if feature_name == "acoustic":
        return np.load(f"data/features/spectrogram/narrative-{narrative}.npy")
    elif feature_name == "articulatory":
        return _get_pickle_feature(feature_name, narrative)
    elif feature_name == "syntactic":
        return _get_pickle_feature(feature_name, narrative)
    elif feature_name == "nuisance":
        return get_nuisance_regressors(narrative)
    else:
        return _get_llm_embs(narrative, modelname=feature_name, **kwargs)


def get_bold(sub_id: int, narrative: str) -> np.ndarray:
    boldpath = Path(
        root="data/derivatives/clean/",
        datatype="func",
        sub=f"{sub_id:03d}",
        task=narrative,
        space="fsaverage6",
        ext=".h5",
    )

    with h5py.File(boldpath, "r") as f:
        bold = f["bold"][...]

    return bold


def build_regressors(narrative: str, modelname: str, **kwargs):
    word_onsets, word_rates = get_feature("nuisance", narrative)
    lexical_embs = get_feature(modelname, narrative, **kwargs)

    # shift
    # if True:
    #     lexical_embs = np.roll(lexical_embs, shift=len(lexical_embs) // 2, axis=0)

    X = np.hstack(
        (
            word_onsets.reshape(-1, 1),
            word_rates.reshape(-1, 1),
            lexical_embs,
        )
    )

    slices = {
        "nuisance": slice(0, 2),
        "lexical": slice(2, X.shape[1]),
    }

    return X, slices


def build_model(
    feature_names: list[str],
    slices: list[slice],
    alphas: np.ndarray,
    n_jobs: int,
    verbose: int = 0,
):
    """Build the pipeline"""

    # Set up modeling pipeline
    delayer_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Delayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )

    # Make kernelizer
    kernelizers_tuples = [
        (name, delayer_pipeline, slice_) for name, slice_ in zip(feature_names, slices)
    ]
    column_kernelizer = ColumnKernelizer(kernelizers_tuples, n_jobs=n_jobs)

    params = dict(
        alphas=alphas,
        progress_bar=verbose,
        n_iter=100,
        diagonalize_method="svd",
    )
    mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver_params=params)
    pipeline = make_pipeline(
        column_kernelizer,
        mkr_model,
    )

    return pipeline


def get_average_sub(narrative: str):
    return get_sub_bold(narrative).mean(0)


def get_sub_bold(narrative: str):
    m = 81924
    n = len(SUBS[narrative])
    Y_bold = np.zeros((n, TRS[narrative], m), dtype=np.float32)
    for i, sub in enumerate(tqdm(SUBS[narrative])):
        Y_bold[i] = get_bold(sub, narrative)
    return Y_bold


def encoding(
    narrative: str,
    modelname: str,
    layer: int,
    alphas: list,
    jobs: int,
    group_sub: bool,
    folds: int,
    suffix: str,
    **kwargs,
):
    X, features = build_regressors(narrative, modelname, layer=layer)
    feature_names = list(features.keys())
    slices = list(features.values())

    pipeline = build_model(feature_names, slices, alphas, jobs)

    narrative2 = "forgot" if narrative == "black" else "black"
    X2, _ = build_regressors(narrative2, modelname, layer=layer)

    atlas = Atlas.schaefer(parcels=1000, networks=17, kong=True)

    subs = SUBS[narrative]
    if group_sub:
        subs = [0]

    for sub_id in tqdm(subs):
        results = defaultdict(list)

        if group_sub:
            Y_bold = get_average_sub(narrative)
        else:
            Y_bold = get_bold(sub_id, narrative)

        # cross-story
        if folds == 1:
            if group_sub:
                Y_bold2 = get_average_sub(narrative2)
            else:
                Y_bold2 = get_bold(sub_id, narrative2)

            Y_bold = atlas.vox_to_parc(Y_bold)
            Y_bold2 = atlas.vox_to_parc(Y_bold2)

            Y_bold = StandardScaler().fit_transform(Y_bold)
            Y_bold2 = StandardScaler().fit_transform(Y_bold2)

            pipeline.fit(X, Y_bold)
            Y_preds = pipeline.predict(X2, split=True)
            scores_split = correlation_score_split(Y_bold2, Y_preds)
            results[f"{narrative2}_actual"].append(Y_bold2)
            results[f"{narrative2}_scores"].append(scores_split.numpy(force=True))
            results[f"{narrative2}_preds"].append(Y_preds.numpy(force=True))

            pipeline.fit(X2, Y_bold2)
            Y_preds = pipeline.predict(X, split=True)
            scores_split = correlation_score_split(Y_bold, Y_preds)
            results[f"{narrative}_actual"].append(Y_bold)
            results[f"{narrative}_scores"].append(scores_split.numpy(force=True))
            results[f"{narrative}_preds"].append(Y_preds.numpy(force=True))

            result = {k: v[0] for k, v in results.items()}
        else:
            K = folds
            kfold = KFold(n_splits=K)
            for train_index, test_index in tqdm(kfold.split(X), leave=False, total=K):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y_bold[train_index], Y_bold[test_index]

                Y_train = StandardScaler().fit_transform(Y_train)
                Y_test = StandardScaler().fit_transform(Y_test)

                pipeline.fit(X_train, Y_train)

                Y_preds = pipeline.predict(X_test, split=True)
                scores_split = correlation_score_split(Y_test, Y_preds)

                results["cv_actual"].append(Y_test)
                results["cv_scores"].append(scores_split.numpy(force=True))
                results["cv_preds"].append(Y_preds.numpy(force=True))
            result = {k: np.stack(v) for k, v in results.items()}

        # save
        pklpath = Path(
            root=f"results/encoding{suffix}",
            sub=f"{sub_id:03d}",
            datatype=modelname + f"_{layer}-layer",
            ext="h5",
        )
        pklpath.mkdirs()
        with h5py.File(pklpath, "w") as f:
            for key, value in result.items():
                f.create_dataset(name=key, data=value)


def main(*args, **kwargs):
    if kwargs["device"] == "cuda":
        set_backend("torch_cuda")
        print("Set backend to torch cuda")
    encoding(*args, **kwargs)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--modelname", type=str, default="gemma-2b")
    parser.add_argument("-l", "--layer", type=int, default=16)
    parser.add_argument("-n", "--narrative", type=str, default="black")
    parser.add_argument("-s", "--suffix", type=str, default="")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("-k", "--folds", type=int, default=1)
    parser.add_argument("--alphas", default=np.logspace(0, 19, 20))
    parser.add_argument("--group-sub", action="store_true")

    main(**vars(parser.parse_args()))

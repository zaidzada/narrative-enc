"""Story encoding"""

from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
from constants import CONFOUND_REGRESSORS, NARRATIVE_SLICE, SUBS, TR, TRS
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import correlation_score_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from util import subject
from util.path import Path
from voxelwise_tutorials.delayer import Delayer

# def get_spectral_features():
#     raise NotImplementedError
#     from transformers import AutoFeatureExtractor
#     from whisperx import load_audio

#     SAMPLING_RATE = 16000.0

#     feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")

#     audiopath = "mats/black_audio.wav"
#     audio = load_audio(audiopath)  # exactly 800 s
#     n_chunks = np.ceil(audio.size / (30 * SAMPLING_RATE))  # 30 s each
#     chunks = np.array_split(audio, n_chunks)
#     features = feature_extractor(chunks, sampling_rate=SAMPLING_RATE)
#     features = np.hstack(features["input_features"])

#     chunks = np.array_split(features, TRS, axis=1)
#     features = np.hstack([c.mean(axis=1, keepdims=True) for c in chunks])
#     features = features.T

#     return features


def get_transcript_features(narrative: str):

    filename = f"data/stimuli/gentle/{narrative}/align.csv"
    df = pd.read_csv(filename, names=["word", "lemma", "onset", "offset"])
    df["onset"] = df["onset"].ffill()
    df["TR"] = df.onset.divide(TR[narrative]).apply(np.floor).apply(int)

    word_onsets = np.zeros(TRS[narrative], dtype=np.float32)
    word_rates = np.zeros(TRS[narrative], dtype=np.float32)
    for tr in range(TRS[narrative]):
        subdf = df[df.TR == tr]
        if len(subdf):
            word_onsets[tr] = 1
            word_rates[tr] = len(subdf)

    return word_onsets, word_rates


def get_llm_embs(narrative: str, modelname: str):
    filename = f"features/{modelname}/desc-{narrative}.h5"
    df = pd.read_hdf(filename, key="df")
    df["start"] = df["start"].ffill()
    df["TR"] = df["start"].divide(TR[narrative]).apply(np.floor).apply(int)

    n_features = df.iloc[0].embedding.size

    embeddings = np.zeros((TRS[narrative], n_features), dtype=np.float32)
    for tr in range(TRS[narrative]):
        subdf = df[df.TR == tr]
        if len(subdf):
            embeddings[tr] = subdf.embedding.mean(0)

    return embeddings


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


def build_regressors(narrative: str, modelname: str = None):
    word_onsets, word_rates = get_transcript_features(narrative)
    lexical_embs = get_llm_embs(narrative, modelname)

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
    verbose: int,
    n_jobs: int,
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
        alphas=alphas, progress_bar=verbose, n_iter=100  # , diagonalize_method="svd",
    )
    mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver_params=params)
    pipeline = make_pipeline(
        column_kernelizer,
        mkr_model,
    )

    return pipeline


# def get_average_sub(narrative: str):
#     # average subject
#     Y_bold = []
#     for sub in tqdm(SUBS[narrative], desc="prep"):
#         Y_bold.append(get_bold(sub, narrative))
#     Y_bold = np.stack(Y_bold).mean(0)
#     return Y_bold


def get_average_sub(narrative: str):
    return get_sub_bold(narrative).mean(0)


def get_sub_bold(narrative: str):
    m = 81924
    n = len(SUBS[narrative])
    Y_bold = np.zeros((n, TRS[narrative], m), dtype=np.float32)
    for i, sub in enumerate(tqdm(SUBS[narrative])):
        Y_bold[i] = get_bold(sub, narrative)
    return Y_bold


def main(args):
    X, features = build_regressors(args.narrative, args.model)
    feature_names = list(features.keys())
    slices = list(features.values())

    pipeline = build_model(feature_names, slices, args.alphas, args.verbose, args.jobs)

    narrative2 = "forgot" if args.narrative == "black" else "black"
    X2, _ = build_regressors(narrative2, args.model)

    subs = SUBS[args.narrative]
    if args.group_sub:
        subs = [0]

    for sub in tqdm(subs):
        results = defaultdict(list)

        if args.group_sub:
            Y_bold = get_average_sub(args.narrative)
        else:
            Y_bold = get_bold(sub, args.narrative)

        # cross-story
        if args.folds == 1:
            if args.group_sub:
                Y_bold2 = get_average_sub(narrative2)
            else:
                Y_bold2 = get_bold(sub, narrative2)

            Y_bold = StandardScaler().fit_transform(Y_bold)
            Y_bold2 = StandardScaler().fit_transform(Y_bold2)

            pipeline.fit(X, Y_bold)
            Y_preds = pipeline.predict(X2, split=True)
            scores_split = correlation_score_split(Y_bold2, Y_preds)
            results["actual1"].append(Y_bold2)
            results["scores1"].append(scores_split.numpy(force=True))
            results["preds1"].append(Y_preds.numpy(force=True))

            pipeline.fit(X2, Y_bold2)
            Y_preds = pipeline.predict(X, split=True)
            scores_split = correlation_score_split(Y_bold, Y_preds)
            results["actual2"].append(Y_bold)
            results["scores2"].append(scores_split.numpy(force=True))
            results["preds2"].append(Y_preds.numpy(force=True))

            result = {k: v[0] for k, v in results.items()}
        else:
            K = args.folds
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
            root=f"encoding{args.suffix}/{args.narrative}",
            sub=f"{sub:03d}",
            datatype=args.model,
            ext="h5",
        )
        pklpath.mkdirs()
        with h5py.File(pklpath, "w") as f:
            for key, value in result.items():
                f.create_dataset(name=key, data=value)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="model-gemma-2b_layer-16")
    parser.add_argument("-n", "--narrative", type=str, default="black")
    parser.add_argument("-s", "--suffix", type=str, default="")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("-k", "--folds", type=int, default=1)
    parser.add_argument("--group-sub", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.alphas = np.logspace(0, 19, 20)
    print(vars(args))

    if args.cuda > 0:
        if torch.cuda.is_available():
            set_backend("torch_cuda")
        else:
            print("[WARN] cuda not available")

    main(args)

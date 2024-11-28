"""Story joint encoding"""

import h5py
import numpy as np
import torch
from constants import SUBS
from encoding import build_model, get_bold, get_feature
from himalaya.backend import set_backend
from himalaya.scoring import correlation_score_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from util.atlas import Atlas
from util.path import Path


def build_regressors(narrative: str, modelname: list[str], **kwargs):

    X = []
    start = 0
    slices = {}
    for feature_space in modelname:

        values = get_feature(feature_space, narrative, **kwargs)
        if isinstance(values, list) or isinstance(values, tuple):
            x_features = np.stack(values).T
        else:
            x_features = values

        X.append(x_features)
        slices[feature_space] = slice(start, start + x_features.shape[1])
        start += x_features.shape[1]

        # print(narrative, feature_space, slices[feature_space], x_features.shape)

    X = np.hstack(X)
    return X, slices


def encoding(
    alias: str,
    narrative: str,
    modelname: list[str],
    layer: int,
    alphas: list,
    jobs: int,
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

    for sub_id in tqdm(subs):
        results = {}

        Y_bold = get_bold(sub_id, narrative)
        Y_bold2 = get_bold(sub_id, narrative2)

        Y_bold = atlas.vox_to_parc(Y_bold)
        Y_bold2 = atlas.vox_to_parc(Y_bold2)

        Y_bold = StandardScaler().fit_transform(Y_bold)
        Y_bold2 = StandardScaler().fit_transform(Y_bold2)

        # fit one first story
        pipeline.fit(X, Y_bold)
        Y_preds = pipeline.predict(X2, split=True)
        scores_split = correlation_score_split(Y_bold2, Y_preds)
        results[f"{narrative2}_actual"] = Y_bold2
        results[f"{narrative2}_scores"] = scores_split.numpy(force=True)
        results[f"{narrative2}_preds"] = Y_preds.numpy(force=True)

        # Xfit = pipeline["columnkernelizer"].get_X_fit()
        # weights = pipeline["multiplekernelridgecv"].get_primal_coef(Xfit)
        # weights_embs = weights[-1]  # take second band weights
        # weights_embs_delay = weights_embs.reshape(-1, 4, weights_embs.shape[-1])
        # weights_embs = weights_embs_delay.mean(1)
        # results[f"{narrative2}_weights"] = weights_embs

        # fit one second story
        pipeline.fit(X2, Y_bold2)
        Y_preds = pipeline.predict(X, split=True)
        scores_split = correlation_score_split(Y_bold, Y_preds)
        results[f"{narrative}_actual"] = Y_bold
        results[f"{narrative}_scores"] = scores_split.numpy(force=True)
        results[f"{narrative}_preds"] = Y_preds.numpy(force=True)

        # Xfit = pipeline["columnkernelizer"].get_X_fit()
        # weights = pipeline["multiplekernelridgecv"].get_primal_coef(Xfit)
        # weights_embs = weights[-1]
        # weights_embs_delay = weights_embs.reshape(-1, 4, weights_embs.shape[-1])
        # weights_embs = weights_embs_delay.mean(1)
        # results[f"{narrative}_weights"] = weights_embs

        # save
        pklpath = Path(
            root=f"results/joint_encoding{suffix}",
            sub=f"{sub_id:03d}",
            datatype=alias,
            ext="h5",
        )
        pklpath.mkdirs()
        with h5py.File(pklpath, "w") as f:
            for key, value in results.items():
                f.create_dataset(name=key, data=value)


def main(*args, **kwargs):
    if kwargs["device"] == "cuda":
        set_backend("torch_cuda")
        print("Set backend to torch cuda")
    encoding(*args, **kwargs)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-a", "--alias", type=str, default="NASL")
    parser.add_argument(
        "-m",
        "--modelname",
        type=str,
        nargs="+",
        default=["nuisance", "acoustic", "syntactic", "gemma2-9b"],
    )
    parser.add_argument("-l", "--layer", type=int, default=22)
    parser.add_argument("-n", "--narrative", type=str, default="black")
    parser.add_argument("-s", "--suffix", type=str, default="")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--alphas", default=np.logspace(0, 19, 20))

    main(**vars(parser.parse_args()))

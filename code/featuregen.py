"""Create features from stimuli to use for encoding models."""

import json

import numpy as np
import pandas as pd
from constants import ARPABET_PHONES, PUNCTUATION, TR, TRS
from tqdm import tqdm
from util.path import Path


def phonemes(narratives: list[str], mode="articulatory", **kwargs):
    from io import StringIO

    from nltk.corpus import cmudict

    arpabet = cmudict.dict()
    phone_set = ARPABET_PHONES
    phonedict = {ph: i for i, ph in enumerate(phone_set)}

    dfp = pd.read_csv(
        StringIO(
            """phoneme,kind,manner,place,phonation,height,position,height2,position2
B,consonant,plosive,bilabial,voiced,,,,
CH,consonant,affricate,postalveolar,unvoiced,,,,
D,consonant,plosive,alveolar,voiced,,,,
DH,consonant,fricative,dental,voiced,,,,
F,consonant,fricative,labiodental,unvoiced,,,,
G,consonant,plosive,velar,voiced,,,,
HH,consonant,fricative,glottal,unvoiced,,,,
JH,consonant,affricate,postalveolar,voiced,,,,
K,consonant,plosive,velar,unvoiced,,,,
L,consonant,lateral,alveolar,voiced,,,,
M,consonant,nasal,bilabial,voiced,,,,
N,consonant,nasal,alveolar,voiced,,,,
NG,consonant,nasal,velar,voiced,,,,
P,consonant,plosive,bilabial,unvoiced,,,,
R,consonant,approximant,alveolar,voiced,,,,
S,consonant,fricative,alveolar,unvoiced,,,,
SH,consonant,fricative,postalveolar,unvoiced,,,,
T,consonant,plosive,alveolar,unvoiced,,,,
TH,consonant,fricative,dental,unvoiced,,,,
V,consonant,fricative,labiodental,voiced,,,,
W,consonant,approximant,velar,voiced,,,,
Y,consonant,approximant,palatal,voiced,,,,
Z,consonant,fricative,alveolar,voiced,,,,
ZH,consonant,fricative,postalveolar,voiced,,,,
AA,vowel,,,,low,back,,
AE,vowel,,,,low,front,,
AH,vowel,,,,mid,central,,
AO,vowel,,,,mid,back,,
AW,vowel,,,,low,central,mid,back
AY,vowel,,,,low,central,mid,front
EH,vowel,,,,mid,front,,
ER,vowel,,,,mid,central,,
EY,vowel,,,,mid,front,,
IH,vowel,,,,mid,front,,
IY,vowel,,,,high,front,,
OW,vowel,,,,mid,back,,
OY,vowel,,,,mid,back,high,front
UH,vowel,,,,high,back,,
UW,vowel,,,,high,back,,"""
        )
    )

    # Build articulatory features of phonemes
    features = dfp.fillna("NA").iloc[:, 2:].values.flatten()
    feature_set = {f: i for i, f in enumerate(np.unique(features[features != "NA"]))}
    phone_embs = {}
    for i, row in dfp.iterrows():
        emb = np.zeros(len(feature_set))
        for feature in row.iloc[2:].dropna():
            emb[feature_set[feature]] = 1
        phone_embs[row.phoneme] = emb

    def get_word_phone_features(word):
        emb = np.zeros(len(feature_set))
        if phones := arpabet.get(word.lower()):
            for phone in phones[0]:
                emb += phone_embs[phone.strip("012")]
        return emb

    def get_word_phone_emb(word):
        emb = np.zeros(len(phone_set))
        if phones := arpabet.get(word.lower()):
            for phone in phones[0]:
                emb[phonedict[phone.strip("012")]] = 1
        return emb

    func = get_word_phone_emb
    if mode == "articulatory":
        func = get_word_phone_features

    for narrative in tqdm(narratives):
        filename = f"data/stimuli/whisperx/{narrative}_audio.json"
        with open(filename, "r") as f:
            d = json.load(f)
            words = []
            for i, segment_dict in enumerate(d["segments"], 1):
                for word_dict in segment_dict["words"]:
                    word_dict["segment"] = i
                    words.append(word_dict)
            df = pd.DataFrame(words)
        df["start"] = df["start"].ffill()
        df["TR"] = df["start"].divide(TR[narrative]).apply(np.floor).apply(int)

        phone_emb = df.word.astype(str).str.strip(PUNCTUATION).apply(func)
        embeddings = np.vstack(phone_emb.values)

        # # remove uninformative dimensions
        # missingMask = embeddings.sum(0) > 0
        # if not np.all(missingMask):
        #     print("[WARNING] contains features with all 0s", missingMask.sum())
        #     embeddings = embeddings[:, missingMask]

        df["embedding"] = [e for e in embeddings]

        transpath = Path(
            root="data/features", datatype=mode, narrative=narrative, ext="pkl"
        )
        transpath.mkdirs()
        df.to_pickle(transpath)


def syntactic(narratives: list[str], **kwargs):

    import spacy
    from sklearn.preprocessing import LabelBinarizer

    nlp = spacy.load("en_core_web_lg")

    taggerEncoder = LabelBinarizer().fit(nlp.get_pipe("tagger").labels)
    dependencyEncoder = LabelBinarizer().fit(nlp.get_pipe("parser").labels)

    for narrative in tqdm(narratives):
        filename = f"data/stimuli/whisperx/{narrative}_audio.json"
        with open(filename, "r") as f:
            d = json.load(f)
            words = []
            for i, segment_dict in enumerate(d["segments"], 1):
                for word_dict in segment_dict["words"]:
                    word_dict["segment"] = i
                    words.append(word_dict)
            df = pd.DataFrame(words)
        df["start"] = df["start"].ffill()
        df["TR"] = df["start"].divide(TR[narrative]).apply(np.floor).apply(int)
        df.to_csv(f"data/stimuli/whisperx/narrative-{narrative}.csv")

        df.insert(0, "word_idx", df.index.values)
        df["word_with_ws"] = df.word.astype(str) + " "
        try:
            df["hftoken"] = df.word_with_ws.apply(nlp.tokenizer)
        except TypeError:
            print("typeerror!")
            breakpoint()
        df = df.explode("hftoken", ignore_index=True)

        features = []
        for _, sentence in df.groupby("segment"):
            # create a doc from the pre-tokenized text then parse it for features
            words = [token.text for token in sentence.hftoken.tolist()]
            spaces = [token.whitespace_ == " " for token in sentence.hftoken.tolist()]
            doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)
            doc = nlp(doc)
            for token in doc:
                features.append([token.text, token.tag_, token.dep_, token.is_stop])
        df2 = pd.DataFrame(
            features, columns=["token", "pos", "dep", "stop"], index=df.index
        )
        df = pd.concat([df, df2], axis=1)

        # generate embeddings
        a = taggerEncoder.transform(df.pos.tolist())
        b = dependencyEncoder.transform(df.dep.tolist())
        c = LabelBinarizer().fit_transform(df.stop.tolist())
        embeddings = np.hstack((a, b, c))

        # # remove uninformative dimensions
        # missingMask = embeddings.sum(0) > 0
        # if not np.all(missingMask):
        #     # print("[WARNING] contains features with all 0s", missingMask.sum())
        #     embeddings = embeddings[:, missingMask]

        df["embedding"] = [e for e in embeddings]

        # not serializable
        df.drop(["hftoken", "word_with_ws"], axis=1, inplace=True)

        transpath = Path(
            root="data/features", datatype="syntactic", narrative=narrative, ext="pkl"
        )
        transpath.mkdirs()
        df.to_pickle(transpath)


def spectral(narratives: list[str], **kwargs):
    from transformers import AutoFeatureExtractor
    from whisperx import load_audio

    SAMPLING_RATE = 16000  # Hz
    CHUNK_LENGTH = 30  # seconds

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")

    for narrative in tqdm(narratives):

        audiopath = f"data/stimuli/audio/{narrative}_audio.wav"

        audio = load_audio(audiopath)  # reads and resamples

        # Split the entire audio into ~30s chunks
        n_chunks = np.ceil(len(audio) / (CHUNK_LENGTH * SAMPLING_RATE))
        chunks = np.array_split(audio, n_chunks)

        # Extract features for each chunk and concatenate to one long array
        features = feature_extractor(chunks, sampling_rate=SAMPLING_RATE)
        features = np.hstack(features["input_features"])

        # Split array into TRs and average data within each
        chunks = np.array_split(features, TRS[narrative], axis=1)
        features = np.hstack([c.mean(axis=1, keepdims=True) for c in chunks])

        audiopath = Path(
            root="data/features", datatype="spectrogram", narrative=narrative, ext="npy"
        )
        audiopath.mkdirs()
        np.save(audiopath, features.T)


def main(**kwargs):
    for feature in kwargs["features"]:
        if feature == "spectral":
            spectral(**kwargs)
        elif feature == "articulatory":
            phonemes(**kwargs, mode="articulatory")
        elif feature == "phonemic":
            phonemes(**kwargs, mode="phonemic")
        elif feature == "syntactic":
            syntactic(**kwargs)
        else:
            raise ValueError(f"Unknown feature set: {feature}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        nargs="+",
        default=["spectral", "articulatory", "syntactic"],
    )
    parser.add_argument(
        "-n", "--narratives", type=str, nargs="+", default=["black", "forgot"]
    )
    main(**vars(parser.parse_args()))

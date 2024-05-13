"UNUSED FOR NOW"
from glob import glob

import pandas as pd
import whisperx
from util.path import Path


def word_align(
    segments: list, audio_file: Path | str, device: str = "cuda", language: str = "en"
):
    audio = whisperx.load_audio(audio_file)
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        segments, model_a, metadata, audio, device, return_char_alignments=False
    )

    return result


def align(uttdf: pd.DataFrame, audio_file: Path | str) -> pd.DataFrame:
    # Reformat into segments
    uttdf.rename(columns={"onset": "start", "offset": "end"}, inplace=True)
    segments = list(uttdf.to_dict(orient="index").values())

    result = word_align(segments, audio_file)
    new_segments = result["segments"]

    if len(new_segments) != sum(len(s["sentence_spans"]) for s in segments):
        raise ValueError("Something wrong with alignment. Double check")

    dfs = []
    i = 0
    for segment in segments:
        for j, sentspan in enumerate(segment["sentence_spans"], start=1):
            if i >= len(new_segments):
                print("wah")
                breakpoint()
            df = pd.DataFrame(new_segments[i]["words"])
            df.insert(0, "sentence", j)
            df.insert(0, "speaker", segment["speaker"])
            i += 1
            dfs.append(df)

    if i != len(new_segments):
        print("something wrong, not same number of segments as sents")
        breakpoint()

    dfn = pd.concat(dfs)
    return dfn


def main(args: dict):
    """Move and process transcripts."""

    path_pattern = "data/stimuli/transcripts/{}_transcript.csv",
    wave_pattern = "data/stimuli/{}_audio.wav",
    narratives = ['black', 'forgot']

    for narrative in narratives:
        transcript_fn = path_pattern.format(narrative)
        wave_fn = wave_pattern.format(narrative)
        print(transcript_fn, wave_fn)
        # dfn = align(transcript_fn, )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)

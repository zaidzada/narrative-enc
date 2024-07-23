"""Generate embedding for a word-aligned transcript.
"""

import json

import h5py
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from util.path import Path

# short names for long model names
HFMODELS = {
    "gemma-2b": "google/gemma-2b",
    "gemma2-9b": "google/gemma-2-9b",
    "gemma2-27b": "google/gemma-2-27b",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
}


def main(narratives: list[str], modelname: str, device: str):

    hfmodelname = HFMODELS[modelname]

    # custom settings
    tokenizer_args = dict(token="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")
    if "gpt2" in hfmodelname or "opt" in hfmodelname:
        tokenizer_args["add_prefix_space"] = True
    model_args = dict(
        token="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",
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(hfmodelname, **tokenizer_args)
    model = AutoModelForCausalLM.from_pretrained(hfmodelname, **model_args)

    print(
        f"Model : {hfmodelname}"
        f"\nLayers: {model.config.num_hidden_layers}"
        f"\nEmbDim: {model.config.hidden_size}"
        f"\nCxtLen: {model.config.max_position_embeddings}"
        f"\nDevice: {device}"
    )
    model = model.eval()
    # model = model.to(device)

    epath = Path(
        root="data/features",
        datatype=modelname,
        suffix=None,
        ext="pkl",
    )

    transcript_fpattern = "data/stimuli/whisperx/{}_audio.json"
    for narrative in tqdm(narratives):
        transcript_file = transcript_fpattern.format(narrative)
        with open(transcript_file, "r") as f:
            d = json.load(f)
            df = pd.DataFrame(d["word_segments"])
        df["start"] = df["start"].ffill()

        # Tokenize input
        df.insert(0, "word_idx", df.index.values)
        df["hftoken"] = df.word.apply(
            lambda x: tokenizer.tokenize(" " + x)
        )  # NOTE manually add space
        df = df.explode("hftoken", ignore_index=True)
        df["token_id"] = df.hftoken.apply(tokenizer.convert_tokens_to_ids)

        if len(df) >= model.config.max_position_embeddings:
            print("WARNING, MODEL TOO SHORT")

        # Set up input
        tokenids = [tokenizer.bos_token_id] + df.token_id.tolist()
        # batch = torch.tensor([tokenids], dtype=torch.long, device=device)
        batch = torch.tensor([tokenids], dtype=torch.long).to("cuda")

        # Run through model
        with torch.no_grad():
            output = model(batch, labels=batch, output_hidden_states=True)
            states = torch.stack(output.hidden_states).numpy(force=True)

            # squeeze, and skip the BOS token we manually added
            states = states[:, 0, 1:, :]  # layers x batch x seq_len x dim

            logits = output.logits[0].detach().cpu()

            batch_cpu = batch.detach().cpu()
            logits_order = logits.argsort(descending=True, dim=-1)
            ranks = torch.eq(logits_order[:-1], batch_cpu[:, 1:].T).nonzero()[:, 1]

            probs = logits[:-1, :].softmax(-1)
            true_probs = probs[0, batch_cpu[0, 1:]]

            entropy = torch.distributions.Categorical(probs=probs).entropy()

        df["rank"] = ranks.numpy(force=True)
        df["true_prob"] = true_probs.numpy(force=True)
        df["entropy"] = entropy.numpy(force=True)

        # save transcript
        epath.update(narrative=narrative, ext="pkl")
        epath.mkdirs()
        df.to_pickle(epath)

        # save embeddings
        epath.update(narrative=narrative, ext=".h5")
        with h5py.File(epath, "w") as f:
            f.create_dataset(name="activations", data=states)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--modelname", default="gemma-2b")
    parser.add_argument(
        "-n", "--narratives", type=str, nargs="+", default=["black", "forgot"]
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    main(**vars(parser.parse_args()))

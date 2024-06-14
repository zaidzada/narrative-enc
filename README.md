# model gradients

to reproduce, run these scripts in order:

1. `clean.py`
    1. `groupavg.py`
1. `transcribe.sh`
    1. `featuregen.py`
    1. `embeddings.py`
        1. `encoding.py`

# slurm

## transcribe
```
sbatch --job-name=transcribe --time=00:30:00 --mem=5G --gres=gpu:1 --partition=mig --ntasks=1 --cpus-per-task=1 code/slurm.sh \
whisperx \
    --model large-v2 \
    --output_dir data/stimuli/whisperx \
    --output_format json \
    --task transcribe \
    --language en \
    --device cuda \
    data/stimuli/audio/*.wav
```

## embeddings
```
sbatch --job-name=gen_emb --time=00:30:00 --mem=32G --gres=gpu:1 --partition=mig --ntasks=1 --cpus-per-task=1 code/slurm.sh \
code/embeddings.py
```

## encoding
models are: acoustic, articulatory, syntactic, gemma-2b
```
sbatch --job-name=enc --time=01:10:00 --gres=gpu:1 --partition=mig --ntasks=1 --cpus-per-task=1 code/slurm.sh -- code/encoding.py -m gemma-2b --suffix _shifted
```

# commands
```
rsync -av results zzada@scotty.pni.princeton.edu:/jukebox/hasson/zaid/narrative-enc/ 
```
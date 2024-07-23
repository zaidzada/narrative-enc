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
sbatch --job-name=emb --time=00:10:00 --mem=24G --gres=gpu:1 --constraint=gpu80 code/slurm.sh -- code/embeddings.py -m llama3-8b

# bigger model
sbatch --job-name=emb --time=00:10:00 --mem=128G --gres=gpu:2 --constraint=gpu80 code/slurm.sh -- code/embeddings.py -m gemma2-27b
```

## encoding
models are: acoustic, articulatory, syntactic, gemma-2b
```
sbatch --job-name=enc --time=01:02:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m gemma-2b 

# gemma2-9b, use layers 11, 22, 32
sbatch --job-name=enc --time=01:02:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m gemma2-9b -l 11

# llama3-8b; layers 16
sbatch --job-name=enc --time=01:02:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m llama3-8b -l 16

# gemma2-27b, use layers 23 35
sbatch --job-name=enc --time=01:02:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m gemma2-27b -l 23

# shfited
sbatch --job-name=enc --time=01:10:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m gemma-2b --suffix _shifted

```

# commands
```
rsync -av results zzada@scotty.pni.princeton.edu:/jukebox/hasson/zaid/narrative-enc/ 
```
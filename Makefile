.ONESHELL: # all the lines in the recipe be passed to a single invocation of the shell

# to reproduce, run these scripts in order:
# 1. `clean.py`
#     a. `groupavg.py` (not needed anymore)
# 2. `transcribe.sh`
#     a. `featuregen.py`
#     b. `embeddings.py`
#         i. `encoding.py`

make-env:
	conda create -n fconv
	conda activate fconv
	pip install accelerate himalaya nilearn scipy scikit-learn spacy tqdm \
			    transformers voxelwise_tutorials gensim pandas matplotlib \
				seaborn torch torchaudio torchvision surfplot neuromaps \
				jupyter tqdm nltk statsmodels h5py netneurotools openpyxl natsort

	# python code/embeddings.py -m gpt2-2b --layer 0

clean:
	python code/clean.py

transcribe:
	sbatch --job-name=transcribe --time=00:30:00 --mem=5G --gres=gpu:1 --partition=mig --ntasks=1 --cpus-per-task=1 code/slurm.sh \
	whisperx \
		--model large-v2 \
		--output_dir data/stimuli/whisperx \
		--output_format json \
		--task transcribe \
		--language en \
		--device cuda \
		data/stimuli/audio/*.wav

# WHISPER
whisper:
	sbatch --job-name=enc --mem=8G --time=00:15:00 --gres=gpu:1 code/slurm.sh -- \
		code/whisper.py -m whisper-medium

whisper_encoding:
	sbatch --job-name=enc --mem=8G --time=00:30:00 --gres=gpu:1 code/slurm.sh -- \
		code/whisper_encoding.py -m whisper-medium -l 20

# OLD:
joint_encoding:
	sbatch --job-name=jenc --time=01:02:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/joint_encoding.py

embeddings:
	sbatch --job-name=emb --time=00:10:00 --mem=24G --gres=gpu:1 --constraint=gpu80 code/slurm.sh -- code/embeddings.py -m llama3-8b
	sbatch --job-name=emb --time=00:10:00 --mem=128G --gres=gpu:2 --constraint=gpu80 code/slurm.sh -- code/embeddings.py -m gemma2-27b

encoding:
	sbatch --job-name=enc --time=01:02:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m gemma-2b 
	# gemma2-9b, use layers 11, 22, 32
	sbatch --job-name=enc --time=01:02:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m gemma2-9b -l 22
	# llama3-8b; layers 16
	sbatch --job-name=enc --time=01:02:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m llama3-8b -l 16
	# gemma2-27b, use layers 23 35
	sbatch --job-name=enc --time=01:02:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m gemma2-27b -l 23
	# shfited
	sbatch --job-name=enc --time=01:10:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py -m gemma-2b --suffix _shifted

backup:
	rsync -av results zzada@scotty.pni.princeton.edu:/jukebox/hasson/zaid/narrative-enc/ 

from_scotty:
	rsync -av --include="*/" --include="*task-black_desc-confounds_regressors.json" --exclude="*" data/derivatives/fmriprep della-vis2.princeton.edu:/scratch/gpfs/zzada/narrative-gradients/data/derivatives
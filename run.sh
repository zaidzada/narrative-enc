sbatch code/encoding.sh code/encoding.py -m model-gemma-2b_layer-16 -n black --folds 1
sbatch code/encoding.sh code/encoding.py -m model-gemma-2b_layer-16 -n black --folds 2 --suffix _2fold
sbatch code/encoding.sh code/encoding.py -m model-gemma-2b_layer-16 -n forgot --folds 2 --suffix _2fold
sbatch code/encoding.sh code/encoding.py -m model-gemma-2b_layer-16 -n black --folds 1 --group-sub --suffix _group
sbatch code/encoding.sh code/encoding.py -m model-gemma-2b_layer-16 -n black --folds 2 --group-sub --suffix _group_2fold
sbatch code/encoding.sh code/encoding.py -m model-gemma-2b_layer-16 -n forgot --folds 2 --group-sub --suffix _group_2fold
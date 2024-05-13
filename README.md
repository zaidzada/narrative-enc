# todo
- upload to github
- README

# choices we're making
- MRI confound regression -> acompcorr, cosines, 6 head mot (same as narratives)
- encoding model bands (LLM; word rate, word onset)
- k-fold split for stories (2-fold within; across-story) -> stick with across-story for now
- within-subject and group-averaged time series

- models (LLM; acoustic control?) -> google gemma, b/c 8192 context length (https://huggingface.co/google/gemma-2b)
- layer of the model -> half or 3/4 of total

# preprocessing
- zscore each fold separately before running

# save
- actual time series (534, 81k)
- predicted time series (534, 81k)
- embeddings with added metrics
- brain maps of encoding performance
- hdf5

# exploratory
- behavioral data
- other stories? slumlord / reach-for-the-skies
- contextual effects

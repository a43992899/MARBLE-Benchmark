python benchmark/tasks/MAESTRO/preprocess.py --mode audio --target_sr 24000 --dataset_dir data/MAESTRO/src/maestro-v2.0.0
python benchmark/tasks/MAESTRO/preprocess.py --mode label --target_sr 24000 --dataset_dir data/MAESTRO/src/maestro-v2.0.0 --frames_per_second 75 --resume


MODEL_PATH="/home/zengs/data/Code/reproduce/protgps/checkpoints/protgps/b4d112887e430dd1c97c826e04439745/b4d112887e430dd1c97c826e04439745epoch=23.ckpt"
ESM_DIR="~/zengs_data/torch_hub/checkpoints/"
DATA_FILENAME="/home/zengs/data/Code/reproduce/protgps/data/official/random_splits.data.xlsx"
OUTPUT_FILENAME="/home/zengs/data/Code/reproduce/protgps/test_runs/official_model_train_random_split_3/prediction.csv"

python inference.py --model_path $MODEL_PATH \
      --esm_dir $ESM_DIR \
      --device gpu \
      --input $DATA_FILENAME \
      --colname sequence \
      --output $OUTPUT_FILENAME
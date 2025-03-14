
MODEL_PATH="/home/zengs/data/Code/reproduce/protgps/checkpoints/protgps/225adf48176033377979a13197baea61/last.ckpt"
ESM_DIR="~/zengs_data/torch_hub/checkpoints/"
DATA_FILENAME="/home/zengs/data/Code/reproduce/protgps/data/dataset_from_json.xlsx"
OUTPUT_FILENAME="/home/zengs/data/Code/reproduce/protgps/test_runs/finetune-26epoch/prediction.csv"

python inference.py --model_path $MODEL_PATH \
      --esm_dir $ESM_DIR \
      --device gpu \
      --input $DATA_FILENAME \
      --colname sequence \
      --output $OUTPUT_FILENAME
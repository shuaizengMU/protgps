
MODEL_PATH="/home/zengs/data/Code/reproduce/protgps/checkpoints/protgps/e2ecc4152487050a625993048b8f0feb/e2ecc4152487050a625993048b8f0febepoch=20.ckpt"
ESM_DIR="~/zengs_data/torch_hub/checkpoints/"
DATA_FILENAME="/home/zengs/data/Code/reproduce/protgps/data/dataset_from_json.xlsx"
OUTPUT_FILENAME="/home/zengs/data/Code/reproduce/protgps/test_runs/20ckpt_prediction.csv"

python inference.py --model_path $MODEL_PATH \
      --esm_dir $ESM_DIR \
      --device gpu \
      --input $DATA_FILENAME \
      --colname sequence \
      --output $OUTPUT_FILENAME
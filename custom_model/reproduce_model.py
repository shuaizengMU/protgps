import torch
import torch.nn as nn
import numpy as np
from transformers import EsmModel, EsmTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, precision_score, recall_score

import pandas as pd
import pickle
import fire

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["WANDB_PROJECT"] = "ProtGPS"
os.environ["WANDB_LOG_MODEL"] = "False"


# Verify the selected GPU
print("Available GPU:", torch.cuda.device_count())
print("Using GPU:", torch.cuda.current_device())
print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))


LABEL_COLS = [
    'TRANSCRIPTIONAL', 'CHROMOSOME', 'NUCLEAR_PORE_COMPLEX',
    'NUCLEAR_SPECKLE', 'P-BODY', 'PML-BDOY', 'POST_SYNAPTIC_DENSITY',
    'STRESS_GRANULE', 'NUCLEOLUS', 'CAJAL_BODY', 'RNA_GRANULE', 'CELL_JUNCTION'
]


POOLING_MASK_MEAN = "POOLING_MASK_MEAN"
POOLING_ALL_MEAN = "POOLING_ALL_MEAN"
POOLING_CLS = "POOLING_CLS"


_POOLING_METHOD = POOLING_CLS
DATASET_FILENAME = "/home/zengs/data/Code/reproduce/protgps/data/dataset_from_json.csv"
OUTPUT_DIR = f"/home/zengs/data/Code/reproduce/protgps/test_runs/reproduce_model-{_POOLING_METHOD}"  



if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_dataset(tokenizer, dataset_filename):
  # Load CSV dataset
  data_df = pd.read_csv(dataset_filename)

  # Convert label columns to lists of integers (multi-hot encoding)
  data_df["labels"] = data_df[LABEL_COLS].values.tolist()

  # Tokenization function
  def tokenize_function(examples):
    return tokenizer(
        examples["sequence"], 
        padding="max_length", 
        truncation=True, 
        max_length=1800
    )

  # Convert to Hugging Face dataset
  dataset = Dataset.from_pandas(data_df)
  dataset = dataset.map(tokenize_function, batched=True, num_proc=16)

  # Ensure labels are in correct format
  def format_labels(example):
    example["labels"] = torch.tensor(example["labels"], dtype=torch.float)
    return example

  dataset = dataset.map(format_labels, num_proc=16)

  # Split dataset
  train_dataset = dataset.filter(lambda x: x["split"] == "train", num_proc=16)
  val_dataset = dataset.filter(lambda x: x["split"] == "dev", num_proc=16)
  test_dataset = dataset.filter(lambda x: x["split"] == "test", num_proc=16)

  return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred):
  preds, labels = eval_pred
  labels = np.array(labels)

  # Convert probs to binary using 0.5 threshold
  preds_binary = (preds > 0.5).astype(int)

  # Compute metrics
  if np.unique(labels).size == 1:
    acc = auc = f1 = mcc = precision = recall = np.nan
  else:
    acc = (preds_binary == labels).mean()
    auc = roc_auc_score(labels, preds, average="micro")
    f1 = f1_score(labels, preds_binary, average="micro")
    mcc = matthews_corrcoef(labels.flatten(), preds_binary.flatten())
    precision = precision_score(labels, preds_binary, average="micro", zero_division=0)
    recall = recall_score(labels, preds_binary, average="micro", zero_division=0)

  return {
      "ACC": acc,
      "AUCROC": auc,
      "F1": f1,
      "MCC": mcc,
      "Precision": precision,
      "Recall": recall,
  }
  
  
# Define model with an MLP classifier
class ESM2MLP(nn.Module):
  def __init__(self, model_name, num_classes=12, pooling=None):
    super().__init__()
    self.esm = EsmModel.from_pretrained(model_name)
    hidden_dim = self.esm.config.hidden_size  # ESM2-8M has 320 hidden dim
    self.classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, 512),
        nn.BatchNorm1d(512), 
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes)  # Output 12 logits
    )
    
    self.criterion = nn.BCEWithLogitsLoss()
    self.pooling = pooling

  def forward(self, input_ids, attention_mask=None, labels=None):
    outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)

    if self.pooling == POOLING_MASK_MEAN:
      attention_mask = attention_mask.unsqueeze(-1).float()
      pooled_output = (outputs.last_hidden_state * attention_mask).sum(axis=1) / attention_mask.sum(axis=1)
    elif self.pooling == POOLING_ALL_MEAN:
      pooled_output = outputs.last_hidden_state.mean(axis=1)
    elif self.pooling == POOLING_CLS:
      pooled_output = outputs.last_hidden_state[:, 0, :]
    else:
      raise ValueError(f"Invalid pooling method: {self.pooling}")
    
    logits = self.classifier(pooled_output)
    
    probs = torch.sigmoid(logits)
    if labels is not None:
      return self.compute_loss(logits, labels), probs

    return probs
  
  def compute_loss(self, logits, labels):
    # Use CrossEntropyLoss to compute the scalar loss
    loss = self.criterion(logits, labels.float())
    return loss


def train_model(pooling_method):
  
  model_name = "facebook/esm2_t6_8M_UR50D"
  tokenizer = EsmTokenizer.from_pretrained(model_name)
  train_dataset, val_dataset, test_dataset = get_dataset(tokenizer, DATASET_FILENAME)
  
  
  model = ESM2MLP(model_name, pooling=pooling_method)

  
  training_args = TrainingArguments(
      output_dir=OUTPUT_DIR,
      evaluation_strategy="epoch",
      save_strategy="epoch",
      per_device_train_batch_size=10,
      per_device_eval_batch_size=20,
      num_train_epochs=90,
      weight_decay=0.0,
      learning_rate=1e-3,
      lr_scheduler_type="cosine",
      warmup_steps=10,
      fp16=True,  # Use mixed precision
      logging_dir=os.path.join(OUTPUT_DIR, "logs"),
      logging_steps=200,
      save_total_limit=10,
      report_to="wandb",
      metric_for_best_model="AUCROC",
      greater_is_better=True,
      load_best_model_at_end=True,
      local_rank=-1
  )

  early_stopping = EarlyStoppingCallback(
      early_stopping_patience=5,  # Stop training if the evaluation metric doesn't improve for 2 evaluations
      early_stopping_threshold=0.0,  # The minimum change in the monitored metric to qualify as an improvement
  )

  # Define trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
      callbacks=[early_stopping]
  )

  # Train model
  trainer.train()
  
  for one_split_name, one_dataset in zip(
    ["train", "dev", "test"], [train_dataset, val_dataset, test_dataset]):
    one_results = trainer.predict(one_dataset)
    output_filename = os.path.join(OUTPUT_DIR, f"{one_split_name}_results.pkl")
    with open(output_filename, "wb") as f:
      pickle.dump(one_results, f)

if __name__ == "__main__":
  fire.Fire(train_model)
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import pickle
import time
import git
import torch
import pytorch_lightning as pl
from pytorch_lightning import _logger as log

from protgps.utils.parsing import parse_args
from protgps.utils.registry import get_object
import protgps.utils.loading as loaders
from protgps.utils.callbacks import set_callbacks

import random
import numpy as np


COMPARTMENTS = [
    "nuclear_speckle",
    "p-body",
    "pml-bdoy",
    "post_synaptic_density",
    "stress_granule",
    "chromosome",
    "nucleolus",
    "nuclear_pore_complex",
    "cajal_body",
    "rna_granule",
    "cell_junction",
    "transcriptional",
]


def set_seed(seed: int):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single-GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (multi-GPU)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Slows down but ensures reproducibility


def train(args):
    
    set_seed(43)
    
    # Remove callbacks from args for safe pickling later
    trainer = pl.Trainer.from_argparse_args(args)
    args.callbacks = None
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_devices
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank
    

    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object
    log.info(
        "\nProject main running by author: {} \ndate:{}, \nfrom commit: {} -- {}".format(
            commit.author,
            time.strftime("%m-%d-%Y %H:%M:%S", time.localtime(commit.committed_date)),
            commit.hexsha,
            commit.message,
        )
    )
    # print args
    for key, value in sorted(vars(args).items()):
        print("{} -- {}".format(key.upper(), value))

    # create or load lightning model from checkpoint
    model = loaders.get_lightning_model(args)

    # logger
    trainer.logger = get_object(args.logger_name, "logger")(args)

    # push to logger
    trainer.logger.setup(**{"args": args, "model": model})

    # add callbacks
    trainer.callbacks = set_callbacks(trainer, args)
    
    print(model)
    print(trainer.__dict__)
    print(args)



    def print_params(model):
        # Get all parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Get trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate the percentage of trainable parameters
        trainable_percentage = (trainable_params / total_params) * 100

        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Percentage of Trainable Parameters: {trainable_percentage:.2f}%")

    # Example: print parameters for your model
    print_params(model)

    # train model
    if args.train:
        train_dataset = loaders.get_train_dataset_loader(args)
        dev_dataset = loaders.get_eval_dataset_loader(args, split="dev")
        
        ##########
        seqs = [] 
        label = []
        ids = []
        for one in train_dataset:
            seqs.append(one['x'])
            label.append(one['y'])
            ids.append(one['entry_id'])
        
        import numpy as np
        import pandas as pd
        seqs = np.concatenate(seqs, axis=0)
        label = np.concatenate(label, axis=0)
        ids = np.concatenate(ids, axis=0)
        
        data_df = pd.DataFrame({
            'id': ids,
        })
        for j, condensate in enumerate(COMPARTMENTS):
            data_df[f"{condensate.upper()}"] = label[:, j].tolist()
        data_df["sequence"] = seqs.tolist()
        
        output_dir = "/home/zengs/data/Code/reproduce/protgps/data/official_dataloader/mmseqs_train.csv"
        data_df.to_csv(output_dir, index=False)

        print(ids[0])
        print(label[0])
        print(seqs[0])
        
        # exit(0)
        ####
        
        log.info("\nTraining Phase...")
        trainer.fit(model, train_dataset, dev_dataset)
        if trainer.checkpoint_callback:
            args.model_path = trainer.checkpoint_callback.best_model_path

    # save args
    if args.local_rank == 0:
        print("Saving args to {}.args".format(args.results_path))
        pickle.dump(vars(args), open("{}.args".format(args.results_path), "wb"))

    return model, trainer.logger


def eval(model, logger, args):
    # reinit trainer
    trainer = pl.Trainer(gpus=1)

    # reset ddp
    args.strategy = None

    # connect to same logger as in training
    trainer.logger = logger

    # set callbacks
    trainer.callbacks = set_callbacks(trainer, args)

    # eval on train
    if args.eval_on_train:
        log.info("\nInference Phase on train set...")
        train_dataset = loaders.get_eval_dataset_loader(args, split="train")

        if args.train and trainer.checkpoint_callback:
            trainer.test(model, train_dataset, ckpt_path=args.model_path)
        else:
            trainer.test(model, train_dataset)

    # eval on dev
    if args.dev:
        log.info("\nValidation Phase...")
        dev_dataset = loaders.get_eval_dataset_loader(args, split="dev")
        if args.train and trainer.checkpoint_callback:
            trainer.test(model, dev_dataset, ckpt_path=args.model_path)
        else:
            trainer.test(model, dev_dataset)

    # eval on test
    if args.test:
        log.info("\nInference Phase on test set...")
        test_dataset = loaders.get_eval_dataset_loader(args, split="test")

        if args.train and trainer.checkpoint_callback:
            trainer.test(model, test_dataset, ckpt_path=args.model_path)
        else:
            trainer.test(model, test_dataset)


if __name__ == "__main__":
    args = parse_args()
    model, logger = train(args)

    if args.dev or args.test or args.eval_on_train:
        if args.strategy == "ddp":
            torch.distributed.destroy_process_group()
            log.info("\n\n")
            log.info(">" * 33)
            log.info("Destroyed process groups for eval")
            log.info("<" * 33)
            log.info("\n\n")

        if args.global_rank == 0:
            eval(model, logger, args)

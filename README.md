## Setup

1. Install mamba (recommended) or conda

- download installer from: https://github.com/conda-forge/miniforge
- install and init

```
bash Miniforge-pypy3-Linux-x86_64.sh
```

2. Create environment 

```
mamba env create -f environment.yml
```
 
3. Activate


```
mamba activate protgps
```

4. Download model checkpoints

5. To train model:
    
```
python scripts/dispatcher.py --config configs/protein_localization/full_prot_comp_pred.json --log_dir /path/to/logdir
```

6. To generate proteins:

```
cd esm/examples/lm-design
./generate_nucleolus.sh
./generate_nuclear_speckle.sh
```
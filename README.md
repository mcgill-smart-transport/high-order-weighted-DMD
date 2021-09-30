# High-order weighted dynamic mode decomposition (HW-DMD)
Code for paper "Real-time forecasting of metro origin-destination matrices with high-order weighted dynamic mode decomposition", https://arxiv.org/abs/2101.00466

## How to use
- The key module of HW-DMD is the `HWDMD` class located in `Experiments/functions.py`.
- Experiments of the paper are shown in the `.ipybb` files in the `Experiments/` directory.
  - The Hangzhou metro dataset (from https://zenodo.org/record/3145404#.YVUaHTHMKPo) is included in the `data/` directory. Experiments on the Hangzhou dataset are runnable from `.ipynb` files.
- The Guangzhou metro dataset is not included. Files for experiments are included for illustration purposes.

## Environment for reproducible results
- Python 3.8.8
- Numpy 1.20.1
- Scipy 1.6.1

For benchmark models that use tensorflow
- Tensorflow 2.6.0
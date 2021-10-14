# High-order weighted dynamic mode decomposition (HW-DMD)
Code for paper "Real-time forecasting of metro origin-destination matrices with high-order weighted dynamic mode decomposition", https://arxiv.org/abs/2101.00466

HW-DMD is a simple yet powerful model for forecasting in high-dimensional time series. The repository includes [a small example](/Example/Seattle_traffic_forecasting.ipynb) to facilitate the rapid adoption of HW-DMD to general forecast problems.

## How to use
The best way to use HW-DMD in your forecast problem is to modify from the [example notebook](/Example/Seattle_traffic_forecasting.ipynb), where we used HW-DMD to forecast the [traffic speed at 323 loop detectors in Seattle](https://github.com/zhiyongc/Seattle-Loop-Data). Download the repository and replace the Seattle data with your data, then play with the code.

The `Experiments/` folder contains the code used in the HW-DMD paper. Some explanations for the organization:
- Experiments of the paper are shown in the `.ipynb` files in the `Experiments/` folder.
- The key module of HW-DMD is the `HWDMD` class located in `Experiments/functions.py`.
- The Hangzhou metro dataset (from https://zenodo.org/record/3145404#.YVUaHTHMKPo) is included in the `data/` directory. Experiments on the Hangzhou dataset are runnable from `.ipynb` files.
- The Guangzhou metro dataset is not included. Files for experiments are included for illustration purposes.

## Environment for reproducible results
- Python 3.8.8
- Numpy 1.20.1
- Scipy 1.6.1

For benchmark models that use tensorflow
- Tensorflow 2.6.0

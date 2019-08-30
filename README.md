# Computationally Efficient Feature Significance and Importance for Machine Learning Models (SFIT)

## Introduction

This repo implements the SFIT method from the
[paper](https://arxiv.org/pdf/1905.09849.pdf) "Computationally Efficient Feature Significance and Importance for
Machine Learning Models".

The Single Feature Introduction Test (SFIT) is a simple and computationally efficient significance test for the features
of a machine learning model. Our forward-selection approach applies to any model specification, learning task and
variable type. The test is non-asymptotic, straightforward to implement, and does not require model refitting.
It identifies the statistically significant features as well as feature interactions of any order in
a hierarchical manner.
For more details, please refer to the
[full paper](https://arxiv.org/pdf/1905.09849.pdf).

## Requirements

The sfit functions require
`numpy`, `scipy`, `statsmodels`, `keras` and `sklearn`.

The main file that illustrates use cases of the code and replicate the results of the simulations from the
[paper](https://arxiv.org/pdf/1905.09849.pdf) require
`sklearn`, `statsmodels`, `tensorflow` and `keras`.


## Running the code

`python main.py`

This generates simulated data as described in the [paper](https://arxiv.org/pdf/1905.09849.pdf) and fit a linear
model and a neural network on them. These models are then used to run the SFIT method. The expected printed output has been
saved in [this file](./expected_output.txt).

## Contact and cite

If you have any questions, please contact Enguerrand Horel (ehorel at stanford dot edu).

If you use this code in your work, please cite:

@article{horel2019computationally,
  title={Computationally Efficient Feature Significance and Importance for Machine Learning Models},
  author={Horel, Enguerrand and Giesecke, Kay},
  journal={arXiv preprint arXiv:1905.09849},
  year={2019}
}

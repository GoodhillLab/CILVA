# Calcium imaging latent variable analysis (CILVA)
![](https://img.shields.io/github/license/GoodhillLab/CILVA.svg)

Python code for fitting the calcium imaging latent variable analysis model. See the companion preprint at https://www.biorxiv.org/content/10.1101/691261v1 for more information.

## Overview

This code fits a latent variable model to a matrix of fluorescence traces. The fitted model provides
- a description of low dimensional latent structure underlying spontaneous activity,
- a decomposition of the fluorescence traces into their evoked and (low dimensional) spontaneous components,
- estimated stimulus filters for each neuron that are unbiased by the modelled spontaneous activity.

## Usage
### Basic model fitting
Fit the model by calling `run.py` with the required arguments: `python cilva/run.py --data $data --L $L --num_iters $num_iters --iters_per_altern $iters_per_altern --max_threads $max_threads --out $out --tau_r $tau_r --tau_d $tau_d --imrate $imrate` where
- `$data` is the path to the training data. The code expects an NxT matrix of fluorescence levels $data.ca2 and a KxT matrix of stimulus onset times $data.stim. These can simply be renamed text files.
- `$L` is the number of latent factors.
- `$num_iters` is the number of times we alternate between estimating the latent variables and the model parameters.
- `$iters_per_altern` is the number of optimiser steps for calculating the MAP estimate of the latent variables.
- `$max_threads` configures the number of threads available for multithreaded processing.
- `$out` is the prefix of the output folder containing the fitted model parameters.
- `$tau_r` and `$tau_d` are the rise and decay time constants for the calcium transients.
- `$imrate` is the imaging rate (or sampling frequency) of the fluorescence microscope. This is used to estmate the imaging noise variance.
More several additional arguments are provided in the `run.py` file.
### Example
The Jupyter notebook `example.ipynb` provides an example of how to fit the model to data and decouple the fluorescence traces (recommended). It can be viewed here:
> https://nbviewer.jupyter.org/github/GoodhillLab/CILVA/blob/master/example.ipynb
### Cross validation
The code also supports a form of single-fold cross validation: the model parameters and latent variables are learned from the training data specified with the `--data` argument, we then reinfer the latent variables on held-out test data using the already-estimated model parameters. Test data is supplied with the `--test` argument.
### Output
When CILVA has finished running, the estimated parameters are written to the directory
>./$out\_L\_$L\_num\_iters\_${num_iters}\_iters\_per\_altern\_${iters_per_altern}\_gamma\_$gamma\_tau\_r\_${tau_r}\_tau\_d\_${tau_d}\_imrate\_${imrate}

where each $arg corresponds to an argument from the `Basic model fitting` section above.

## Requirements
This code requires Python 3.5+ and the NumPy (tested on v1.16.2) and SciPy (tested on v1.1.0) packages.

## Contact
Feedback or questions about the code can be directed to [marcus.triplett@columbia.edu](marcus.triplett@columbia.edu).

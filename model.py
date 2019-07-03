
'''

CILVA model fitting and parameter identification.

Author: Marcus A. Triplett. (2019). University of Queensland, Australia.


'''

import numpy as np
import scipy as sp
import time
import core
import os

def train(data, convert_stim, L, num_iters, iters_per_altern, gamma, tau_r, tau_d, imrate):
	'''

		Train the calcium-imaging latent variable analysis model.

	'''

	# Load training data
	f, s 	= core.load_data(data, convert_stim)
	N, T 	= f.shape
	K 		= s.shape[0]

	# Estimate imaging-noise variance
	sigma = core.estimate_noise_sdevs(f, N, T, imrate)

	# Initialise params
	kernel 		= core.calcium_kernel(tau_r, tau_d, T)
	init_alpha 	= np.random.normal(1, 1e-2, N)
	init_beta 	= np.zeros((N, ))
	init_w 		= core.init_filters(f, s, kernel, N, T, K)
	init_b 		= np.random.rand(N, L)
	init_x 		= 1e-1 * np.random.rand(L, T)

	initial_params = [init_alpha, init_beta, init_w, init_b, init_x]
	args = [f, s, kernel, N, T, K, L, gamma, sigma]

	# Set non-negative parameter bounds for L-BFGS-B optimisation
	eps = 1e-8
	static_bounds = [(eps, None)] * N # alpha
	static_bounds += [(0, None)] * (N + N * K + N * L) # beta, w, b
	latent_bounds = [(0, None)] * (L * T) # x

	# Alternating optimisation of static parameters and latent variables
	t_start = time.time()
	alpha, beta, w, b, x = core.alternating_minimisation(
		initial_params, args, static_bounds, latent_bounds, num_iters, iters_per_altern)
	t_end = time.time()
	print('Total elapsed time: %.2fs (%im).'%(t_end - t_start, (t_end - t_start)//60))

	# Parameter identification
	param_identification_args = [N, L, s]
	alpha_hat, beta_hat, w_hat, b_hat, x_hat, sigma_hat = core.identify_params(alpha, beta, w, b, x, sigma,
		param_identification_args)

	return [alpha_hat, beta_hat, w_hat, b_hat, x_hat, sigma_hat]

def cvd(data, L, convert_stim, params, num_iters, gamma, tau_r, tau_d):
	'''

		Estimate latent factor activity on held-out test data, keeping static params fixed.

	'''

	f, s = core.load_data(data, convert_stim)
	N, T = f.shape
	K = s.shape[0]

	alpha, beta, w, b, sigma = params

	kernel = core.calcium_kernel(tau_r, tau_d, T)
	init_x = np.random.rand(L, T)

	latent_bounds = [(0, None)] * (L * T)
	args = [f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b]

	t_start = time.time()
	x_hat = core.estimate_latents(init_x, args, latent_bounds, num_iters)
	t_end = time.time()
	print('Total elapsed time: %.2fs (%im).'%(t_end - t_start, (t_end - t_start)//60))

	return x_hat


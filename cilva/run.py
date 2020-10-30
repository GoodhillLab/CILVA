
''' 

Fits the calcium imaging latent variable analysis (CILVA) model to population calcium imaging data. 
This code alternately optimises for the static model parameters and MAP estimates of the latent variables.

Arguments:
	data:
		Location of experimental data. Expects an NxT matrix of fluorescence levels matching ${data}.ca2 and 
		a Tx1 vector of stimulus onset times matching ${data}.stim.

	L:
		Number of latent factors (i.e., dimensionality of latent state).

	num_iters:
		Number of iterations of the alternating minimisation MAP estimator.

	iters_per_altern:
		Number of iterations of the gradient-based optimisation within each alternation of the MAP estimator.

	max_threads:
		Number of threads for multithreaded computing. Often must be specified for cluster computing.

	out:
		Prefix for output data folder. Output has form 
		'./${out}_L_${L}_gamma_${gamma}_num_iters_${num_iters}_iters_per_altern_${iters_per_altern}'. 
		Typically used in conjunction with job IDs for cluster computing.

	gamma:
		Rate parameter for exponential prior on latent activity states. Acts as a sparsity penalty.

	tau_r:
		Calcium transient rise time constant.

	tau_d:
		Calcium transient decay time constant.

	test:
		Location of test data. Expects a matrix of fluorescence levels and stimulus onset times similar to --data.

	imrate:
		Imaging rate of fluorescence microscope. Required for estimating imaging noise variance.

	convert_stim:
		Convert from a 1 dimensional representation of the stimulus to a 2 dimensional (1-of-K) representation. (Boolean)


Author: Marcus A. Triplett. (2019). University of Queensland, Australia.

See the GitHub page https://github.com/GoodhillLab/CILVA for more information.

Tested on:
	Python v3.6.4
	NumPy v1.16.2
	SciPy v1.1.0

'''

import argparse
import os

if __name__ == '__main__':
	# Parse args
	parser = argparse.ArgumentParser()
	parser.add_argument('--data')
	parser.add_argument('--L')
	parser.add_argument('--num_iters')
	parser.add_argument('--iters_per_altern')
	parser.add_argument('--max_threads')
	parser.add_argument('--out')
	parser.add_argument('--gamma')
	parser.add_argument('--tau_r')
	parser.add_argument('--tau_d')
	parser.add_argument('--test')
	parser.add_argument('--imrate')
	parser.add_argument('--convert_stim', action='store_true')
	args = parser.parse_args()

	if args.imrate:
		imrate = float(args.imrate)
	else:
		raise Exception('Imaging frequency must be supplied via the --imrate argument.')

	# Default parameter values
	default_L = 3
	default_num_iters = 40
	default_iters_per_altern = 40
	default_max_threads = '2'
	default_gamma = 1.00
	default_tau_r = 5.681 / imrate # Note: default rise and decay time constants are appropriate for our GCaMP6s zebrafish larvae. They may not be suitable for other indicators or animal models.
	default_tau_d = 11.551 / imrate

	data = args.data

	# Set defaults
	if args.L:
		L = int(args.L)
	else:
		L = default_L
		print('No latent dimensionality specified. Defaulting to L = %i.'%L)

	if args.num_iters:
		num_iters = int(args.num_iters)
	else:
		num_iters = default_num_iters
		print('No number of iterations specified for the alternating MAP estimator. Defaulting to num_iters = %i.'%num_iters)

	if args.iters_per_altern:
		iters_per_altern = int(args.iters_per_altern)
	else:
		iters_per_altern = default_iters_per_altern
		print('No number of iterations per alternation of the MAP estimator specified. Defaulting to iters_per_altern = %i.'%iters_per_altern)

	if args.max_threads:
		max_threads = args.max_threads
	else:
		max_threads = default_max_threads
		print('Number of threads unspecified for multithreading. Defaulting to max_threads = %s.'%max_threads)

	if args.out:
		out = args.out
	else:
		out = ''

	if args.gamma:
		gamma = float(args.gamma)
	else:
		gamma = default_gamma
		print('No prior mean specified for latent factor activity states. Defaulting to gamma = %.3f.'%gamma)

	if args.tau_r:
		tau_r = float(args.tau_r)
	else:
		tau_r = default_tau_r
		print('No calcium rise time specified. Defaulting to tau_r = %.3f.'%tau_r)

	if args.tau_d:
		tau_d = float(args.tau_d)
	else:
		tau_d = default_tau_d
		print('No calcium decay time specified. Defaulting to tau_d = %.3f.'%tau_d)

	if args.convert_stim:
		convert_stim = True
	else:
		convert_stim = False

	# Configure threading
	os.environ["MKL_NUM_THREADS"] = max_threads
	os.environ["NUMEXPR_NUM_THREADS"] = max_threads
	os.environ["OMP_NUM_THREADS"] = max_threads

	import numpy as np # Importing numpy and model must be performed *after* multithreading is configured.
	import model

	# Train model
	print('Fitting the Calcium Imaging Latent Variable Analysis model...')
	learned_params = model.train(data, convert_stim, L, num_iters, iters_per_altern, gamma, tau_r, tau_d, imrate)
	learned_params += [[tau_r], [tau_d], [gamma], [L]]
	param_names = ['alpha', 'beta', 'w', 'b', 'x', 'sigma', 'tau_r', 'tau_d', 'gamma', 'L']

	# Save learned params to file
	print('Saving parameters to file...')
	path = './' + out + '_L_{}_num_iters_{}_iters_per_altern_{}_gamma_{:.2f}_tau_r_{:.2f}_tau_d_{:.2f}_imrate_{:.4f}/'.format(
		L, num_iters, iters_per_altern, gamma, tau_r, tau_d, imrate)
	if not os.path.isdir(path): os.mkdir(path)
	for i, param_name in enumerate(param_names):
		np.savetxt(path + param_name, learned_params[i], fmt='%1.6e')

	# Estimate factor activity on held-out test data
	if args.test:
		cvd_param_names = ['alpha', 'beta', 'w', 'b', 'sigma']
		cvd_params = []
		for param_name in cvd_param_names:
			# Collect parameters from learned_params
			cvd_params += [learned_params[param_names.index(param_name)]]

		x_test = model.cvd(args.test, L, convert_stim, cvd_params, num_iters * iters_per_altern, gamma, tau_r, tau_d)
		np.savetxt(path + 'x_test', x_test, fmt='%1.6e')
	print('Model-fitting complete.')

'''

Model-selection tools for fitting CILVA.

Author: Marcus A. Triplett. (2019). University of Queensland, Australia.

'''

import os
import core
import numpy as np

def select_model(folder, data, data_type, return_fit = 'both', convert_stim = False):
	'''
	
		Selects the model-fit with the highest posterior probability among a folder of fits.

		Arguments: 
			folder:
				Directory containing a set of model fits.

			data:
				Path to the associated data.

			data_type:
				Set to 'train' if input data is training data, 'test' for test data.

			return_type:
				Set to 'train' to return the fits for training data, 'test' to return the fits for test data,
				or 'both' to return both fits.

			convert_stim:
				Set to True if stimulus must be converted from a 1d to 2d representation.


	'''

	model_fits = [f for f in os.listdir(folder) if not f.startswith('.')]
	llhs = [None] * len(model_fits)
	f, s = core.load_data(data, convert_stim)
	N, T = f.shape
	for findx, fit in enumerate(model_fits):
		alpha, beta, w, b, x, sigma, tau_r, tau_d, gamma, L = load_fit(folder + '/' + fit, data_type)
		if L == 1:
			x = np.reshape(x, [int(L), T])
			b = np.reshape(b, [N, int(L)])
		kernel = core.calcium_kernel(tau_r, tau_d, T)
		K = s.shape[0]
		llhs[findx] = -core.log_joint(f, s, kernel, N, T, K, L, gamma, sigma, alpha, beta, w, b, x)

	return load_fit(folder + '/' + model_fits[np.argmax(llhs)], return_fit)


def load_fit(file, data_type):
	'''
		
		Load fitted model parameters.

	'''


	alpha 	= np.loadtxt(file + '/alpha')
	beta 	= np.loadtxt(file + '/beta')
	w 		= np.loadtxt(file + '/w')
	b 		= np.loadtxt(file + '/b')
	sigma 	= np.loadtxt(file + '/sigma')
	tau_r 	= np.loadtxt(file + '/tau_r')
	tau_d 	= np.loadtxt(file + '/tau_d')
	gamma 	= np.loadtxt(file + '/gamma')
	L 		= np.loadtxt(file + '/L')

	if data_type == 'train':
		x = np.loadtxt(file + '/x')
		return [alpha, beta, w, b, x, sigma, tau_r, tau_d, gamma, L]
	elif data_type == 'test':
		x = np.loadtxt(file + '/x_test')
		return [alpha, beta, w, b, x, sigma, tau_r, tau_d, gamma, L]
	elif data_type == 'both':
		x = np.loadtxt(file + '/x')
		x_test = np.loadtxt(file + '/x_test')
		return [alpha, beta, w, b, x, sigma, tau_r, tau_d, gamma, L, x_test]
	else:
		raise Exception('Input string data_type must be either \'train\', \'test\', or \'both\'.')

	
def reconstruction(alpha, beta, w, b, x, kernel, s):
	'''

		Reconstruct calcium imaging data using the learned model parameters.

	'''


	N = alpha.shape[0]	
	T = s.shape[1]
	f_hat = np.zeros((N, T))

	if len(x.shape) == 1:
		x = np.reshape(x, [1, T])
		b = np.reshape(b, [N, 1])

	lambda_ = w @ s + b @ x

	for n in range(N):
		f_hat[n] = alpha[n] * np.convolve(kernel, lambda_[n])[:T] + beta[n]

	return f_hat

def decouple_traces(alpha, beta, w, b, x, kernel, s):
	'''
	
		Compute evoked and spontaneous components of the fluorescence levels using the 
		learned model parameters.

	'''


	N = alpha.shape[0]	
	T = s.shape[1]
	f_evoked = np.zeros((N, T))
	f_spont = np.zeros((N, T))	

	if len(x.shape) == 1:
		x = np.reshape(x, [1, T])
		b = np.reshape(b, [N, 1])

	stim = w @ s 
	spont = b @ x

	for n in range(N):
		f_evoked[n] = alpha[n] * np.convolve(kernel, stim[n])[:T] + beta[n]
		f_spont[n] = alpha[n] * np.convolve(kernel, spont[n])[:T] + beta[n]

	return [f_evoked, f_spont]

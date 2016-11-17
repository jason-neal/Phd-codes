#!/usr/bin/env python
# Test alpha variation at which cannot detect a planet

# Create a combined spectra with a planet at an alpha value.
# try and detect it by varying RV and alpha.
# At some stage the alpha will not vary when it becomes to small
# This will be the alpha detection limit.

# Maybe this is a wavelength dependant?

# The goal is to get something working and then try improve the performance
# for complete simulations.

# Create the test spectra.
from __future__ import division, print_function

from IP_multi_Convolution import IPconvolution
import numpy as np
import multiprocess as mprocess
from tqdm import tqdm
import scipy.stats
# from scipy.stats import chisquare
from Planet_spectral_simulations import combine_spectra
from Planet_spectral_simulations import load_PHOENIX_hd30501
import itertools
from collections import defaultdict
from datetime import datetime as dt
import time
import pickle
# import matplotlib.pyplot as plt
# from astropy.io import fits
from spectrum_overload.Spectrum import Spectrum
import copy
from numba import jit
from joblib import Memory
from joblib import Parallel, delayed
import os
import sys
sys.path.append("/home/jneal/Phd/Codes/UsefulModules/Convolution")


path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"  # save path
cachedir = os.path.join(path, "cache")  # save path
memory = Memory(cachedir=cachedir, verbose=0)

@jit
def chi_squared(observed, expected, error=None):
    """Calculate chi squared.
    Same result as as scipy.stats.chisquare
    """
    if np.any(error):
        chisqr = np.sum((observed-expected)**2 / error**2)
    else:
        # chisqr = np.sum((observed-expected)**2)
        chisqr = np.sum((observed-expected)**2 / expected)
        # When divided by exted the result is identical to scipy
    return chisqr


@jit
def alternate_chi_squared(observed, expected, error=None):
    """Calculate chi squared.
    Same result as as scipy.stats.chisquare
    """
    if error:
        chisqr = np.sum((observed-expected)**2 / observed)
    else:
        # chisqr = np.sum((observed-expected)**2)
        chisqr = np.sum((observed-expected)**2 / expected)
        # When divided by exted the result is identical to scipy
    return chisqr


@jit
def add_noise(flux, SNR):
    "Using the formulation mu/sigma"
    sigma = flux / SNR
    # Add normal distributed noise at the SNR level.
    noisey_flux = flux + np.random.normal(0, sigma)
    return noisey_flux


@memory.cache
def apply_convolution(model_spectrum, R=None, chip_limits=None):
    """ Apply convolution to spectrum object"""
    if chip_limits is None:
        chip_limits = (np.min(model_spectrum.xaxis), np.max(model_spectrum.xaxis))

    if R is None:
        return copy.copy(model_spectrum)
    else:
        ip_xaxis, ip_flux = IPconvolution(model_spectrum.xaxis[:],
                                          model_spectrum.flux[:], chip_limits,
                                          R, FWHM_lim=5.0, plot=False,
                                          verbose=True)

        new_model = Spectrum(xaxis=ip_xaxis, flux=ip_flux,
                             calibrated=model_spectrum.calibrated,
                             header=model_spectrum.header)

        return new_model

@memory.cache
def store_convolutions(spectrum, resolutions, chip_limits=None):
    """ Convolve spectrum to many resolutions and store in a dict to retreive.
     This prevents multiple convolution at the same resolution.
    """
    d = dict()
    for resolution in resolutions:
        d[resolution] = apply_convolution(spectrum, resolution, chip_limits=chip_limits)

    return d

@memory.cache
def generate_observations(model_1, model_2, rv, alpha, resolutions, snrs):
    """ Create an simulated observation for combinations of resolution and snr.

    Paramters:
    model_1: and model_2 are Spectrum objects.
    rv: the rv offset applied to model_2
    alpha: flux ratio I(model_2)/I(model_1)
    resolutions: list of resolutions to simulate
    snrs: list of snr values to simulate

    Returns:
    observations: dict[resolution][snr] containing a simulated spectrum.

    """
    observations = defaultdict(dict)
    iterator = itertools.product(resolutions, snrs)
    for resolution, snr in iterator:
        # Preform tasks to simulate an observation
        spec_1 = model_1[resolution]

        spec_2 = model_2[resolution]
        spec_2.doppler_shift(rv)
        # model1 and model2 are already normalized and convovled to each resolution using
        # store_convolutions
        combined_model = combine_spectra(spec_1, spec_2, alpha)

        combined_model.flux = add_noise(combined_model.flux, snr)

        observations[resolution][snr] = combined_model

    return observations


def parallel_chisquared(i, j, alpha, rv, res, snr, observation, host_models, companion_models, output1, output2):
    """ Function for parallel processing in chisqr for resolution and snr
    uses numpy memap to store data.

    Inputs:
    i,j = matrix locations of this alpha and rv
    alpha =  flux ratio
    rv = radial velocity offset of companion
    snr = signal to noise ratio of observation

    Output:
    None: Changes data in output1 and output2 numpy memmaps.
    """
    # print("i, j, Resolution, snr, alpha, RV")
    # print(i, j, res, snr, alpha, rv)
    host_model = host_models[res]
    companion_model = companion_models[res]
    companion_model.doppler_shift(rv)
    combined_model = combine_spectra(host_model, companion_model, alpha)
    # model_new = combine_spectra(convolved_star_models[resolution], convolved_planet_models[resolution].doppler_shift(RV), alpha)

    combined_model.wav_select(2100, 2200)
    observation.wav_select(2100, 2200)

    # print("i", i, "j", j, "chisqr", scipy.stats.chisquare(observation.flux, combined_model.flux).statistic)

    output1[i, j] = scipy.stats.chisquare(observation.flux, combined_model.flux).statistic
    output2[i, j] = chi_squared(observation.flux, combined_model.flux, error=observation.flux/snr)


def wrapper_parallel_chisquare(args):
    """ Wrapper for parallel_chisquare needed to unpack the arguments for
    parallel_chisquare as multiprocess.Pool.map does not accept multiple
    arguments
    """
    return parallel_chisquared(*args)


# @jit
def main():
    """ Chisquare determinination to detect minimum alpha value"""
    print("Loading Data")

    path = "/home/jneal/Phd/Codes/Phd-codes/Simulations/saves"  # save path

    chip_limits = [2080, 2220]

    (w_mod, I_star, I_bdmod,
        hdr_star, hdr_bd) = load_PHOENIX_hd30501(limits=chip_limits,
                                                 normalize=True)

    org_star_spec = Spectrum(xaxis=w_mod, flux=I_star, calibrated=True)
    org_bd_spec = Spectrum(xaxis=w_mod, flux=I_bdmod, calibrated=True)

    Resolutions = [None, 50000]
    snrs = [100, 101, 110, 111]   # Signal to noise levels
    alphas = 10**np.linspace(-5, -0.2, 200)
    RVs = np.arange(10, 30, 0.1)
    # Resolutions = [None, 1000, 10000, 50000, 100000, 150000, 200000]
    # snrs = [50, 100, 200, 500, 1000]   # Signal to noise levels
    # alphas = 10**np.linspace(-4, -0.1, 200)
    # RVs = np.arange(-100, 100, 0.05)

    # RV and alpha value of Simulations
    RV_val = 20
    Alpha = 0.1  # Vary this to determine detection limit
    input_parameters = (RV_val, Alpha)

    convolved_star_model = store_convolutions(org_star_spec, Resolutions, chip_limits=chip_limits)
    convolved_planet_model = store_convolutions(org_bd_spec, Resolutions, chip_limits=chip_limits)

    # print(type(convolved_star_model))
    # print(type(convolved_planet_model))
    simulated_obersvations = generate_observations(convolved_star_model,
                                                   convolved_planet_model,
                                                   RV_val, Alpha,
                                                   Resolutions, snrs)

    # Not used with gernerator function
    goal_planet_shifted = copy.copy(org_bd_spec)
    # RV shift BD spectra
    goal_planet_shifted.doppler_shift(RV_val)

    # These should be replaced by
    res_stored_chisquared = dict()
    res_error_stored_chisquared = dict()
    # This
    res_snr_storage_dict = defaultdict(dict)  # Dictionary of dictionaries
    error_res_snr_storage_dict = defaultdict(dict)  # Dictionary of dictionaries
    # Iterable over resolution and snr to process
    # res_snr_iter = itertools.product(Resolutions, snrs)
    # Can then store to dict store_dict[res][snr]

    print("Starting loops")

    # multiprocessing part
    numProcs = None
    if numProcs is None:
        numProcs = mprocess.cpu_count() - 1

    # mprocPool = mprocess.Pool(processes=numProcs)
    timeInit = dt.now()

    for resolution in tqdm(Resolutions):
        chisqr_snr_dict = dict()  # store 2d array in dict of SNR
        error_chisqr_snr_dict = dict()
        print("\nSTARTING run of RESOLUTION={}\n".format(resolution))
        # chisqr_snr_dict = dict()  # store 2d array in dict of SNR
        # error_chisqr_snr_dict = dict()

        for snr in snrs:
            sim_observation = simulated_observations[resolution][snr]

            # Multiprocessing part
            scipy_filename = os.path.join(path, "scipychisqr.memmap")
            my_filename = os.path.join(path, "mychisqr.memmap")
            scipy_memmap = np.memmap(scipy_filename, dtype='float32', mode='w+', shape=X.shape)
            my_chisqr_memmap = np.memmap(my_filename, dtype='float32', mode='w+', shape=X.shape)

            # args_generator = tqdm([[i, j, alpha, rv, resolution, snr, sim_observation, convolved_star_models, convolved_planet_models, scipy_memmap, my_chisqr_memmap]
            #                      for i, alpha in enumerate(alphas) for j, rv in enumerate(RVs)])
        # if resolution is None:
        #    star_spec = copy.copy(org_star_spec)
        #    goal_planet = copy.copy(goal_planet_shifted)
        # else:
        #    ip_xaxis, ip_flux = IPconvolution(org_star_spec.xaxis,
    #             org_star_spec.flux, chip_limits, resolution,
    #            FWHM_lim=5.0, plot=False, verbose=True)

    #        star_spec = Spectrum(xaxis=ip_xaxis, flux=ip_flux,
        #                                 calibrated=True,
        #                                 header=org_star_spec.header)

    #        ip_xaxis, ip_flux = IPconvolution(goal_planet_shifted.xaxis,
    #            goal_planet_shifted.flux, chip_limits, resolution,
    #            FWHM_lim=5.0, plot=False, verbose=False)

    #        goal_planet = Spectrum(xaxis=ip_xaxis, flux=ip_flux,
    #                                     calibrated=True,
    #                                     header=goal_planet_shifted.header)

        print("Starting SNR loop for resolution value of {}".format(resolution))
        for snr in snrs:
            loop_start = time.time()
            print("Calculation with snr level", snr)
            # This is the signal to try and recover
            Alpha_Combine = combine_spectra(star_spec, goal_planet, Alpha)
            Alpha_Combine.wav_select(2100, 2200)
            Alpha_Combine.flux = add_noise2(Alpha_Combine.flux, snr)

            # Test plot
            # plt.plot(Alpha_Combine.xaxis, Alpha_Combine.flux)
            sim_observation = simulated_obersvations[resolution][snr]
            # plt.plot(this_simulation.xaxis, this_simulation.flux, label="function generatred")
            # plt.legend()
            # plt.show()

            # chisqr_store = np.empty((len(alphas), len(RVs)))
            scipy_chisqr_store = np.empty((len(alphas), len(RVs)))
            error_chisqr_store = np.empty((len(alphas), len(RVs)))
            new_scipy_chisqr_store = np.empty((len(alphas), len(RVs)))
            new_error_chisqr_store = np.empty((len(alphas), len(RVs)))
            # Save the results to a file to stop repeating loops

            for key, val in chisqr_snr_dict.items():
                np.save(os.path.join(path,
                        "scipy_chisquare_data_snr_{0}_res{1}".format(key,
                                                                     resolution
                                                                     )
                                     ), val)
            for key, val in error_chisqr_snr_dict.items():
                np.save(os.path.join(path,
                        "error_chisquare_data_snr_{0}_res{1}".format(key,
                                                                     resolution
                                                                     )
                                     ), val)
            # Store in dictionary
            res_stored_chisquared[resolution] = chisqr_snr_dict
            res_error_stored_chisquared[resolution] = error_chisqr_snr_dict

            print("SNR Loop time = {}".format(time.time() - loop_start))

    print("Finished Resolution {}".format(resolution))
    # Save the results to a file to stop repeating loops
    X, Y = np.meshgrid(RVs, alphas)
    np.save(os.path.join(path, "RV_mesgrid"), X)
    np.save(os.path.join(path, "alpha_meshgrid"), Y)
    np.save(os.path.join(path, "snr_values"), snrs)
    np.save(os.path.join(path, "Resolutions"), Resolutions)

            # mprocPool.map(wrapper_parallel_chisquare, args_generator)

    with open(os.path.join(path, "input_params.pickle"), "wb") as f:
        pickle.dump(input_parameters, f)
            print(scipy_memmap)
            res_snr_chisqr_dict[resolution][snr] = np.copy(scipy_memmap)
            error_res_snr_chisqr_dict[resolution][snr] = np.copy(my_chisqr_memmap)

    # mprocPool.close()
    timeEnd = dt.now()
    print("Multi-Proc chisqr has been completed in "
          "{} using {}/{} cores.\n".format(timeEnd-timeInit, numProcs,
                                           mprocess.cpu_count()))

    # Save the results to a file to stop repeating loops
    # Try pickling the data

    with open(os.path.join(path, "alpha_chisquare.pickle"), "wb") as f:
        pickle.dump((Resolutions, snrs, X, Y, res_stored_chisquared, res_error_stored_chisquared), f)

        with open(os.path.join(path, "new_res_snr_chisquare.pickle"), "wb") as f:
            pickle.dump((Resolutions, snrs, X, Y, res_snr_storage_dict, error_res_snr_storage_dict), f)


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time to run = {} seconds".format(time.time()-start))

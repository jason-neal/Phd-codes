#!/usr/bin/env python
# -*- coding: utf8 -*-
""" Codes for Telluric contamination removal 
    Interpolates telluric spectra to the observed spectra.
    Divides spectra telluric spectra
    can plot result

"""
import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.io import fits
import argparse
import GaussianFitting as gf
import Obtain_Telluric as obt

def divide_spectra(spec_a, spec_b):
    """ Assumes that the spectra have been interpolated to same wavelength step"""
    """ Divide two spectra"""
    assert(len(spec_a) == len(spec_b)), "Not the same length"
    divide = spec_a / spec_b
    return divide

def match_wl(wl, spec, ref_wl, method="scipy", kind="linear"):
    """Interpolate Wavelengths of spectra to common WL
    Most likely convert telluric to observed spectra wl after wl mapping performed"""
    starttime = time.time()
    if method == "scipy":
        print(kind + " scipy interpolation")
        linear_interp = interp1d(wl, spec, kind=kind)
        new_spec = linear_interp(ref_wl)
    elif method == "numpy":
        if kind.lower() is not "linear":
            print("Warning: Cannot do " + kind + " interpolation with numpy, switching to linear" )
        print("Linear numpy interpolation")
        new_spec = np.interp(ref_wl, wl, spec)  # 1-d peicewise linear interpolat
    else:
        print("Method was given as " + method)
        raise("Not correct interpolation method specified")
    print("Interpolation Time = " + str(time.time() - starttime) + " seconds")

    return new_spec  # test inperpolations 

def plot_spectra(wl, spec, colspec="k.-", label=None, title="Spectrum"):
    """ Do I need to replicate plotting code?
     Same axis
    """
    plt.plot(wl, spec, colspec, label=label)
    plt.title(title)
    plt.legend()
    plt.show(block=False)
    return None
    
# def test_plot_interpolation(x1, y1, x2, y2, methodname=None):
#     """ Plotting code """
#     plt.plot(x1, y1, label="original values")
#     plt.plot(x2, y2, label="new points")
#     plt.title("testing Interpolation: ", methodname)
#     plt.legend()
#     plt.xlabel("Wavelength (nm)")
#     plt.ylabel("Norm Intensity")
#     plt.show()
#     return None

def telluric_correct(wl_obs, spec_obs, wl_tell, spec_tell, kind="linear", method="scipy"):
    """Code to contain other functions in this file

     1. Interpolate spectra to same wavelengths with match_wl()
     2. Divide by Telluric
     3.   ...
    """
    interped_spec = match_wl(wl_tell, spec_tell, wl_obs, kind=kind, method=method)
    
    corrected_spec = divide_spectra(spec_obs, interped_spec) # Divide by telluric spectra

    # other corrections?
    
    return corrected_spec

def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Telluric Removal')
    parser.add_argument('fname', help='Input fits file')
    parser.add_argument('-o', '--output', default=False,
                        help='Ouput Filename')
    parser.add_argument('-k', '--kind', default="linear",
                        help='Interpolation order, linear, quadratic or cubic')
    parser.add_argument('-m', '--method', default="scipy",
                        help='Interpolation method numpy or scipy')
    args = parser.parse_args()
    return args

def main(fname, output=False, kind="linear", method="scipy"):
    homedir = os.getcwd()
    data = fits.getdata(fname)
    wl = data["Wavelength"] 
    I = data["Extracted_DRACS"]
    hdr = fits.getheader(fname)
    datetime = hdr["DATE-OBS"]
    obsdate, obstime = datetime.split("T")
    obstime, __ = obstime.split(".")
    tellpath = "/home/jneal/Phd/data/Tapas/"
    tellname = obt.get_telluric_name(tellpath, obsdate, obstime) 
    print("tell name", tellname)

    tell_data = obt.load_telluric(tellpath, tellname[0])

    wl_lower = np.min(wl/1.0001)
    wl_upper = np.max(wl*1.0001)
    tell_data = gf.slice_spectra(tell_data[0], tell_data[1], wl_lower, wl_upper)
    #tell_data =
    # print("After slice spectra")
    # plt.figure()
    # plt.plot(wl, I, label="Spectra")
    # plt.plot(tell_data[0], tell_data[1], label="Telluric lines")
    # plt.legend()
    # plt.show()

    # Loaded in the data
    # Now perform the telluric removal

    I_corr = telluric_correct(wl, I, tell_data[0], tell_data[1], kind=kind, method=method)

    #print("After telluric_correct")
    plt.figure()
    plt.plot(wl, I, label="Observed Spectra")
    plt.plot(wl, I_corr, label="Corrected Spectra")
    plt.plot(tell_data[0], tell_data[1], label="Telluric Spectra")
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    args = vars(_parser())
    fname = args.pop('fname')
    opts = {k: args[k] for k in args}

    main(fname, **opts)



    # """ Some test code for testing functions """
    # sze = 20
    # x2 = range(sze)
    # y2 = np.random.randn(len(x2)) + np.ones_like(x2)
    # y2 = 0.5 * np.ones_like(x2)
    # x1 = np.linspace(1, sze-1.5, 9)
    # y1 = np.random.randn(len(x1)) + np.ones_like(x1)
    # y1 = np.ones_like(x1)
    # print(x1)
    # print(x2)
    # #print(y1)
    # #print(y2)
    # y1_cor = telluric_correct(x1, y1, x2, y2)
    # print(x1)
    # print(y1)
    # print(y1_cor)  
    
    

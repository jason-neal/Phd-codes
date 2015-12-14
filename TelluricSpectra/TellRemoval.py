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

def telluric_correct(wl_obs, spec_obs, wl_tell, spec_tell, obs_airmass, tell_airmass, kind="linear", method="scipy"):
    """Code to contain other functions in this file

     1. Interpolate spectra to same wavelengths with match_wl()
     2. Divide by Telluric
     3.   ...
    """
    interped_tell = match_wl(wl_tell, spec_tell, wl_obs, kind=kind, method=method)
    
    """ from Makee: Atmospheric Absorption Correction
    Assuming that the telluric spectra I have is equvilant to the star 
    flux and steps 1-3 have been done by using the Tapas spectrum and 
    my wavelngth calibration procedure.
    
    Step 4: Exponential correction to the flux level of the star
    new_star_flux = old_star_flux**B where B is a function of the 
    airmass. B = (object airmass)/(star airmass)

    """

    bad_correction = divide_spectra(spec_obs, interped_tell)
    
    B = obs_airmass/tell_airmass
    print("Airmass Ratio B = ", B)

    bmin, bminpeaks, bslopes = B_minimization(wl_obs, spec_obs, interped_tell, B_init=False)
    print("Minimized B value = ", bmin)
    print("Minimized peaks B value = ", bminpeaks)
    print("Minimized slopes B value = ", bslopes)

    new_tell = interped_tell ** B
    corrected_spec = divide_spectra(spec_obs, new_tell) # Divide by telluric spectra
    new_tell_min = interped_tell ** bmin
    corrected_min_spec = divide_spectra(spec_obs, new_tell_min) # Divide by telluric spectra
    new_tell_minpeaks = interped_tell ** bminpeaks
    corrected_minpeaks_spec = divide_spectra(spec_obs, new_tell_minpeaks) # Divide by telluric spectra
    new_tell_slopes = interped_tell ** bslopes
    corrected_slopes_spec = divide_spectra(spec_obs, new_tell_slopes) # Divide by telluric spectra

    # other corrections?
    
    return corrected_spec, interped_tell, bad_correction, new_tell, corrected_min_spec, new_tell_min, corrected_minpeaks_spec, new_tell_minpeaks, corrected_slopes_spec, new_tell_slopes


def B_minimization(wl, spec_obs, spec_tell, B_init=False):
    """
    Find Optimal B that scales the telluric spectra to best match the
    intesity of the observed spectra
    """
    blist = np.linspace(0.20, 2, 100)
    diffs = []
    peakdiffs = []
    Slopediffs = []
    peaks = spec_tell<0.98
    obs_peaks = spec_obs[peaks]
    tell_peaks = spec_tell[peaks]

    for bb in blist:
        diffs.append(sum(abs(spec_obs - spec_tell**bb)))
        peakdiffs.append(sum(abs(obs_peaks - tell_peaks**bb))) 
        tell = spec_tell**bb
        corr = spec_obs/tell
        len(corr[1:])
        len(corr[:-1])
        slopes = corr[1:]-corr[:-1]
        Slopediffs.append(sum(abs(slopes)))

    plt.figure()
    plt.plot(blist, diffs, label="all")
    plt.plot(blist, peakdiffs, label="<0.98")
    plt.plot(blist, Slopediffs, label="Slopes")
    plt.xlabel("b values")
    plt.ylabel("Summed difference")
    plt.title("B minimization testing")
    plt.legend()
    plt.show()

    B = blist[diffs.index(min(diffs))]
    Bpeaks = blist[peakdiffs.index(min(peakdiffs))]
    Bslope =  blist[Slopediffs.index(min(Slopediffs))]
    #print("B min =", B)
    #print("B peaks min = ", Bpeaks)
    #print("B slopes min = ", Bslope)
    return B, Bpeaks, Bslope



def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Telluric Removal')
    parser.add_argument('fname', help='Input fits file')
    parser.add_argument('-x', '--export', default=False,
                        help='Export result to fits file True/False')
    parser.add_argument('-o', '--output', default=False,
                        help='Ouput Filename')
    parser.add_argument('-k', '--kind', default="linear",
                        help='Interpolation order, linear, quadratic or cubic')
    parser.add_argument('-m', '--method', default="scipy",
                        help='Interpolation method numpy or scipy')
    args = parser.parse_args()
    return args


def export_correction_2fits(filename, wavelength, corrected, original, telluric, hdr, hdrkeys, hdrvals, tellhdr):
    """ Write Telluric Corrected spectra to a fits table file"""
    col1 = fits.Column(name="Wavelength", format="E", array=wavelength) # colums of data
    col2 = fits.Column(name="Corrected_DRACS", format="E", array=corrected)
    col3 = fits.Column(name="Extracted_DRACS", format="E", array=original)
    col4 = fits.Column(name="Interpolated_Tapas", format="E", array=telluric)
    cols = fits.ColDefs([col1, col2, col3, col4])
    tbhdu = fits.BinTableHDU.from_columns(cols) # binary tbale hdu
    prihdr = append_hdr(hdr, hdrkeys, hdrvals)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    # telluric head to go as a second extension !!!

    #print("Writing to fits file")
    thdulist.writeto(filename, output_verify="silentfix")   # Fixing errors to work properly
    return None

# could make new module for fits handlers like this
def append_hdr(hdr, keys, values ,item=0):
    ''' Apend/change parameters to fits hdr, 
    can take list or tuple as input of keywords 
    and values to change in the header 
    Defaults at changing the header in the 0th item 
    unless the number the index is givien,
    If a key is not found it adds it to the header'''
    
    if type(keys) == str:           # To handle single value
        hdr[keys] = values
    else:
        assert len(keys) == len(values), 'Not the same number of keys as values' 
        for i in range(len(keys)):
            hdr[keys[i]] = values[i]
            print(repr(hdr[-2:10]))
    return hdr


def main(fname, export=False, output=False, kind="linear", method="scipy"):
    homedir = os.getcwd()
    data = fits.getdata(fname)
    wl = data["Wavelength"] 
    I = data["Extracted_DRACS"]
    hdr = fits.getheader(fname)
    datetime = hdr["DATE-OBS"]
    airmass_start = hdr["HIERARCH ESO TEL AIRM START"]
    airmass_end = hdr["HIERARCH ESO TEL AIRM END"]
    obs_airmass = (airmass_start + airmass_end)/2
    print("Starting Airmass", airmass_start, "Ending Airmass", airmass_end)
    obsdate, obstime = datetime.split("T")
    obstime, __ = obstime.split(".")
    tellpath = "/home/jneal/Phd/data/Tapas/"
    tellname = obt.get_telluric_name(tellpath, obsdate, obstime) 
    print("tell name", tellname)
    
    tell_data, tell_hdr = obt.load_telluric(tellpath, tellname[0])
    tell_airmass = float(tell_hdr["airmass"])
    print("Telluric Airmass ", tell_airmass)
    wl_lower = np.min(wl/1.0001)
    wl_upper = np.max(wl*1.0001)
    tell_data = gf.slice_spectra(tell_data[0], tell_data[1], wl_lower, wl_upper)
    
    # Telluric Normalization (use first 50 points below 1.2 as constant continuum)
    I_tell = tell_data[1]
    maxes = I_tell[(I_tell < 1.2)].argsort()[-50:][::-1]
    tell_data = (tell_data[0], tell_data[1] / np.median(I_tell[maxes]))
    print("Telluric normaliztion value", np.median(I_tell[maxes]))

    # print("After slice spectra")
    # plt.figure()
    # plt.plot(wl, I, label="Spectra")
    # plt.plot(tell_data[0], tell_data[1], label="Telluric lines")
    # plt.legend()
    # plt.show()

    # Loaded in the data
    # Now perform the telluric removal

    I_corr, Tell_interp, bad_correction, tell_Amass_corr, I_corr_min, tell_Amass_corr_min, I_corr_minpeaks, new_tell_minpeaks, I_corr_slopes, new_tell_slopes = telluric_correct(wl, I, tell_data[0], tell_data[1], obs_airmass, tell_airmass, kind=kind, method=method)

    #print("After telluric_correct")
    plt.figure()
    plt.plot(wl, I,"k", label="Observed Spectra")
    plt.plot(wl, Tell_interp, label="Telluric")
    plt.plot(wl, bad_correction, label="Base Correction")
    plt.plot(wl, I_corr, label="Corrected B")
    plt.plot(wl, tell_Amass_corr, label="Telluric ** B")
    plt.plot(wl, I_corr_min, label="Corrected minimized B")
    plt.plot(wl, tell_Amass_corr_min, label="Telluric minimized B")
    plt.plot(wl, I_corr_minpeaks, label="Corrected minimized peaks B")
    plt.plot(wl, new_tell_minpeaks, label="Telluric minimized peaks B")
    plt.plot(wl, I_corr_slopes, label="Corrected minimized slopes B")
    plt.plot(wl, new_tell_slopes, label="Tell minimized slopes B")
    plt.plot(wl, np.ones_like(wl), "--")
    plt.legend(loc="best")
    plt.show()
    

    ### SAVING Telluric Corrected Spectra ###
    # PROBABALY NEED TO HARDCODE IN THE HEADER LINES...
    os.chdir(homedir)   # to make sure saving where running
    if output:  
            Output_filename = output
    else:
            Output_filename = fname.replace(".fits", ".tellcorr.fits")
    hdrkeys = ["Correction", "Tapas Interpolation method", "Interpolation kind", "Correction Params A, B"]
    hdrvals = [("Tapas divion","Spectra Correction"),(method, "numpy or scipy"),(kind, "linear,slinear,quadratic,cubic"),("Blankety ", "Blank")]
    tellhdr = False   ### need to correctly get this from obtain telluric
    
    if export:
        export_correction_2fits(Output_filename, wl, I_corr, I, Tell_interp, hdr, hdrkeys, hdrvals, tellhdr)
        print("Saved coorected telluric spectra to " + str(Output_filename))
    else:
        print("Skipped Saving coorected telluric spectra ")


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
    
    

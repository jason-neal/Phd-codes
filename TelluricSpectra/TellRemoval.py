#!/usr/bin/env python3
# -*- coding: utf8 -*-
""" Codes for Telluric contamination removal 
    Interpolates telluric spectra to the observed spectra.
    Divides spectra telluric spectra
    can plot result

"""
from __future__ import division, print_function
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
from SpectralTools import wav_selector, wl_interpolation

def divide_spectra(spec_a, spec_b):
    """ Assumes that the spectra have been interpolated to same wavelength step"""
    """ Divide two spectra"""
    assert(len(spec_a) == len(spec_b)), "Not the same length"
    divide = spec_a / spec_b
    return divide


def plot_spectra(wl, spec, colspec="k.-", label=None, title="Spectrum"):
    """ Do I need to replicate plotting code?
     Same axis
    """
    plt.plot(wl, spec, colspec, label=label)
    plt.title(title)
    plt.legend()
    plt.show(block=False)
    return None
    
def airmass_scaling(spectra, spec_airmass, obs_airmass):
    """Scale the Telluric spectra to match the airmass of the observation"""
    B = obs_airmass / spec_airmass
    new_spec = spectra ** B
    return new_spec 

def telluric_correct(wl_obs, spec_obs, wl_tell, spec_tell, obs_airmass, tell_airmass, kind="linear", method="scipy"):
    """Code to contain other functions in this file

     1. Interpolate spectra to same wavelengths with wl_interpolation()
     2. Divide by Telluric (bad correction)
     3. Scale telluric and divide again (Better correction)
     4.   ...
    """
    interped_tell = wl_interpolation(wl_tell, spec_tell, wl_obs, kind=kind, method=method)
    
    """ from Makee: Atmospheric Absorption Correction
    Assuming that the telluric spectra I have is equvilant to the star 
    flux and steps 1-3 have been done by using the Tapas spectrum and 
    my wavelngth calibration procedure.
    
    Step 4: Exponential correction to the flux level of the star
    new_star_flux = old_star_flux**B where B is a function of the 
    airmass. B = (object airmass)/(star airmass)

    """
    Corrections = []
    Correction_Bs = []
    Correction_tells = []
    Correction_labels = []

    bad_correction = divide_spectra(spec_obs, interped_tell)
    Corrections.append(bad_correction)
    Correction_Bs.append(1)
    Correction_tells.append(interped_tell)
    Correction_labels.append("No Airmass B Correction")
 
    B = obs_airmass/tell_airmass
    print("Airmass scaling Ratio B = ", B)

    new_tell = airmass_scaling(interped_tell, tell_airmass, obs_airmass)
    corr_spec = divide_spectra(spec_obs, new_tell) # Divide by telluric spectra

    Corrections.append(corr_spec)
    Correction_Bs.append(B)
    Correction_tells.append(new_tell)
    Correction_labels.append("Header values B Correction")

    return Corrections, Correction_tells, Correction_Bs, Correction_labels



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
    """Apend/change parameters to fits hdr, 
    can take list or tuple as input of keywords 
    and values to change in the header 
    Defaults at changing the header in the 0th item 
    unless the number the index is givien,
    If a key is not found it adds it to the header"""
    
    if type(keys) == str:           # To handle single value
        hdr[keys] = values
    else:
        assert len(keys) == len(values), 'Not the same number of keys as values' 
        for i in range(len(keys)):
            hdr[keys[i]] = values[i]
            print(repr(hdr[-2:10]))
    return hdr

def get_observation_averages(homedir):
    """ Based on arrangement of the Organise IRAF script I have to 
    clean up IRAF output files mess
    
    homedir is the path where the function is being called from where the file is.
    output:
        average obs airmass value
        average obs time value

    Uses the list_spectra.txt that is in this directory to open each file and extract values from the headers.
    *Possibly add extra extensions to the headrer in the future when combining.
    """
    Raw_path = homedir[:-13] + "Raw_files/"
    list_name = "list_spectra.txt"

    Nod_airmass = []
    Nod_median_time = []
    with open(list_name, "r") as f:
        for line in f:
            fname = line[:-1] + ".fits"
            hdr = fits.getheader(Raw_path + fname)
            datetime = hdr["DATE-OBS"]
            time = datetime[11:19]
            airmass_start = hdr["HIERARCH ESO TEL AIRM START"]
            airmass_end = hdr["HIERARCH ESO TEL AIRM END"]
            Nod_mean_airmass = round((airmass_start + airmass_end) / 2 , 4)
            Nod_airmass.append(Nod_mean_airmass)
            Nod_median_time.append(time)
    
    print("Observation Nod_airmass ", Nod_airmass)
    print("Observation Nod_time ", Nod_median_time)
    return np.mean(Nod_airmass), Nod_median_time

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
    parser.add_argument('-t', '--tellpath', default=False,
                        help='Path to find the telluric spectra to use.')
    parser.add_argument('-k', '--kind', default="linear",
                        help='Interpolation order, linear, quadratic or cubic')
    parser.add_argument('-m', '--method', default="scipy",
                        help='Interpolation method numpy or scipy')
    parser.add_argument("-s", "--show", default=True, 
                        help="Show plots") #Does not wokwithout display though for some reason
    parser.add_argument("-c", "--h2o_scaling", action='store_true',
                        help="Perform separate H20 scaling")
    parser.add_argument("-n", "--new_method", action='store_true',
                        help="Use new code method")
    args = parser.parse_args()
    return args

def main(fname, export=False, output=False, tellpath=False, kind="linear", method="scipy", show=False, h2o_scaling=False, new_method=False):
    # Set and test homedir
    homedir = os.getcwd()
    if homedir[-13:] is not "Combined_Nods":
        print("Not running telluric removal from Combined_Nods folder \n Crashing")
    
    # Load in Crires Spectra
    data = fits.getdata(fname)
    hdr = fits.getheader(fname)
    
    wl = data["Wavelength"] 
    I = data["Extracted_DRACS"]
    
    # Wavelength bounds to select the telluric spectra
    wl_lower = np.min(wl)/1.0001
    wl_upper = np.max(wl)*1.0001


 #################################################################  NEW METHOD section ##########################################################
    if new_method:
        # Changing for new telluric line location defaults (inside the Combined_nods)
        if h2o_scaling:
            # load separated H20 tapas datasets
            tapas_h20 = "../HD30501_data/1/tapas_2012-04-07T00-24-03_ReqId_12_No_Ifunction_barydone-NO.ipac"
            tapas_not_h20 = "../HD30501_data/1/tapas_2012-04-07T00-24-03_ReqId_18_R-50000_sratio-10_barydone-NO.ipac"
            tapas_h20_data, tapas_h20_hdr = obt.load_telluric("", tapas_h20)
            tapas_not_h20_data, tapas_not_h20_hdr = obt.load_telluric("", tapas_not_h20)
            tapas_airmass = float(tapas_h20_hdr["airmass"])
            
            # Select section by wavelength
            tell_h20_section = wav_selector(tapas_h20_data[0], tapas_h20_data[1], wl_lower, wl_upper)
            tell_not_h20_section = wav_selector(tapas_not_h20_data[0], tapas_not_h20_data[1], wl_lower, wl_upper)


        else:
            # load combined dataset only
            tapas_all = "../HD30501_data/1/tapas_2012-04-07T00-24-03_ReqId_10_R-50000_sratio-10_barydone-NO.ipac"
            tapas_all_data, tapas_all_hdr = obt.load_telluric("", tapas_all)
            tapas_airmass = float(tapas_all_hdr["airmass"])

            # Select section by wavelength
            tell_all_section = wav_selector(tell_all_data[0], tell_all_data[1], wl_lower, wl_upper)

            #  values needed for header
            #H20_scaling_val = None
            #resolution_val = None
    ################################################# REPLACING this / or if still given different location for tapas files#######################
    else:   # old method

        # Get airmass for entire observation
        #airmass_start = hdr["HIERARCH ESO TEL AIRM START"]
        #airmass_end = hdr["HIERARCH ESO TEL AIRM END"]
        #obs_airmass = (airmass_start + airmass_end) / 2
        Average_airmass, average_time = get_observation_averages(homedir)
        """ When using averaged airmass need almost no airmass scalling of 
            model as it is almost the airmass given by tapas"""
        obs_airmass = Average_airmass
        print("From all 8 raw spectra: \nAverage_airmass", Average_airmass, "\nAverage_time", average_time)

        obs_datetime = hdr["DATE-OBS"]
        obsdate, obstime = obs_datetime.split("T")
        obstime, __ = obstime.split(".")
        print("tellpath before", tellpath)

        if tellpath:
            tellname = obt.get_telluric_name(tellpath, obsdate, obstime) 
        else:
            tellpath = "/home/jneal/Phd/data/Tapas/"
            tellname = obt.get_telluric_name(tellpath, obsdate, obstime) 
        print("Returned Mathching filenames", tellname)
        
        print("tellpath after", tellpath)
        assert len(tellname) < 2, "Multiple tapas filenames match"
        
        tell_data, tell_hdr = obt.load_telluric(tellpath, tellname[0])
        #print("Telluric Header ", tell_hdr)
        tell_airmass = float(tell_hdr["airmass"])
        print("Observation Airmass ", obs_airmass)
        print("Telluric Airmass ", tell_airmass)
        tell_respower = int(float((tell_hdr["respower"])))
        print("Telluric Resolution Power =", tell_respower)
        
        #wl_lower = np.min(wl)/1.0001
        #wl_upper = np.max(wl)*1.0001
        tell_data = wav_selector(tell_data[0], tell_data[1], wl_lower, wl_upper)
        
        # Telluric Normalization (use first 50 points below 1.2 as constant continuum)
        # For selected section
        I_tell = tell_data[1]
        maxes = I_tell[(I_tell < 1.2)].argsort()[-50:][::-1]
        tell_data = (tell_data[0], tell_data[1] / np.median(I_tell[maxes]))
        print("Telluric normaliztion value", np.median(I_tell[maxes]))


        Corrections, Correction_tells, Correction_Bs, Correction_labels = telluric_correct(wl, I, tell_data[0], tell_data[1], obs_airmass, tell_airmass, kind=kind, method=method)
    
        if show:
            plt.figure()  # Tellurics
            plt.plot(wl, I, "--", linewidth=2, label="Observed Spectra")
            for corr, tell, B, label in zip(Corrections, Correction_tells, Correction_Bs, Correction_labels):
                #plt.plot(wl, corr, "--", label=(label + ", B = {0:.2f}".format(B)))
                plt.plot(wl, tell, linewidth=2, label=("Telluric " + label + ", B = {0:.3f}".format(B)))
                plt.plot(wl, np.ones_like(wl), "-.")
                plt.legend(loc="best")
                plt.title("Telluric Scaling with Tapas Resolution power = {}".format(tell_respower))

            plt.figure() # Corrections
            plt.plot(wl, I, "--", linewidth=2, label="Observed Spectra")
            for corr, tell, B, label in zip(Corrections, Correction_tells, Correction_Bs, Correction_labels):
                plt.plot(wl, corr, linewidth=2, label=(label + ", B = {0:.3f}".format(B)))
                #plt.plot(wl, tell, label=("Telluric " + label + ", B = {0:.2f}".format(B)))
                plt.plot(wl, np.ones_like(wl), "-.")
                plt.legend(loc="best")
                plt.title("Telluric Corrections with tapas Resolution power = {}".format(tell_respower))

            plt.show()

        # B corr is almost not needed but include here for now 31/3/16 to make a correction 
        print(Correction_labels)
        print(Corrections)
        I_corr = Corrections[1]  # using B scaling
        Tell_interp = Correction_tells[1]   


    ###########################################################   Ends  HERE #################################################################################  

    ### SAVING Telluric Corrected Spectra ###
    # PROBABALY NEED TO HARDCODE IN THE HEADER LINES...
    os.chdir(homedir)   # to make sure saving where running

    ### TO DO add mutually exclusive flag (with output) to add extra suffixs on end by .tellcorr.
    if output:  
            Output_filename = output
    else:
            Output_filename = fname.replace(".fits", ".tellcorr.fits")

    # Work out values for header
    if new_method:
        if h2o_scaling:
        
            # To be updated when implemented
            H20_scaling_val = None
            resolution_val = None
        else:
            H20_scale_val = None
            resolution_val = None

        # Keys and values for Fits header file        
        hdrkeys = ["Correction", "Tapas Interpolation method", "Interpolation kind", "Correction Params A, B", "H20 Scaling", "H20 Scaling Value", "Convolution R"]
        hdrvals = [("Tapas division" ,"Spectral Correction"), (method, "numpy or scipy"), (kind, "linear,slinear,quadratic,cubic"), (h2o_scaling, "Was separate H20 scaling 1 = Yes"), ( H20_scale_val, "H20 scale value used"), (resolution_val , "Resolution used for H20 Convolution")]
        tellhdr = False   ### need to correctly get this from obtain telluric
    else:   # Old method 
        # Keys and values for Fits header file        
        hdrkeys = ["Correction", "Tapas Interpolation method", "Interpolation kind", "Correction Params A, B", "H20 Scaling"]
        hdrvals = [("Tapas division" ,"Spectral Correction"), (method, "numpy or scipy"), (kind, "linear,slinear,quadratic,cubic"), (h2o_scaling, "Was separate H20 scaling Done 1 = Yes")]
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

    

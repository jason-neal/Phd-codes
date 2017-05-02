#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""Telluric Removal.

Removes the telluric contimination useing tapas spectra.

Codes for Telluric contamination removal
Interpolates telluric spectra to the observed spectra.
Divides spectra telluric spectra
can plot result.

"""
from __future__ import division, print_function
import os
import lmfit
# import time
import logging
import argparse
import numpy as np
# import pandas as pd
# import scipy as sp
from logging import debug
from debug_utils import pv
from astropy.io import fits
import Obtain_Telluric as obt
# import GaussianFitting as gf
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from Get_filenames import get_filenames
# from scipy.interpolate import interp1d
from IP_multi_Convolution import ip_convolution
from SpectralTools import wav_selector, wl_interpolation, instrument_convolution


from eniric.IOmodule import pdwrite_cols
from eniric.atmosphere import barycenter_shift


def setup_debug(debug_val):
    """Set debug level."""
    if debug_val:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    return None


def divide_spectra(spec_a, spec_b):
    """ Assumes that the spectra have been interpolated to same wavelength step

    Divide two spectra
    """
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

    B = obs_airmass / tell_airmass
    print("Airmass scaling Ratio B = ", B)

    # new_tell = interped_tell ** B
    new_tell = airmass_scaling(interped_tell, tell_airmass, obs_airmass)
    corr_spec = divide_spectra(spec_obs, new_tell)  # Divide by telluric spectra

    Corrections.append(corr_spec)
    Correction_Bs.append(B)
    Correction_tells.append(new_tell)
    Correction_labels.append("Header values B Correction")
    return Corrections, Correction_tells, Correction_Bs, Correction_labels


def export_correction_2fits(filename, wavelength, corrected, original, telluric, hdr, hdrkeys, hdrvals, tellhdr):
    """ Write Telluric Corrected spectra to a fits table file"""
    col1 = fits.Column(name="Wavelength", format="E", array=wavelength)  # colums of data
    col2 = fits.Column(name="Corrected_DRACS", format="E", array=corrected)
    col3 = fits.Column(name="Extracted_DRACS", format="E", array=original)
    col4 = fits.Column(name="Interpolated_Tapas", format="E", array=telluric)
    cols = fits.ColDefs([col1, col2, col3, col4])
    tbhdu = fits.BinTableHDU.from_columns(cols)  # binary tbale hdu
    prihdr = append_hdr(hdr, hdrkeys, hdrvals)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    # telluric head to go as a second extension !!!

    # print("Writing to fits file")
    thdulist.writeto(filename, output_verify="silentfix")   # Fixing errors to work properly
    return None


# could make new module for fits handlers like this
def append_hdr(hdr, keys, values, item=0):
    """Apend/change parameters to fits hdr,
    can take list or tuple as input of keywords
    and values to change in the header
    Defaults at changing the header in the 0th item
    unless the number the index is givien,
    If a key is not found it adds it to the header.
    """

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
    raw_path = homedir[:-13] + "Raw_files/"
    list_name = "list_spectra.txt"

    nod_airmass = []
    nod_median_time = []
    with open(list_name, "r") as f:
        for line in f:
            fname = line[:-1] + ".fits"
            hdr = fits.getheader(raw_path + fname)
            datetime = hdr["DATE-OBS"]
            time = datetime[11:19]
            airmass_start = hdr["HIERARCH ESO TEL AIRM START"]
            airmass_end = hdr["HIERARCH ESO TEL AIRM END"]
            nod_mean_airmass = round((airmass_start + airmass_end) / 2, 4)
            nod_airmass.append(nod_mean_airmass)
            nod_median_time.append(time)

    print("Observation Nod_airmass ", nod_airmass)
    print("Observation Nod_time ", nod_median_time)
    return np.mean(nod_airmass), nod_median_time


def h20_residual(params, obs_data, telluric_data):
    # Parameters
    scale_factor = params["scale_factor"].value
    R = params["R"].value
    fwhm_lim = params["fwhm_lim"].value
    # n_jobs = params["n_jobs"].value  # parallel implementaiton
    # chip_select = params["chip_select"].value
    verbose = params["verbose"].value
    fit_lines = params["fit_lines"].value  # if true only fit areas deeper than 0.995

    # Data
    obs_wl = obs_data[0]
    obs_I = obs_data[1]
    telluric_wl = telluric_data[0]
    telluric_I = telluric_data[1]

    # Telluric scaling T ** x
    scaled_telluric_I = telluric_I ** scale_factor

    # smallest wl step in telluric wl
    min_dwl = np.min(telluric_wl[1:] - telluric_wl[:-1])
    # Make sure atleast 1 telluric value is outside wl range of observation for interpoltion later
    chip_limits = [obs_wl[0] - 2 * min_dwl, obs_wl[-1] + 2 * min_dwl]
    # Convolution
    # def convolution_nir(wav, flux, chip, R, fwhm_lim=5.0, plot=True):
    #    return [wav_chip, flux_conv_res]
    # conv_tell_wl, conv_tell_I = instrument_convolution(telluric_wl, scaled_telluric_I, chip_limits,
    #                                                    R, fwhm_lim=fwhm_lim, plot=False, verbose=verbose)
    conv_tell_wl, conv_tell_I = ip_convolution(telluric_wl, scaled_telluric_I, chip_limits,
                                               R, fwhm_lim=fwhm_lim, plot=False, verbose=verbose)

    # print("Obs wl- Min ", np.min(obs_wl)," Max ", np.max(obs_wl))
    # print("Input telluic wl- Min ", np.min(telluric_wl)," Max ", np.max(telluric_wl))
    # print("conv tell wl- Min ", np.min(conv_tell_wl)," Max ", np.max(conv_tell_wl))
    interped_conv_tell = wl_interpolation(conv_tell_wl, conv_tell_I, obs_wl)
    logging.info("Convolution and interpolation inside residual function was done")

    # Mask fit to peaks in telluric data
    if fit_lines:
        # wl_interpolation(telluric_wl, telluric_I, obs_wl)
        Tell_line_mask = wl_interpolation(telluric_wl, telluric_I, obs_wl) < 0.995
        return 1 - (obs_I / interped_conv_tell)[Tell_line_mask]
    else:
        return 1 - (obs_I / interped_conv_tell)


def h2o_telluric_correction(obs_wl, obs_I, h20_wl, h20_I, R):
    """ H20 Telluric correction
    Performs nonlinear least squares fitting to fit the scale factor
    Uses the scale factor
    Then Convolves by the instrument resolution

    """
    params = Parameters()
    params.add('scale_factor', value=1)   # add min and max values ?
    params.add('R', value=R, vary=False)
    params.add('fwhm_lim', value=5, vary=False)
    params.add('fit_lines', value=True, vary=False)   # only fit the peaks of lines < 0.995
    params.add("verbose", value=False, vary=False)

    out = minimize(h20_residual, params, args=([obs_wl, obs_I], [h20_wl, h20_I]))

    outreport = lmfit.fit_report(out)
    print(outreport)

    # Telluric scaling T ** x
    Scaled_h20_I = h20_I ** out.params["scale_factor"].value

    # Convolved_h20_wl, Convolved_h20_I = instrument_convolution(h20_wl, Scaled_h20_I, [h20_wl[0], h20_wl[-1]],
    #                                                           R, fwhm_lim=5, plot=False, verbose=True)
    Convolved_h20_wl, Convolved_h20_I = ip_convolution(h20_wl, Scaled_h20_I, [h20_wl[0], h20_wl[-1]],
                                                       R, fwhm_lim=5, plot=False, verbose=True)

    # assert np.allclose(Convolved_h20_I_2, Convolved_h20_I)
    # print("Convolution Methods give the same result")

    # Interpolation to obs positions
    interp_conv_h20_I = wl_interpolation(Convolved_h20_wl, Convolved_h20_I, obs_wl)

    h20_corrected_obs = divide_spectra(obs_I, interp_conv_h20_I)

    return h20_corrected_obs, interp_conv_h20_I, out, outreport


def telluric_correction(obs_wl, obs_I, obs_airmass, tell_wl, tell_I, spec_airmass):
    """ Set obs_airmas and spec_airmass equal to achieve a scaling factor of 1 = No scaling"""
    # tell_I = airmass_scaling(tell_I, spec_airmass, obs_airmass)
    tell_I = tell_I ** (obs_airmass / spec_airmass)   # Airmass scaling
    interp_tell_I = wl_interpolation(tell_wl, tell_I, obs_wl)
    corrected_obs = divide_spectra(obs_I, interp_tell_I)

    return corrected_obs, interp_tell_I, obs_airmass / spec_airmass


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Telluric Removal')
    parser.add_argument('fname', help='Input fits file')
    parser.add_argument('-e', '--export', action='store_true',
                        help='Export/save results to fits file')
    parser.add_argument('-o', '--output', default=False,
                        help='Ouput Filename')
    parser.add_argument('-t', '--tellpath', default=False,
                        help='Path to find the telluric spectra to use.')
    parser.add_argument('-k', '--kind', default="linear",
                        help='Interpolation order, linear, quadratic or cubic')
    parser.add_argument('-m', '--method', default="scipy",
                        help='Interpolation method numpy or scipy')

    parser.add_argument("-s", "--show", action='store_true',
                        help="Show plots")  # Does not work without a display
    parser.add_argument("-c", "--h2o_scaling", action='store_true',
                        help="Perform separate H20 scaling")
    parser.add_argument("-n", "--new_method", action='store_true',
                        help="Use new code method")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Include debug statements.")
    parser.add_argument("--mask", default=False, action="store_true", help=("Store atmopsheric masks and calculate"
                        " percentage saved."))
    args = parser.parse_args()
    return args


def mask_flux(flux, depth: float=2):
    """depth = percentage depth."""
    depth /= 100.         # turn into fraction.
    if depth < 0 or depth > 1:
        raise ValueError("Depth outside range [0,1]")
    return flux > (1 - depth)


def mask_and_baryshift(tell_wav, tell_trans, depth: float=2):
    """Mask out telluric lines above a given depth,

    Inlcude effect of barycentric shift +-30km/s.
    """
    masked_tell = mask_flux(tell_trans, depth)
    mask_tell_bary = barycenter_shift(tell_wav, masked_tell, consecutive_test=False)

    assert np.all(tell_trans[masked_tell] > (1 - (depth / 100.)))
    assert np.all(tell_trans[mask_tell_bary] > (1 - (depth / 100.)))
    debug(pv("depth"))
    debug(pv("len(tell_trans[masked_tell])"))
    debug(pv("sum(masked_tell)"))

    return masked_tell, mask_tell_bary


def telluric_rv_masks(wav_obs, tell_wav, tell_trans, save_name=False, h20=False):
    """Masks for pixels affected by telluric spectrum.

    Assumes tell_trans is the transmission, i.e 1 = no absorption.

    h20: bool
       Bool for if H20 scalling was used.
    """
    mask_2, masked_bary_2 = mask_and_baryshift(tell_wav, tell_trans, depth=2)
    mask_5, masked_bary_5 = mask_and_baryshift(tell_wav, tell_trans, depth=5)

    #obs_mask_2 = wl_interpolation(tell_wav, mask_2_percent, wav_obs)
    #obs_mask_2_30 = wl_interpolation(tell_wav, mask_2per_30km, wav_obs)
    #obs_mask_5 = wl_interpolation(tell_wav, mask_5_percent, wav_obs)
    #obs_mask_5_30 = wl_interpolation(tell_wav, mask_5per_30km, wav_obs)
    obs_data = [wl_interpolation(tell_wav, mask, wav_obs) for mask in [mask_2, masked_bary_2, mask_5, masked_bary_5]]
    debug(pv("obs_data"))
    # Save as a csv
    if save_name:
        if h20:
            mask_name = save_name.replace(".fits", ".h20tellmasks.txt")
            percent_name = save_name.split(".nod.")[0][:-2] + ".h20tellpercentages.txt"
        else:
            mask_name = save_name.replace(".fits", ".tellmasks.txt")
            percent_name = save_name.split(".nod.")[0][:-2] + ".tellpercentages.txt"
        cols = ["wav_nm", r"depth>2% masked", r"depth>2%+baryshift", r"depth>5% masked", r"depth>5%+baryshift"]
        # pdwrite_cols(filename, *data, **kwargs)
        data = [wav_obs, *obs_data]
        debug(pv("data"))
        pdwrite_cols(mask_name, *data, header=cols)

        # Based on names
        # CRIRE.2012-04-07T00-08-29.976_2.nod.ms.norm.sum.fits
        with open(percent_name, "a") as f:
            f.write("Percentage of spectra covered by deep lines.\n")
            f.write("Detector Chip = {}\n".format(save_name.split(".nod.")[0][-1]))
            for m, n in zip(data[1:], cols[1:]):
                num_good = np.sum(m)
                num_tell = len(m) - num_good

                correct_frac = num_tell / len(m)  # m has

                f.write("{!s:16}\t{:6.02%}\n".format(n, correct_frac))
        debug("Should have saved values to {!s}".format(percent_name))

    return obs_data  # obs_mask_2, obs_mask_2_30, obs_mask_5, obs_mask_5_30


def main(fname, export=False, output=False, tellpath=False, kind="linear", method="scipy",
         show=False, h2o_scaling=False, new_method=False, mask=False):
    # Set and test homedir
    homedir = os.getcwd()
    if homedir[-13:] != "Combined_Nods":
        print("Not running telluric removal from Combined_Nods folder \n Crashing")
        print("Actual directory currently in is", homedir)
        raise ValueError("Not correct path")

    # Load in Crires Spectra
    data = fits.getdata(fname)
    hdr = fits.getheader(fname)

    wl = data["Wavelength"]
    I = data["Extracted_DRACS"]

    # Wavelength bounds to select the telluric spectra
    wl_lower = np.min(wl) / 1.0001
    wl_upper = np.max(wl) * 1.0001

    # Get airmass for entire observation
    # airmass_start = hdr["HIERARCH ESO TEL AIRM START"]
    # airmass_end = hdr["HIERARCH ESO TEL AIRM END"]
    # obs_airmass = (airmass_start + airmass_end) / 2
    Average_airmass, average_time = get_observation_averages(homedir)
    """ When using averaged airmass need almost no airmass scalling of
            model as it is almost the airmass given by tapas """
    obs_airmass = Average_airmass
    print("From all 8 raw spectra: \nAverage_airmass", Average_airmass,
          "\nAverage_time", average_time)

    # Calculate Resolving Power.
    # Using the rule of thumb equation from the CRIRES manual.
    # Check for adaptive optics use.
    horder_loop = hdr["HIERARCH ESO AOS RTC LOOP HORDER"]    # High order loop on or off
    # lgs_loop = hdr["HIERARCH ESO AOS RTC LOOP LGS"]          # LGS jitter loop on or off
    loopstate = hdr["HIERARCH ESO AOS RTC LOOP STATE"]       # Loop state, open or closed
    tiptilt_loop = hdr["HIERARCH ESO AOS RTC LOOP TIPTILT"]  # Tip Tilt loop on or off
    slit_width = hdr["HIERARCH ESO INS SLIT1 WID"]           # Slit width

    if any([horder_loop, loopstate, tiptilt_loop]) and (loopstate != "OPEN"):
        print("Adaptive optics was used - Rule of thumb for Resolution is not good enough")
    else:
        R = int(100000 * 0.2 / slit_width)

    # ################################################  NEW METHOD section ############################
    if new_method:
        # Changing for new telluric line location defaults (inside the Combined_nods)
        if not tellpath:
            tellpath = "./"

        debug(pv("tellpath"))
        if h2o_scaling:
            # load separated H20 tapas datasets

            tapas_h20 = get_filenames(tellpath, "tapas_*", "*_ReqId_12_No_Ifunction*")
            debug(pv("tapas_h20"))

            if len(tapas_h20) > 1:
                print("Warning Too many h20 tapas files returned")
            tapas_not_h20 = get_filenames(tellpath, "tapas_*", "*_ReqId_18_R-*")
            debug(pv("tapas_not_h20"))
            if len(tapas_not_h20) > 1:
                print("Warning Too many h20 tapas files returned")
            # tapas_h20 = "../HD30501_data/1/tapas_2012-04-07T00-24-03_ReqId_12_No_Ifunction_barydone-NO.ipac"
            # tapas_not_h20 = "../HD30501_data/1/tapas_2012-04-07T00-24-03_ReqId_18_R-50000_sratio-10_barydone-NO.ipac"
            tapas_h20_data, tapas_h20_hdr = obt.load_telluric("", tapas_h20[0])
            tapas_not_h20_data, tapas_not_h20_hdr = obt.load_telluric("", tapas_not_h20[0])
            tapas_airmass = float(tapas_h20_hdr["airmass"])

            # Select section by wavelength
            tapas_h20_section = wav_selector(tapas_h20_data[0], tapas_h20_data[1], wl_lower, wl_upper)
            tapas_not_h20_section = wav_selector(tapas_not_h20_data[0], tapas_not_h20_data[1], wl_lower, wl_upper)

            # no h20 correction
            non_h20_correct_I, noh20tell_used, b_used = telluric_correction(wl, I, obs_airmass,
                                                                            tapas_not_h20_section[0],
                                                                            tapas_not_h20_section[1], tapas_airmass)
            # h20 correction and

            (h20_corrected_obs, h20tell_used, out,
                outreport) = h2o_telluric_correction(wl, non_h20_correct_I, tapas_h20_section[0],
                                                     tapas_h20_section[1], R)

            I_corr = h20_corrected_obs
            tell_used = noh20tell_used * h20tell_used   # Combined corrections

        else:
            # load combined dataset only
            tapas_all = get_filenames(tellpath, "tapas_*", "*_ReqId_10_R-*")
            debug(pv("tapas_all"))

            if len(tapas_all) > 1:
                print("Warning Too many h20 tapas files returned")
            # tapas_all = "../HD30501_data/1/tapas_2012-04-07T00-24-03_ReqId_10_R-50000_sratio-10_barydone-NO.ipac"
            tapas_all_data, tapas_all_hdr = obt.load_telluric("", tapas_all[0])
            tapas_airmass = float(tapas_all_hdr["airmass"])

            # Select section by wavelength
            # tapas_all_section = wav_selector(tapas_all_data[0], tapas_all_data[1], wl_lower, wl_upper)
            # Unneeded due to interpolation

            I_corr, tell_used, b_used = telluric_correction(wl, I, obs_airmass, tapas_all_data[0],
                                                            tapas_all_data[1], tapas_airmass)

        if show:
            plt.figure()  # Corrections
            plt.plot(wl, I, "--", linewidth=2, label="Observed Spectra")
            plt.plot(wl, I_corr, linewidth=2, label=("Corrected spectra"))
            # plt.plot(wl, tell, label=("Telluric " + label + ", B = {0:.2f}".format(B)))
            plt.hlines(1, wl[0], wl[-1], color="grey", linestyles='dashed')
            plt.legend(loc="best")
            plt.title("Telluric Corrections")

            plt.show()

    # ################# REPLACING this / or if still given different location for tapas files#######
    else:   # old method

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
        debug(pv("tellname"))
        if len(tellname) > 1:
            suitable_index = [i for i, name in enumerate(tellname) if "ReqId_10" in name]
            if len(suitable_index) == 1:
                tellname = [tellname[suitable_index[0]]]
            else:
                raise(ValueError("Issue with Tapas filenames"))
        assert len(tellname) < 2, "Multiple tapas filenames match"

        tell_data, tell_hdr = obt.load_telluric(tellpath, tellname[0])
        # print("Telluric Header ", tell_hdr)
        tell_airmass = float(tell_hdr["airmass"])
        print("Observation Airmass ", obs_airmass)
        print("Telluric Airmass ", tell_airmass)
        tell_respower = int(float((tell_hdr["respower"])))
        print("Telluric Resolution Power =", tell_respower)

        # wl_lower = np.min(wl)/1.0001
        # wl_upper = np.max(wl)*1.0001
        tell_data = wav_selector(tell_data[0], tell_data[1], wl_lower, wl_upper)

        # Telluric Normalization (use first 50 points below 1.2 as constant continuum)
        # For selected section
        I_tell = tell_data[1]
        maxes = I_tell[(I_tell < 1.2)].argsort()[-50:][::-1]
        tell_data = (tell_data[0], tell_data[1] / np.median(I_tell[maxes]))
        print("Telluric normaliztion value", np.median(I_tell[maxes]))


        (Corrections, Correction_tells, Correction_Bs,
            Correction_labels) = telluric_correct(wl, I, tell_data[0], tell_data[1], obs_airmass, tell_airmass,
                                                  kind=kind, method=method)

        if show:
            plt.figure()  # Tellurics
            plt.plot(wl, I, "--", linewidth=2, label="Observed Spectra")
            for corr, tell, B, label in zip(Corrections, Correction_tells, Correction_Bs, Correction_labels):
                # plt.plot(wl, corr, "--", label=(label + ", B = {0:.2f}".format(B)))
                plt.plot(wl, tell, linewidth=2, label=("Telluric " + label + ", B = {0:.3f}".format(B)))
                plt.plot(wl, np.ones_like(wl), "-.")
                plt.legend(loc="best")
                plt.title("Telluric Scaling with Tapas Resolution power = {}".format(tell_respower))

            plt.figure()  # Corrections
            plt.plot(wl, I, "--", linewidth=2, label="Observed Spectra")
            for corr, tell, B, label in zip(Corrections, Correction_tells, Correction_Bs, Correction_labels):
                plt.plot(wl, corr, linewidth=2, label=(label + ", B = {0:.3f}".format(B)))
                # plt.plot(wl, tell, label=("Telluric " + label + ", B = {0:.2f}".format(B)))
                plt.plot(wl, np.ones_like(wl), "-.")
                plt.legend(loc="best")
                plt.title("Telluric Corrections with tapas Resolution power = {}".format(tell_respower))

            plt.show()

        # B corr is almost not needed but include here for now 31/3/16 to make a correction
        print(Correction_labels)
        print(Corrections)
        I_corr = Corrections[1]  # using B scaling
        tell_used = Correction_tells[1]
        b_used = Correction_Bs[1]

    # ######################################   Ends  HERE ##########################################

    if mask:
        if h2o_scaling:
            telluric_rv_masks(wl, wl, tell_used, save_name=os.path.join(homedir, fname), h20=True)  # tell_used wavelength in this
            #                                                                               case is the same length due to
            #                                                                               interpolation.
        else:
            telluric_rv_masks(wl, wl, tell_used, save_name=os.path.join(homedir, fname))  # tell_used wavelength in this
        #                                                                               case is the same length due to
        #                                                                               interpolation.


    # ## SAVING Telluric Corrected Spectra ###
    # PROBABALY NEED TO HARDCODE IN THE HEADER LINES...
    os.chdir(homedir)   # to make sure saving where running

    # ## TO DO add mutually exclusive flag (with output) to add extra suffixs on end by .tellcorr.
    if output:
            output_filename = output
    else:
        if h2o_scaling:
            output_filename = fname.replace(".fits", ".h2otellcorr.fits")
        else:
            output_filename = fname.replace(".fits", ".tellcorr.fits")

    # Work out values for FITS header
    b_used = round(b_used, 4)

    if new_method & h2o_scaling:
            h2o_scale_val = out.params["scale_factor"].value
    else:
        h2o_scale_val = None

    # Keys and values for Fits header file
    hdrkeys = ["Correction", "Tapas Interpolation method",
               "Interpolation kind", "B PARAM",
               "H20 Scaling", "H20 Scaling Value", "Calculated R"]
    hdrvals = [("Tapas division", "Spectral Correction"),
               (method, "numpy or scipy"),
               (kind, "linear,slinear,quadratic,cubic"),
               (b_used, "Airmass scaling parameter"),
               (h2o_scaling, "Was separate H20 scaling 1 = Yes"),
               (h2o_scale_val, "H20 scale value used"),
               (R, "Observation resolution calculated by rule of thumb")]
    tellhdr = False   # need to correctly get this from obtain telluric


    if export:
        export_correction_2fits(output_filename, wl, I_corr, I, tell_used,
                                hdr, hdrkeys, hdrvals, tellhdr)
        print("Saved corected telluric spectra to " + str(output_filename))
    else:
        print("Skipped Saving corected telluric spectra ")


if __name__ == "__main__":
    args = vars(_parser())
    fname = args.pop('fname')
    debug_bool = args.pop('debug')
    opts = {k: args[k] for k in args}

    setup_debug(debug_bool)
    main(fname, **opts)

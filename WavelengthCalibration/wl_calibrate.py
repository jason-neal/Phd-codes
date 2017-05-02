#!/usr/bin/env python3

"""wl_calibrate.py

Script to run wavelength calibration on input crires fits file.

Uses telluric modeling from TAPAS.
"""
# from __future__ import division, print_function

import os
import time
import pickle
import logging
import argparse
import numpy as np
from astropy.io import fits
import GaussianFitting as gf
from PyAstronomy import pyasl
import matplotlib.pyplot as plt

import Obtain_Telluric as obt
# from plot_fits import get_wavelength
from SpectralTools import wav_selector
from TellRemoval import airmass_scaling
from Tapas_Berv_corr import tapas_helcorr
import XCorrWaveCalScript as XCorrWaveCal
from Gaussian_fit_testing import Get_DRACS


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
debug = logging.debug


def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Wavelength Calibrate CRIRES \
                                    Spectra')
    parser.add_argument('fname', help='Input fits file')
    parser.add_argument('-o', '--output', default=False,
                        help='Ouput Filename')
    parser.add_argument('-t', '--telluric', default=False,
                        help='Telluric line Calibrator')
    parser.add_argument('-m', '--model', default=False,
                        help='Stellar Model')
    parser.add_argument('-r', '--ref', default=False,
                        help='Reference Object with different RV')  # The other observation to identify shifted lines
    parser.add_argument('-b', '--berv_corr', default=False, action="store_true",
                        help='Apply Berv corr to plot limits if using berv corrected tapas')
    parser.add_argument('-u', '--use_rough', default=True, action="store_false",
                        help=" Get rough coordinates from stored pickle file, (if present).")

    args = parser.parse_args()
    return args


def get_wavelength(hdr):
    """Return the wavelength vector calculated from the header of a FITS
    file.

    :hdr: Header from a FITS ('CRVAL1', 'CDELT1', and 'NAXIS1' is required as
            keywords)
    :returns: Equidistant wavelength vector
    " From plot_fits.py"
    """
    w0, dw, n = hdr['CRVAL1'], hdr['CDELT1'], hdr['NAXIS1']
    w1 = w0 + dw * n
    return np.linspace(w0, w1, n, endpoint=False)


def export_wavecal_2fits(filename, wavelength, spectrum, pixelpos, hdr, hdrkeys, hdrvals):
    """ Write Combined DRACS CRIRES NOD Spectra to a fits table file"""
    col1 = fits.Column(name="Wavelength", format="E", array=wavelength)  # colums of data
    col2 = fits.Column(name="Extracted_DRACS", format="E", array=spectrum)
    col3 = fits.Column(name="Pixel", format="E", array=pixelpos)
    cols = fits.ColDefs([col1, col2, col3])
    tbhdu = fits.BinTableHDU.from_columns(cols)  # binary table hdu
    prihdr = append_hdr(hdr, hdrkeys, hdrvals)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    # print("Writing to fits file")
    try:
        thdulist.writeto(filename, output_verify="silentfix")   # Fixing errors to work properly
    except IOError:
        print("A calibtration already exists. What do you want to do?")
        ans = input(" o-Overwrite, a-append number")
        if ans.lower() == "o":
            os.rename(filename, filename + "_old")
            thdulist.writeto(filename, output_verify="silentfix")
        elif ans.lower() == "a":
            thdulist.writeto(filename + "_new", output_verify="silentfix")
        else:
            print("Did not append to name or overwrite fits file")
    return None


# could make new module for fits handlers like this
def append_hdr(hdr, keys, values, item=0):
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


def save_calibration_coords(filename, obs_pixels, obs_depths, obs_STDs, wl_vals, wl_depths, wl_STDs):
    with open(filename, "w") as f:
        f.write("# Pixels  \t Obs Depths \t Obs STD \t Wavelengths \t Tell depths \t Tell STD\n")
        for pixel, obs_d, obs_s, wl, wl_d, wl_s in zip(obs_pixels, obs_depths, obs_STDs, wl_vals, wl_depths, wl_STDs):
            f.write(("{:5.40f}\t{:01.40f}\t{:02.40f}\t{:5.40f}\t{:01.40f}\t{:02.40f}\n"
                     "").format(pixel, 1 - obs_d, obs_s, wl, 1 - wl_d, wl_s))
    return None


def convert_labels_to_rv(x, labels):
    """Convert ytick labels to rv.

    Relative to middle of wavelenght axis."""
    x_mid = np.mean(x)
    c = 299792.458   # km/s
    transform = ["{0:3.1f}".format((label / x_mid) * c) for label in labels]
    return transform


def main(fname, output=None, telluric=None, model=None, ref=None, berv_corr=False, use_rough=True):
    homedir = os.getcwd()
    print("Input name", fname)
    print("Output name", output)

    data = fits.getdata(fname)
    if data.shape[0] == 3:   # used extras
        debug(data.shape)
        data = data[0][0]
        debug(data.shape)
        debug(data)

    test0 = ".ms.norm.comb.fits" in fname  # from python combination
    test1 = ".ms.sum.norm.fits" in fname
    test2 = ".ms.Apos.norm.fits" in fname
    test3 = ".ms.Bpos.norm.fits" in fname
    test4 = ".ms.norm.sum.fits" in fname
    test5 = ".ms.norm.Apos.fits" in fname
    test6 = ".ms.norm.Bpos.fits" in fname
    if test0:
        uncalib_combined = np.array(data["Combined"], dtype="float64")
        # uncalib_noda = uncalib_data["Nod A"]
        # uncalib_nodb = uncalib_data["Nod B"]
    elif test1 or test2 or test3 or test4 or test5 or test6:
        uncalib_combined = np.array(data, dtype="float64")
    else:
        print("Unrecgonized input filename. Can take ouput from sumnormnodcycle8jn.cl, "
              "normalizeobsrsum.cl or Combine_nod_spectra.py")
        raise("Spectra_Error", "Unrecgonized input filename type")

    # uncalib_data = [range(1, len(uncalib_combined) + 1), uncalib_combined]
    uncalib_data = [np.arange(len(uncalib_combined)) + 1, uncalib_combined]

    # Get time from header to then get telluric lines
    hdr = fits.getheader(fname)
    wl_lower = hdr["HIERARCH ESO INS WLEN STRT"]
    wl_upper = hdr["HIERARCH ESO INS WLEN END"]
    datetime = hdr["DATE-OBS"]

    obsdate, obstime = datetime.split("T")
    obstime, __ = obstime.split(".")

    if telluric:  # manually spcified telluric line
        tellpath = os.getcwd() + "/"
        tell_data, tell_header = obt.load_telluric(tellpath, telluric)
    else:
        raise("Please specify the telluric line model to calibrate against.")
    #    tellpath = "/home/jneal/Phd/data/Tapas/"
    #    tellname = obt.get_telluric_name(tellpath, obsdate, obstime) # to within the hour
    #    tell_data, tell_header = obt.load_telluric(tellpath, tellname[0])

    print("obs data 0 type", type(uncalib_data[0]), "dtype", uncalib_data[0].dtype)
    print("obs data 1 type", type(uncalib_data[1]), "dtype", uncalib_data[1].dtype)
    print("telluric type", type(tell_data[1]), "dtype", tell_data[0].dtype, tell_data[1].dtype)

    # Scale telluric lines to airmass
    # ###### this needs t be corrected to middle of hole obs. Not just first observation
    obs_airmass = (hdr["HIERARCH ESO TEL AIRM START"] + hdr["HIERARCH ESO TEL AIRM END"]) / 2

    tell_airmass = float(tell_header["airmass"])

    tell_data[1] = airmass_scaling(tell_data[1], tell_airmass, obs_airmass)

    if tell_header["barydone"] == "YES" and berv_corr:
        # Only if berv has been done on tapas and the user asks for it
        # ## BERV adjust from tapas the wl_limits to align the plotting
        old_wl_lower = wl_lower
        old_wl_upper = wl_upper
        tapas_berv_value = tapas_helcorr(tell_header)
        # Doppler shift the detector limits
        __, wlprime = pyasl.dopplerShift(np.array([wl_lower, wl_upper]), np.array([1, 1]),
                                         tapas_berv_value[0], edgeHandling=None, fillValue=None)
        wl_lower = wlprime[0]
        wl_upper = wlprime[1]
        # print("Old detector limits", [old_wl_lower, old_wl_upper])
        print("New Berv shifted detector limits", [wl_lower, wl_upper])
    elif berv_corr:
        print("Berv_corr flag given but tapas data was not berv corrected. Not adjusting limits")

    # ## Air wavelengths
    # Convert limits if using air wavelengths
    if tell_header["WAVSCALE"] == "air":
        logging.warning("Using Air Wavelengths")
        # vac2air on the crires limits
        # print("Using AIR wavelength scale so changing wl limits")
        wl_lower_vac = wl_lower
        wl_upper_vac = wl_upper
        # The other modes don't work above 1.69 micron
        wl_lower = pyasl.vactoair2(wl_lower, mode="edlen53")
        wl_upper = pyasl.vactoair2(wl_upper, mode="edlen53")
        # print("Vacuum detector limits", [wl_lower_vac, wl_upper_vac])
        # print("New berv shifted detector limits", [wl_lower, wl_upper])


    # Sliced to wavelength measurement of detector
    # calib_data = gf.slice_spectra(tell_data[0], tell_data[1], wl_lower, wl_upper)
    calib_data = wav_selector(tell_data[0], tell_data[1], wl_lower, wl_upper)

    gf.print_fit_instructions()  # Instructions on how to calibrate
    if ref:  # Reference object spectra to possibly identify shifted/blended lines
        I_ref = fits.getdata(ref)
        w_ref = range(1, 1025)
        maxes = I_ref[(I_ref < 1.2)].argsort()[-50:][::-1]
        I_ref /= np.median(I_ref[maxes])

    if model:
        modelpath = "/home/jneal/Phd/data/phoenixmodels/"
        modelwave = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

        I_mod = fits.getdata(model)
        hdr = fits.getheader(model)
        if 'WAVE' in hdr.keys():
            w_mod = fits.getdata(modelpath + modelwave)
            w_mod = w_mod / 10.
        else:
            w_mod = get_wavelength(hdr)
        # nre = nrefrac(w_mod)  # Correction for vacuum to air (ground based)
        # w_mod = w_mod / (1 + 1e-6 * nre)

        i = (w_mod > wl_lower) & (w_mod < wl_upper)
        w_mod = w_mod[i]
        I_mod = I_mod[i]

        if len(w_mod) > 0:
            # https://phoenix.ens-lyon.fr/Grids/FORMAT
            # I_mod = 10 ** (I_mod-8.0)
            I_mod /= np.median(I_mod)
            # Normalization (use first 50 points below 1.2 as continuum)
            maxes = I_mod[(I_mod < 1.2)].argsort()[-50:][::-1]
            I_mod /= np.median(I_mod[maxes])
            # if ccf in ['model', 'both'] and rv1:
            #    print('Warning: RV set for model. Calculate RV with CCF')
            # if rv1 and ccf not in ['model', 'both']:
            #    I_mod, w_mod = dopplerShift(wvl=w_mod, flux=I_mod, v=rv1, fill_value=0.95)
        else:
            print('Warning: Model spectrum not available in wavelength range.')
            model = False

    rough_coord_name = fname.split(".fits")[0] + "_rough_cords.pickle"
    try:
        if use_rough:
            rough_a, rough_b = pickle.load(open(rough_coord_name, "rb"))
        else:
            raise
    except:
        rough_a, rough_b = gf.get_rough_peaks(uncalib_data[0], uncalib_data[1], calib_data[0], calib_data[1])
        pickle.dump((rough_a, rough_b), open(rough_coord_name, "wb"))

    rough_x_a = [coord[0] for coord in rough_a]
    rough_x_b = [coord[0] for coord in rough_b]
    if model:
        fit_results = gf.adv_wavelength_fitting(uncalib_data[0], uncalib_data[1],
                                                rough_x_a, calib_data[0], calib_data[1],
                                                rough_x_b, model=[w_mod, I_mod])
    elif ref:
        fit_results = gf.adv_wavelength_fitting(uncalib_data[0], uncalib_data[1],
                                                rough_x_a, calib_data[0], calib_data[1],
                                                rough_x_b, ref=[w_ref, I_ref])
    else:
        fit_results = gf.adv_wavelength_fitting(uncalib_data[0], uncalib_data[1],
                                                rough_x_a, calib_data[0], calib_data[1],
                                                rough_x_b)
    good_a, peaks_a, std_a, good_b, peaks_b, std_b = fit_results

    lin_map = gf.wavelength_mapping(good_a, good_b, order=1)
    wl_map = gf.wavelength_mapping(good_a, good_b, order=2)
    cube_map = gf.wavelength_mapping(good_a, good_b, order=3)
    # quartic_map = gf.wavelength_mapping(good_a, good_b, order=4) # 4th order is not good

    lin_calibrated_wl = np.polyval(lin_map, uncalib_data[0])
    calibrated_wl = np.polyval(wl_map, uncalib_data[0])
    cube_calibrated_wl = np.polyval(cube_map, uncalib_data[0])
    # quartic_calibrated_wl = np.polyval(quartic_map, uncalib_data[0])

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(calibrated_wl, uncalib_data[1], label="Quad Calibrated spectra")
    plt.plot(cube_calibrated_wl, uncalib_data[1], label="Cube Calibrated spectra")
    # plt.plot(quartic_calibrated_wl, uncalib_data[1], label="Quartic Calibrated spectra")
    plt.plot(calib_data[0], calib_data[1], label="Telluric spectra")
    plt.title("Wavelength Calibrated Spectra")
    # Stopping scientific notation offset in wavelength
    ax1 = plt.gca()
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.legend()

    ax2 = plt.subplot(2, 1, 2)
    plt.plot(uncalib_data[0], calibrated_wl - lin_calibrated_wl, "+", label="Quad Calibrated spectra")
    plt.plot(uncalib_data[0], cube_calibrated_wl - lin_calibrated_wl, "x", label="Cube Calibrated spectra")
    # plt.plot(uncalib_data[0], quartic_calibrated_wl - lin_calibrated_wl, "o", label="Quartic Calibrated spectra")
    # ax2 = plt.gca()
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    # RV Scale
    ax3 = ax2.twinx()
    ax3.plot(uncalib_data[0], calibrated_wl - lin_calibrated_wl, "+", label="Quad Calibrated spectra")
    ax3.plot(uncalib_data[0], cube_calibrated_wl - lin_calibrated_wl, "x", label="Cube Calibrated spectra")
    tick_labels = [item for item in ax3.get_yticks()]
    new_ticklabels = convert_labels_to_rv(calib_data[0], tick_labels)
    ax3.set_yticklabels(new_ticklabels)
    ax3.set_ylabel("Relative RV (km/s)")

    plt.xlabel("Pixel number")
    ax2.set_ylabel("Delta lambda(nm)\nfrom Linear Fit")
    plt.title("Wavelength Differences of Models")
    plt.legend()
    plt.show(block=False)

    # Save output now
    # Do you want to fine turn this calibration?
    ans = input("Do you want to finetune the calibtration?\n")
    if ans in ['yes', 'y', 'Yes', 'YES']:
        print("\n\nFinetune with XCORR WAVECAL using this result as the guess wavelength\n")
        Finetuned_wl, finetuned_params = XCorrWaveCal.wl_xcorr((calibrated_wl, uncalib_data[1]),
                                                               (tell_data[0], tell_data[1]), increment=0.1)
        fig = plt.figure()
        plt.plot(calibrated_wl, uncalib_data[1], label="Calibrated spectra")
        plt.plot(calib_data[0], calib_data[1], label="Telluric spectra")
        plt.plot(Finetuned_wl, uncalib_data[1], label="Finetuned Wl spectra")
        plt.title("Wavelength Calibrated Output with Finetuneing")
        # Stopping scientific notation offset in wavelength
        ax = plt.gca()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Intensity")
        plt.legend()
        plt.show(fig, block=True)

        print("Warning - at this stage the fine tuning does not get saved.")
    # This to possible tune,  sample_num, ratio, increment
    else:
        print("Did not fine tune calibration with Xcorr")
    # Do you want to save this output?
    ans = input("Do you want to save the calibration?\n")
    if ans in ['yes', 'y', 'Yes', 'YES']:
        os.chdir(homedir)   # to make sure saving where running
        if output:
            Output_filename = output
        else:
            Output_filename = fname.replace(".fits", ".wavecal.fits")

        T_now = str(time.gmtime()[0:6])

        hdrkeys = ["Calibration", "CALIB TIME", "Tapas ID number",
                   "Number Fitted", 'PIXELMAP PARAM1', "PIXELMAP PARAM2",
                   "PIXELMAP PARAM3", "Tapas Barycenter Correction",
                   "Tapas wavelength scale"]
        hdrvals = ["DRACS Wavelength Calibration with Tapas spectrum",
                   (T_now, "Time of Calibration"),
                   (tell_header["run_id"], "Tapas unique ID number"),
                   (len(good_a), "Number of points in calibration map"),
                   (wl_map[0], "Squared term"), (wl_map[1], "Linear term"),
                   (wl_map[2], "Constant term"),
                   (tell_header["barydone"], "Barycenter correction done by Tapas"),
                   (tell_header["WAVSCALE"], "Either air, vacuum, or wavenumber")]
        # # ADD OTHER parameter need to store above - estimated errors of fitting?

        export_wavecal_2fits(Output_filename, calibrated_wl, uncalib_data[1], uncalib_data[0], hdr, hdrkeys, hdrvals)

        # Save calibration values to a txt file
        coord_txt_fname = "Coordinates_" + fname[:-5] + ".txt"

        save_calibration_coords(coord_txt_fname, good_a, std_a, peaks_a, good_b, peaks_b, std_a)
        # save_calibration_coords(filename, obs_pixels, obs_depths, obs_STDs, wl_vals, wl_depths, wl_STDs)
        print("Succesfully saved calibration to file -".format(Output_filename))
    else:
        print("Did not save calibration to file.")


    ans = input("Do you want to observe the line depths?\n")
    if ans in ['yes', 'y', 'Yes', 'YES']:
        # observe heights of fitted peaks
        plt.figure
        plt.plot(peaks_a, label="Specta line depths")
        plt.plot(peaks_b, label="Telluric line depths")
        plt.show(block=True)

    linedepthpath = "/home/jneal/Phd/data/Crires/BDs-DRACS/"
    ans = input("Do you want to export the line depths to a file?\n")
    if ans in ['yes', 'y', 'Yes', 'YES']:
        with open(linedepthpath + "New_Spectral_linedepths.txt", "a") as f:
            for peak in peaks_a:
                print(peak)
                f.write(str(peak) + "\n")
        with open(linedepthpath + "New_Telluric_linedepths.txt", "a") as f:
            for peak in peaks_b:
                f.write(str(peak) + "\n")
        print("Saved line depths to New_xxxx_linedepths.txt in {}".format(linedepthpath))


if __name__ == '__main__':
    args = vars(_parser())
    fname = args.pop('fname')
    opts = {k: args[k] for k in args}

    main(fname, **opts)

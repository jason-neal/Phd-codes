#!/usr/bin/env python

""" Script to run wavelength calibration on input fits file"""
from __future__ import division, print_function

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from gooey import Gooey, GooeyParser
import IOmodule
import GaussianFitting as gf
from Gaussian_fit_testing import Get_DRACS
import Obtain_Telluric as obt

#@Gooey(program_name='Plot fits - Easy 1D fits plotting', default_size=(610, 730))
def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    #parser = GooeyParser(description='Wavelength Calibrate CRIRES Spectra')
    parser = argparse.ArgumentParser(description='Wavelength Calibrate CRIRES \
                                    Spectra')
    parser.add_argument('fname', help='Input fits file')
    parser.add_argument('-o', '--output', default=False,
                        help='Ouput Filename',)
    parser.add_argument('-t', '--telluric', default=False,
                       help='Telluric line Calibrator')
    #parser = GooeyParser(description='Wavelength Calibrate CRIRES Spectra')
    #parser.add_argument('fname',
    #                    action='store',
    #                    widget='FileChooser',
    #                    help='Input fits file')
    #parser.add_argument('-o', '--output',
    #                    default=False,
    #                    action='store',
    #                    widget='FileChooser',
    #                    help='Ouput Filename',)
    #parser.add_argument('-t', '--telluric',
    #                    default=False,
    #                    action='store',
    #                    widget='FileChooser',
    #                    help='Telluric line Calibrator',)
    
    args = parser.parse_args()
    return args


def export_wavecal_2fits(filename, wavelength, spectrum, pixelpos, hdr, hdrkeys, hdrvals):
    """ Write Combined DRACS CRIRES NOD Spectra to a fits table file"""
    col1 = fits.Column(name="Wavelength", format="E", array=wavelength) # colums of data
    col2 = fits.Column(name="Extracted_DRACS", format="E", array=spectrum)
    col3 = fits.Column(name="Pixel", format="E", array=pixelpos)
    cols = fits.ColDefs([col1, col2, col3])
    tbhdu = fits.BinTableHDU.from_columns(cols) # binary tbale hdu
    prihdr = append_hdr(hdr, hdrkeys, hdrvals)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
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


def main(fname, output=False, telluric=False):
    homedir = os.getcwd()
    print("Input name", fname)
    print("Output name", output)

    #tellpath = "/home/jneal/Phd/Codes/Phd-codes/WavelengthCalibration/testfiles/"  # Updated for git repo
    tellpath = "/home/jneal/Phd/data/Tapas/"
    #ext = 0   # for testing
    #hdr, data = Get_DRACS(fname, ext)  #Get DRACS only finds relevant file in the path folder
    
    data = fits.getdata(fname)

    # get time from header to then get telluric lines
    hdr = fits.getheader(fname)
    wl_lower = hdr["HIERARCH ESO INS WLEN STRT"]
    wl_upper = hdr["HIERARCH ESO INS WLEN END"]
    datetime = hdr["DATE-OBS"]
    print("Observation time ", datetime)
    obsdate, obstime = datetime.split("T")
    
    obstime, __ = obstime.split(".")
    print(" time ", obstime)
    print(" date ", obsdate)
    tellname = obt.get_telluric_name(tellpath, obsdate, obstime) # to within the hour
    print("tell name", tellname)
    print("listing telluric files")
    
    uncalib_combined = data["Combined"]
    #uncalib_noda = uncalib_data["Nod A"]
    #uncalib_nodb = uncalib_data["Nod B"]
    uncalib_data = [range(1, len(uncalib_combined) + 1), uncalib_combined]
    # get hdr information and then find coresponding Telluric spectra
    #calib_data = IOmodule.read_2col(tellpath + "Telluric_spectra_CRIRES_Chip-" + 
    #                                str(1) + ".txt")

    # telluric spectra is way to long, need to reduce it to similar size as ccd    
    tell_data = obt.load_telluric(tellpath, tellname[0])
    # Sliced to wavelength measurement of detector
    calib_data = gf.slice_spectra(tell_data[0], tell_data[1], wl_lower, wl_upper)


    gf.print_fit_instructions()

    rough_a, rough_b = gf.get_rough_peaks(uncalib_data[0], uncalib_data[1], calib_data[0], calib_data[1])
    rough_x_a = [coord[0] for coord in rough_a]
    rough_x_b = [coord[0] for coord in rough_b]
    good_a, good_b = gf.adv_wavelength_fitting(uncalib_data[0], uncalib_data[1], 
                                       rough_x_a, calib_data[0], calib_data[1],
                                       rough_x_b)

    wl_map = gf.wavelength_mapping(good_a, good_b)

    calibrated_wl = np.polyval(wl_map, uncalib_data[0])
    plt.figure()
    plt.plot(calibrated_wl, uncalib_data[1], label="Calibrated spectra")
    plt.plot(calib_data[0], calib_data[1], label="Telluric spectra")
    plt.title("Calibrated Output")
    plt.show(block=True)


    # Save output now
    #Do you want to fine turn this calibration?
    ans = raw_input("Do you want to finetune the calibtration?")
    if ans in ['yes', 'y', 'Yes', 'YES']:
            print("Are not able to fine tune at this time.")
            #gf.get_rough_peaks()
            #adv_wavelength_fitting()
            # This will leave 2 conversion equations that prob should refit
 # numpy.polyval(p, x)[source]
 #    Evaluate a polynomial at specific values.
 #    If p is of length N, this function returns the value:
 #        p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
 #    If x is a sequence, then p(x) is returned for each element of x. If x is another polynomial then the composite polynomial p(x(t)) is returned.

    #Do you want to save this output?
    ans = raw_input("Do you want to save this calibration?")
    if ans in ['yes', 'y', 'Yes', 'YES']:
        os.chdir(homedir)   # to make sure saving where running
        if output:  
            Output_filename = output
        else:
            Output_filename = fname.replace(".fits", ".wavecal.fits")
        
            

        T_now = str(time.gmtime()[0:6])
        hdrkeys = ["Calibration", "CALIBRATION TIME", 'PIXELMAP PARAM1', \
                  "PIXELMAP PARAM2", "PIXELMAP PARAM3"]
        hdrvals = ["DRACS Wavelength Calibration with Tapas spectrum", \
                   (T_now, "Time of Calibration"), (wl_map[0], "Squared term"),\
                   (wl_map[1], "Linear term"), (wl_map[2], "Constant term")]
                ###### ADD OTHER parameter need to store above - estimated errors of fitting?
        export_wavecal_2fits(Output_filename, calibrated_wl, uncalib_data[1], uncalib_data[0], hdr, hdrkeys, hdrvals)
    else:
        print("Did not save calibration to file.")
    

if __name__ == '__main__':
    args = vars(_parser())
    fname = args.pop('fname')
    opts = {k: args[k] for k in args}

    main(fname, **opts)
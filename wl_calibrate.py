#

""" Script to run wavelength calibration on input fits file"""
from __future__ import division, print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from gooey import Gooey, GooeyParser
import IOmodule
import GaussianFitting as gf
from Gaussian_fit_testing import Get_DRACS


@Gooey(program_name='Plot fits - Easy 1D fits plotting', default_size=(610, 730))
def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = GooeyParser(description='Wavelength Calibrate CRIRES Spectra')
    parser.add_argument('fname',
                        action='store',
                        widget='FileChooser',
                        help='Input fits file')
    parser.add_argument('-o', '--output',
                        default=False,
                        action='store',
                        widget='FileChooser',
                        help='Ouput Filename',)
    parser.add_argument('-t', '--telluric',
                        default=False,
                        action='store',
                        widget='FileChooser',
                        help='Telluric line Calibrator',)
    #parser.add_argument('-t', '--telluric',
    #                    help='Over plot telluric spectrum',
    #                    action='store_true')
    
    #parser.add_argument('-c', '--ccf',
    #                   default='none',
    #                    choices=['none', 'sun', 'model', 'telluric', 'both'],
    #                    help='Calculate the CCF for Sun/model or tellurics '
    #                    'or both.')
    #parser.add_argument('--ftype', help='Select which type the fits file is',
    #                    choices=['ARES', 'CRIRES'], default='ARES')
    #parser.add_argument('--fitsext', help='Select fits extention, Default 0.',
    #                    choices=['0', '1', '2', '3', '4'], default='0')
    #parser.add_argument('--fitsext', default=0, type=int, 
    #                    help='Select fits extention, 0 = Primary header')
    args = parser.parse_args()
    return args

def main(fname, output=None):
    print("Input name", fname)
    print("Output name", output)

    tellpath = "/home/jneal/Phd/Codes/Phd-codes/WavelengthCalibration/testfiles/"  # Updated for git repo
   
    #ext = 0   # for testing
    #hdr, data = Get_DRACS(fname, ext)  #Get DRACS only finds relevant file in the path folder
    hdr = fits.getheader(fname)
    print("Observation time ", hrd[])
    data = fits.getdata(fname)
    
    uncalib_combined = data["Combined"]
    #uncalib_noda = uncalib_data["Nod A"]
    #uncalib_nodb = uncalib_data["Nod B"]
    uncalib_data = [range(1,len(uncalib_combined)+1), uncalib_combined]
    # get hdr information and then find coresponding Telluric spectra
    calib_data = IOmodule.read_2col(tellpath + "Telluric_spectra_CRIRES_Chip-" + 
                                    str(1) + ".txt")
    
    gf.print_fit_instructions()

    rough_a, rough_b = gf.get_rough_peaks(uncalib_data[0], uncalib_data[1], calib_data[0], calib_data[1])
    rough_x_a = [coord[0] for coord in rough_a]
    rough_x_b = [coord[0] for coord in rough_b]
    good_a, good_b = gf.adv_wavelength_fitting(uncalib_data[0], uncalib_data[1], 
                                       rough_x_a, calib_data[0], calib_data[1],
                                       rough_x_b)

    wl_map = gf.wavelength_mapping(good_a, good_b)

    calibrated_wl = np.polyval(wl_map, uncalib_data[0])
    
    plt.plot(calibrated_wl, uncalib_data[1], label="Calibrated spectra")
    plt.plot(calib_data[0], calib_data[1], label="Telluric spectra")
    plt.title("Calibration Output")
    plt.show()

    # Save output now


if __name__ == '__main__':
    args = vars(_parser())
    fname = args.pop('fname')
    opts = {k: args[k] for k in args}

    main(fname, **opts)
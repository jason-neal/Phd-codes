#!/usr/bin/env python3

""" Wavelength Calibration Check
Simple wavelength plotting to check results the result of wl_calibrate.py

Run this script in the file that contains the CRIRES*.sum.wavecal.fits files and the tapas file used to calibrate them with.
The script will quickly find the files automatically and plot them together.

TODO - Add a argparse flag/fname to save the plot?
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import Obtain_Telluric as obt
from Get_filenames import get_filenames

cwd = os.getcwd() + "/"

# get files
crires_files = get_filenames(cwd, "CRIRE.*", "*norm.sum.wavecal.fits*")

tapas_file = get_filenames(cwd, "tapas_*", "*_ReqId_10_R*")

# Extract tapas data
tapas_data, tapas_hdr = obt.load_telluric(cwd, tapas_file[0])

# start figure
plt.figure()
plt.plot(tapas_data[0], tapas_data[1], label=" Tapas")

for crires_name in crires_files:
    # extract Crires data
    Obs_data  = fits.getdata(crires_name)
    # Obs_hdr = fits.getheader(crires_name)  # Don't need header atm
    label = "Detector " + crires_name[30]   # Chip # from the filename
    plt.plot(Obs_data["Wavelength"], Obs_data["Extracted_DRACS"], label=label)

plt.legend(loc=0)  # best legend location
plt.xlabel("Wavelength (nm)")
plt.ylabel("Flux/Transmittance")
plt.title("Wavelength Calibration check for {0}".format(cwd.split("/")[7]))
plt.show()

#!/usr/bin/env python3

# Tapas Species
# Test Joining a Separated Tapas sectra
# Want to plot all 4 chips to see what liens are present.
# Not many lines of 02 to calibrate against. only in chip 1 mainly.

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import WavelengthCalibration.GaussianFitting as gf
import TelluricSpectra.Obtain_Telluric as obt
from TelluricSpectra.TellRemoval import airmass_scaling

tapas_path = "/home/jason/Phd/data/Tapas/Tapas_march2016/HD30501_1_separated/"

filenames = ["tapas_00000{0}.ipac".format(i + 1) for i in range(6)]

#cr_name = r"/home/jason/MyCodes/Phd-codes/TelluricSpectra/CRIRE.2012-04-07T00%3A08%3A29.976_1.nod.ms.sum.norm.wavecal.fits"
cr_name = r"/home/jason/Phd/Codes/Phd-codes/TelluricSpectra/CRIRE.2012-04-07T00%3A08%3A29.976_1.nod.ms.sum.norm.wavecal.fits"

cr_data = fits.getdata(cr_name)
wl = cr_data["Wavelength"]
I = cr_data["Extracted_DRACS"]

print(filenames)
ax = plt.subplot(111)
ax.plot(wl, I, "k", label="Observation")
for name in filenames:
    data, hdr = obt.load_telluric(tapas_path, name)
    res = float(hdr["resPower"])
    sampRati = hdr["sampRati"]
    airmass = hdr["airmass"]
    ax.plot(data[0], data[1])

# ax2 = plt.twiny(ax)
ax.legend(["Observation", "H2O", "O3", "O2", "CO2", "CH4", "N2O"], loc=0)
plt.show()


ax = plt.subplot(111)
data, hdr = obt.load_telluric(tapas_path, filenames[0])
wavelength = data[0]
Combined_spectra = data[1]
ax.plot(data[0], data[1])
for name in filenames[1:]:
    data, hdr = obt.load_telluric(tapas_path, name)
    ax.plot(data[0], data[1])
    Combined_spectra *= data[1]

ax.plot(wavelength, Combined_spectra)
ax.legend(["H2O", "O3", "O2", "CO2", "CH4", "N2O", "Combined"], loc=0)
plt.show()

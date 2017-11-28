#!/usr/lib/python3
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import WavelengthCalibration.GaussianFitting as gf
import TelluricSpectra.Obtain_Telluric as obt
from TelluricSpectra.TellRemoval import airmass_scaling

ESO_path = "/home/jneal/Phd/data/ESO-Skydata/"
tapas_path = "/home/jneal/Phd/data/tapas-testing/"

filenames = ["HD30501-1-R50000-gaussianconvolution-2FWHM.dat", \
            "HD30501-1-R50000-gaussianconvolution-5FWHM.dat", \
            "HD30501-1-R50000-gaussianconvolution-8FWHM.dat", \
            "HD30501-1-R50000-noconvolution-.dat", \
            "HD30501-1-R100000-gaussianconvolution-5FWHM.dat", \
            "HD30501-1-R100000-noconvolution-.dat"]

Fits_skytables = ["HD30501-1-100000-no-convolution.fits", \
                    "HD30501-1-50000-no-convolution.fits"]


# See what is in the Skytables
plt.figure()
for name in Fits_skytables:
    res = name.split("-")[2]
    data = fits.getdata(ESO_path + name)
    wl = data["lam"]
    Tr = data["trans"]

    plt.plot(wl, Tr, label=res)

plt.legend()
plt.xlim([2117, 2121])
# plt.show()


no_conv_filenames = ["HD30501-1-R50000-noconvolution-.dat", \
                    "HD30501-1-R100000-noconvolution-.dat"]
plt.figure()
for name in no_conv_filenames:
    wl, Tr = np.loadtxt(ESO_path + name, unpack=True)
    res = name.split("-")[2]
    plt.plot(wl, Tr, label=res)

plt.legend()
plt.title("Resolution Change Effect with No Convolution")
# plt.show()

R5_filenames = ["HD30501-1-R50000-noconvolution-.dat", \
            "HD30501-1-R50000-gaussianconvolution-2FWHM.dat", \
            "HD30501-1-R50000-gaussianconvolution-5FWHM.dat", \
            "HD30501-1-R50000-gaussianconvolution-8FWHM.dat"]
plt.figure()
for name in R5_filenames:
    wl, Tr = np.loadtxt(ESO_path + name, unpack=True)
    split = name.split("-")
    res = split[2]
    conv = split[3]

    if conv[0:8] == "gaussian":
        print(conv)
        fwhm = split[-1].split(".")[0]
    else:
        fwhm = ""
    plt.plot(wl, Tr, label=res + " " + conv + " " + fwhm)

plt.legend()
plt.xlim([2117, 2121])
plt.title("50000 Resolution Effect of Line Profile")
# plt.show()


R10_filenames = ["HD30501-1-R100000-gaussianconvolution-5FWHM.dat", \
            "HD30501-1-R100000-noconvolution-.dat"]
plt.figure()
for name in R10_filenames:
    wl, Tr = np.loadtxt(ESO_path + name, unpack=True)
    split = name.split("-")
    res = split[2]
    conv = split[3]
    if conv[0:8] == "gaussian":
        print(conv)
        fwhm = split[-1].split(".")[0]
    else:
        fwhm = ""

    plt.plot(wl, Tr, label=res + " " + conv + " " + fwhm)

plt.legend()
plt.xlim([2117, 2121])
plt.title("R=100000 Convolution Effect")
# plt.show()


#####  Tapas at R 50000
# path = "/home/jneal/Phd/data/tapas-testing/"

tapas1_filenames = ["tapas_2012-04-07T01:20:20-HD30501-1-R50000-sample-10.ipac", \
            "tapas_2012-04-07T01:20:20-HD30501-1-R50000-sample-5.ipac", \
            "tapas_2012-04-07T01:20:20-HD30501-1-R50000-sample-15.ipac"]
plt.figure()
for name in tapas1_filenames:
    data, hdr = obt.load_telluric(tapas_path, name)

    res = float(hdr["resPower"])
    sampRati = hdr["sampRati"]
    airmass = hdr["airmass"]
    plt.plot(data[0], data[1], label="R =" + str(int(res)) + ", sampling =" + sampRati + " airmass=" + str(airmass))

    # plt.plot(wl, Tr, label=res + " " + conv + " " + fwhm)
plt.legend()
plt.xlim([2117, 2121])
plt.title("Tapas R=50000 sampling Effect")
# plt.show()


tapas2_filenames = ["tapas_test2_1.ipac", \
            "tapas_test2_2.ipac", \
            "tapas_test2_3.ipac", \
            "tapas_test2_4.ipac"]
plt.figure()
for name in tapas2_filenames:
    data, hdr2 = obt.load_telluric(tapas_path, name)
    for key, val in hdr2.iteritems():
        print("Key", key, "Value", val)

    try:
        res = hdr2["RESPOWER"]
        res = float(res)
        sampRati = hdr2["sampRati"]
        airmass = hdr2["airmass"]
    except:
        print("No RESPOWER for", name)
        res = 0
        sampRati = 0
        airmass = hdr2["airmass"]

    plt.plot(data[0], data[1], label="R =" + str(int(res)) + ", sampling =" + str(sampRati) + " airmass=" + str(airmass))
    # plt.plot(wl, Tr, label=res + " " + conv + " " + fwhm)
plt.legend(loc="best")
plt.xlim([2117, 2121])
plt.title("Tapas Instrument Effect")
# plt.show()


# Effect of 1 hour time increments on tapas spectra
# (to see the effect of the 1hr timing difference given by tapas)
plt.figure()
tapas3_filenames = ["tapas_test3_timing_-3.ipac", \
            "tapas_-2timing.ipac", \
            "tapas_test3_timing_-2.ipac", \
            "tapas_test3_timing_-1.ipac", \
            "tapas_test3_timing_0.ipac", \
            "tapas_test3_timing_+1.ipac", \
            "tapas_test3_timing_+2.ipac"]
            # "tapas_test3_timing_+3.ipac"

for name in tapas3_filenames:
    data, hdr = obt.load_telluric(tapas_path, name)
    # wl, Tr = np.loadtxt(tapas_path + name, unpack=True)

    res = float(hdr["resPower"])
    time = hdr["date-obs"]
    sampRati = hdr["sampRati"]
    airmass = hdr["airmass"]

    plt.plot(data[0], data[1], label= time + " airmass=" + str(airmass))
    # plt.plot(wl, Tr, label=res + " " + conv + " " + fwhm)
plt.legend(loc="best")
plt.title("Tapas Timing Effects \n R=50000, Sample Ratio=10\nObs Time=00:20:20")
plt.xlim([2117, 2121])
# plt.show()


### Scale to AIRMASS of actual time 0:20:20
# Effect of 1 hour time increments on tapas spectra
# (to see the effect of the 1hr timing difference given by tapas)
plt.figure()
tapas3_filenames = ["tapas_test3_timing_-3.ipac", \
            "tapas_-2timing.ipac", \
            "tapas_test3_timing_-2.ipac", \
            "tapas_test3_timing_-1.ipac", \
            "tapas_test3_timing_0.ipac", \
            "tapas_test3_timing_+1.ipac", \
            "tapas_test3_timing_+2.ipac"]
            # "tapas_test3_timing_+3.ipac"

for name in tapas3_filenames:
    data, hdr = obt.load_telluric(tapas_path, name)
    # wl, Tr = np.loadtxt(tapas_path+name, unpack=True)

    res = float(hdr["resPower"])
    time = hdr["date-obs"]
    sampRati = hdr["sampRati"]
    airmass = hdr["airmass"]

    # Airmass Scaling
    scaled_data = airmass_scaling(data[1], float(airmass), 1.6)
    # plt.plot(data[0], data[1], label= time + " airmass=" + str(airmass))
    plt.plot(data[0], scaled_data, label= time + " airmass=" + str(airmass) + " scalled=" + str(1.6))

    # plt.plot(wl, Tr, label=res + " " + conv + " " + fwhm)
plt.legend(loc="best")
plt.title("Tapas Timing Effects\nScalled to Airmass 1.6\nR=50000, Sample Ratio = 10\nObs Time=00:20:20")
plt.xlim([2117, 2121])
# plt.show()


# Effect BEERV Correction increments on tapas spectra

# BERV is the Acronym of Barycentric Earth Radial Velocity.
# It must be realized that the data reduction pipeline may use
# two different systems for the wavelength calibration of the spectrum.
# One is an absolute wavelength calibration system, based on
# recording of some Thorium Argon spectral lines delivered by a
# dedicated lamp. With such a system, there is no need to apply
# the BERV correction described below. Actually, the positions
# of observed narrow atmospheric lines (O2 or H2O) may be
# compared to the output of TAPAS, and may provide an excellent
# wavelength calibration of the observed spectrum, since it is
# taken simultaneously with the target spectrum.

tapas4_filenames = ["tapas_test3_timing_0_no_berv_corr.ipac", \
            "tapas_test3_timing_0.ipac"]
plt.figure()
for name in tapas4_filenames:
    data, hdr = obt.load_telluric(tapas_path, name)

    res = float(hdr["resPower"])
    sampRati = hdr["sampRati"]
    airmass = hdr["airmass"]
    barydone = hdr["barydone"]

    plt.plot(data[0], data[1], label= "barydone " + barydone + " airmass=" + str(airmass))
plt.legend(loc="best")
plt.title("Tapas Effect of Barycentric Correction (Normally on)\nR=50000, Sample Ratio = 10" )
plt.xlim([2117, 2121])
# plt.show()




##################################################################### 3

## Compare ESO Sky Calc and TAPAS
comp_filenames = ["tapas_test3_timing_-1.ipac", \
        "HD30501-1-R50000-gaussianconvolution-2FWHM.dat", \
        "HD30501-1-R50000-gaussianconvolution-5FWHM.dat", \
        "HD30501-1-R50000-gaussianconvolution-8FWHM.dat"]
plt.figure()
for name in comp_filenames:
    if name.split("_")[0] == "tapas":
        data, hdr = obt.load_telluric(tapas_path, name)

        res = float(hdr["resPower"])
        sampRati = hdr["sampRati"]
        airmass = hdr["airmass"]

        plt.plot(data[0], data[1], label= "Tapas R=" + str(res) + ", Sampling=" + sampRati + " airmass=" + str(airmass))

    elif name.split("-")[0] == "HD30501":
        wl, Tr = np.loadtxt(ESO_path + name, unpack=True)
        split = name.split("-")
        res = split[2][1:]
        conv = split[3]
        if conv[0:8] == "gaussian":
            fwhm = split[-1].split(".")[0]
        else:
            fwhm = ""
        plt.plot(wl, Tr, label="ESO SkyCalc, R=" + res +", fwhm=" + fwhm)
plt.legend(loc="best")
plt.title("Tapas/ ESO SkyCal Comparison\nR=50000" )
plt.xlim([2117, 2121])
plt.show()

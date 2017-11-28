# Testing wavelength ranges of HD30501 data and models

# to try identify the issue with the wavelength solution

import os
import sys
from ajplanet import pl_rv_array

import ephem
import matplotlib.pyplot as plt
import numpy as np
from simulators.Planet_spectral_simulations import simple_normalization
# Add vac to air
from astropy.io import fits
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from spectrum_overload import Spectrum

import TelluricSpectra.Obtain_Telluric as ot

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'Simulations'))


def load_data():
    obsdata = fits.getdata("/home/jneal/Phd/data/Crires/BDs-DRACS/HD30501-1/Combined_Nods/CRIRE.2012-04-07T00-08-29.976_2.nod.ms.norm.sum.wavecal.fits")
    obsheader = fits.getheader("/home/jneal/Phd/data/Crires/BDs-DRACS/HD30501-1/Combined_Nods/CRIRE.2012-04-07T00-08-29.976_2.nod.ms.norm.sum.wavecal.fits")
    obs_spec = Spectrum(flux=obsdata["Extracted_DRACS"], xaxis=obsdata["Wavelength"], header=obsheader, calibrated=True)

    telldata, hdr = ot.load_telluric("/home/jneal/Phd/data/Crires/BDs-DRACS/HD30501-1/Telluric_files/", "tapas_2012-04-07T00-24-03_ReqId_10_R-50000_sratio-10_barydone-NO.ipac")
    tell_spec = Spectrum(flux=telldata[1], xaxis=telldata[0], header=hdr, calibrated=True)

    wave = fits.getdata("/home/jneal/Phd/data/phoenixmodels/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
    model = fits.getdata("/home/jneal/Phd/data/phoenixmodels/HD30501-lte05200-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
    model_spec = Spectrum(flux=model, xaxis=wave/10, calibrated=True)
    model_spec.wav_select(2100, 2170)
    model_spec = simple_normalization(model_spec)
    return obs_spec, tell_spec, model_spec


def plot_(obs_spec, tell_spec, model_spec, corr_spec=False, aircorr_spec=False):
    plt.plot(tell_spec.xaxis, tell_spec.flux, label="Telluric")
    plt.plot(obs_spec.xaxis, obs_spec.flux, label="obs")
    plt.plot(model_spec.xaxis, model_spec.flux, label="Phoenix Model")

    if corr_spec:
        plt.plot(corr_spec.xaxis, corr_spec.flux, label="Berv Corr spec")
        plt.title("Testing Wavelengths of HD30501 and barycorr")
    else:
        plt.title("Testing Wavelengths of HD30501")

    if aircorr_spec:
        plt.plot(aircorr_spec.xaxis, aircorr_spec.flux, label="Air Berv Corr spec")

    plt.legend()
    plt.show()



obs_spec, tell_spec, model_spec = load_data()

mean_val = 23.71   # km/s mean motion of star
# RV at this time = -246.88
RV_star= -246.88 / 1000
# Shift to location of Star by correcting for RV of host. Negative the RV.

# Parameters
HD30501_params =  [23.710, 1703.1,   70.4, 0.741,   53851.5,   2073.6, 0.81, 90]  # Sahlman
HD30501_params[1] = HD30501_params[1] / 1000   # Convert K! to km/s
HD30501_params[2] = np.deg2rad(HD30501_params[2]) # Omega needs to be in radians for ajplanet

obs_time = obs_spec.header["DATE-OBS"]
jd = ephem.julian_date(obs_time.replace("T"," ").split(".")[0])
Host_RV = pl_rv_array(jd, *HD30501_params[0:6])[0]
print("Host_RV", Host_RV, "km/s")
#print("mean_val", mean_val, "km/s")
#print("RV_star", RV_star, "km/s")
print("mean_val + RV_star", (mean_val + RV_star))

corr_spec = barycorr_crires_spectrum(obs_spec, extra_offset=-Host_RV[0])

# Try correct for the mean motion of the star

# air_wav = pyasl.vactoair(corr_spec.xaxis * 10) / 10
# print(corr_spec.xaxis)
# print(air_wav)
# air_corr_spec = Spectrum(xaxis=air_wav, flux=corr_spec.flux, header=corr_spec.header, calibrated=True)

plot_(obs_spec, tell_spec, model_spec, corr_spec)

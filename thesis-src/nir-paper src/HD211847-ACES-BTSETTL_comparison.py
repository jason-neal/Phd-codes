# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from convolve_spectrum import convolve_spectrum
from matplotlib import rc
from spectrum_overload.spectrum import Spectrum

from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
# Plot spectra with models from ACES and BT-Settl


# Obs1_name = "/home/jneal/Documents/data/handy/HD211847-1-mixavg-tellcorr_1_bervcorr_masked.fits"
Obs1_name = "/home/jneal/.handy_spectra/HD211847-1-mixavg-tellcorr_1_bervcorr_masked.fits"

aces_name = "data/lte05700-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
aces_wav_name = "/home/jneal/.data/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
settl_name = "data/lte5700-4.50-0.0a+0.0.BT-dusty-giant-2013.cf128.sc.spid.fits"

obs = fits.getdata(Obs1_name)
settl = fits.getdata(settl_name)
aces = fits.getdata(aces_name)
aces_wav = fits.getdata(aces_wav_name)

aces_spec = Spectrum(xaxis=aces_wav / 10, flux=aces, header=fits.getheader(aces_name))
settl_spec = Spectrum(xaxis=settl["Wavelength"] * 1e3, flux=settl["flux"], header=fits.getheader(settl_name, 1))
obs_spec = Spectrum(xaxis=obs["wavelength"], flux=obs["flux"], header=fits.getheader(Obs1_name))

# Normalize
aces_spec.wav_select(wav_min=2100, wav_max=2160)
aces_spec = aces_spec.normalize(method="poly", degree=3)

settl_spec.wav_select(wav_min=2100, wav_max=2160)
settl_spec = settl_spec.normalize(method="poly", degree=3)

# Convolve spectra
R = 50000
aces_spec = convolve_spectrum(aces_spec, chip_limits=[aces_spec.xaxis[0], aces_spec.xaxis[-1]], R=R, plot=False)
settl_spec = convolve_spectrum(settl_spec, chip_limits=[settl_spec.xaxis[0], settl_spec.xaxis[-1]], R=R, plot=False)

# shift by 6.6km/s to match observation
aces_spec.doppler_shift(7)
settl_spec.doppler_shift(7)
# Split up sections of spectra over the gaps

wave_diffs = np.diff(obs_spec.xaxis)
indx = np.where(wave_diffs > 0.05)[0] + 1
if len(indx) > 0:
    wave_chunks = np.split(obs_spec.xaxis, indx)
    flux_chunks = np.split(obs_spec.flux, indx)
else:
    wave_chunks = [obs_spec.xaxis]
    flux_chunks = [obs_spec.flux]


@styler
def f(fig, *args, **kwargs):
    for ii, (wchunk, fchunk) in enumerate(zip(wave_chunks, flux_chunks)):
        # plt.plot(obs_spec.xaxis, obs_spec.flux + 0.05, label="HD211847-1", lw=0.6)
        label = "Observed" if ii == 0 else ""
        plt.plot(wchunk, fchunk + 0.05, "C0", label=label, lw=0.6)
    plt.plot(aces_spec.xaxis, aces_spec.flux, "-.C1", label="ACES", lw=0.6)
    plt.plot(settl_spec.xaxis, settl_spec.flux - 0.05, "--C2", label="BT-Settl", lw=0.6)
    plt.legend(fontsize="medium")
    plt.xlim(2111.5, 2124.4)
    plt.ylim([0.8, 1.1])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Flux")
    plt.show()


if __name__ == "__main__":
    print("Starting")

    f(type="one", tight=True, dpi=400, save="../final/HD211847_ACES_BTSettl.pdf", figsize=(None, .70), axislw=0.5,
      formatcbar=False, formatx=False, formaty=False)

    print("Done")

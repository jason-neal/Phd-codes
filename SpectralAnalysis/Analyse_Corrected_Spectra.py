#!/usr/bin/env python3

# Find RV of host:
# from gooey import Gooey, GooeyParser
# from plot_fits import astro_ccf
import argparse

# " Analyse Telluric Corrected spectra"
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sci
from astropy.io import fits

import WavelengthCalibration.GaussianFitting as gf


# @Gooey(program_name='Plot fits - Easy 1D fits plotting', default_size=(610, 730))
def _parser():
    """Take care of all the argparse stuff.

    :returns: the args
    """
    # parser = GooeyParser(description='Analyse Spectra')
    parser = argparse.ArgumentParser(description="Analyse Spectra")
    parser.add_argument("fname", action="store", help="Input Corrected fits spectra")
    # parser.add_argument('-o', '--output', default=False,
    #                    help='Ouput Filename',)
    parser.add_argument(
        "-t",
        "--telluric",
        action="store",
        default=False,
        help="Telluric line Calibrator",
    )
    parser.add_argument(
        "-m", "--model", action="store", default=False, help="Stellar Model"
    )
    # parser = GooeyParser(description='Wavelength Calibrate CRIRES Spectra')
    # parser.add_argument('fname',
    #                    action='store',
    #                    widget='FileChooser',
    #                    help='Input fits file')
    # parser.add_argument('-o', '--output',
    #                    default=False,
    #                    action='store',
    #                    widget='FileChooser',
    #                    help='Ouput Filename',)
    # parser.add_argument('-t', '--telluric',
    #                    default=False,
    #                    action='store',
    #                    widget='FileChooser',
    #                    help='Telluric line Calibrator',)

    args = parser.parse_args()
    return args


def dopplerShift(wvl, flux, v, edgeHandling="firstlast", fill_value=None):
    """Doppler shift a given spectrum.
    Does not interpolate to a new wavelength vector, but does shift it.
    """

    # Shifted wavelength axis
    wlprime = wvl * (1.0 + v / 299792.458)
    return flux, wlprime


def ccf_astro(spectrum1, spectrum2, rvmin=-200, rvmax=200, drv=1):
    """Make a CCF between 2 spectra and find the RV

    :spectrum1: The stellar spectrum
    :spectrum2: The model, sun or telluric
    :dv: The velocity step
    :returns: The RV shift
    """
    # Calculate the cross correlation
    s = False
    w, f = spectrum1
    tw, tf = spectrum2
    if not len(w) or not len(tw):
        return 0, 0, 0, 0, 0
    c = 299792.458
    drvs = np.arange(rvmin, rvmax, drv)
    cc = np.zeros(len(drvs))
    for i, rv in enumerate(drvs):
        fi = sci.interp1d(tw * (1.0 + rv / c), tf)
        # Shifted template evaluated at location of spectrum
        try:
            fiw = fi(w)
            cc[i] = np.sum(f * fiw)
        except ValueError:
            s = True
            fiw = 0
            cc[i] = 0

    if s:
        print("Warning: Lower the bounds on RV")

    if not np.any(cc):
        return 0, 0, 0, 0, 0

    # Fit the CCF with a gaussian
    cc[cc == 0] = np.mean(cc)
    cc = (cc - min(cc)) / (max(cc) - min(cc))
    RV, g = _fit_ccf(drvs, cc)
    return RV, drvs, cc, drvs, g(drvs)


def _fit_ccf(rv, ccf):
    """Fit the CCF with a 1D gaussian
    :rv: The RV vector
    :ccf: The CCF values
    :returns: The RV, and best fit gaussian
    """
    from astropy.modeling import models, fitting

    ampl = 1
    mean = rv[ccf == ampl]
    I = np.where(ccf == ampl)[0]

    g_init = models.Gaussian1D(amplitude=ampl, mean=mean, stddev=5)
    fit_g = fitting.LevMarLSQFitter()

    try:
        g = fit_g(g_init, rv[I - 10 : I + 10], ccf[I - 10 : I + 10])
    except TypeError:
        print("Warning: Not able to fit a gaussian to the CCF")
        return 0, g_init
    RV = g.mean.value
    return RV, g


# Load Spectra

# spectrum 1 and 2 are (wve,flux) tupples


# Spectral recovery of companion?


def main(fname, telluric=False, model=False):
    data = fits.getdata(fname)
    wl = data["Wavelength"]
    I = data["Corrected_DRACS"]

    plt.plot(wl, I, label="corrected spectra")
    plt.xlim([min(wl), max(wl)])
    # plt.show()

    # Load Model
    pathwave = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    if model:
        I_mod = fits.getdata(model)
        hdr = fits.getheader(model)
        if "WAVE" in hdr.keys():
            w_mod = fits.getdata(pathwave)
            w_mod /= 10
            # Slice model
            w_mod, I_mod = gf.slice_spectra(w_mod, I_mod, min(wl) - 0.5, max(wl) + 0.5)
        # Normalise model
        I_mod /= np.median(I_mod)
        plt.plot(w_mod, I_mod, label="Model")

    plt.legend()
    plt.show(block=False)

    RV, drvs, cc, drvs, g = ccf_astro(
        (wl, I), (w_mod, I_mod), rvmin=-50, rvmax=50, drv=1
    )
    # rv2, r_tel, c_tel, x_tel, y_tel = ccf_astro

    print("RV, drvs, cc, drvs, g(drvs)")
    print(RV, drvs, cc, drvs, g)
    plt.figure()
    plt.plot(drvs, cc, "-k", lw=2)
    plt.plot(drvs, g, "--r", lw=2)
    plt.title("CCF (mod): {0!s} km/s".format(int(RV)))
    # ax3.set_title('CCF (tel)')
    plt.xlabel("RV [km/s]")
    # ax3.set_xlabel('RV [km/s]')
    # plt.legend()
    plt.show()

    # Doppler shift the spectra to overlap the stellar features
    # ( possibly use the known values from the BERV coordinates)

    # Subtract the two spectra and see what is left

    return None


if __name__ == "__main__":
    args = vars(_parser())
    fname = args.pop("fname")
    opts = {k: args[k] for k in args}

    main(fname, **opts)

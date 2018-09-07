"""Theoretical differential amplitude.

To make figure for paper.


TODO: Attempt auto-correlation?
"""

import PyAstronomy.pyasl as pyasl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from styler import styler


def lorentzian(wav, x0, fwhm):
    """https://en.wikipedia.org/wiki/Spectral_line_shape"""
    x = (wav - x0) / (fwhm / 2)
    return 1.0 / (1.0 + x ** 2)


def gaussian(wav, x0, fwhm):
    """https://en.wikipedia.org/wiki/Spectral_line_shape"""
    x = (wav - x0) / (fwhm / 2)
    return np.exp(-np.log(2) * x ** 2)


def sprof_gaussian(wav, k, x, fwhm):
    return (1.0 - gaussian(wav, x + k / 2, fwhm)) - (
        1.0 - gaussian(wav, x - k / 2, fwhm)
    )


def sprof_lorentz(wav, k, x, fwhm):
    return (1.0 - lorentzian(wav, x + k / 2, fwhm)) - (
        1.0 - lorentzian(wav, x - k / 2, fwhm)
    )


R = 50000
c = 299792.458  # km/s
rv_fwhm = c / R

# Load the model spectrum
pathwave = "/home/jneal/Phd/data/phoenixmodels/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
# Z-0.0/lte02500-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
specpath = (
    "/home/jneal/Phd/data/phoenixmodels/HD30501b-lte02500-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
)
w_mod = fits.getdata(pathwave)
w_mod /= 10  # to nm

outer_limits = [2080, 2180]
inner_limits = [2110, 2160]
chip_limits = [[2110, 2123], [2135, 2145], [2147, 2153], [2155, 2166], [2110, 2150]]
w_theory = np.linspace(inner_limits[0], inner_limits[1], 40000)
flux = fits.getdata(specpath)
hdr = fits.getheader(specpath)

# Cut at 2050 - 2150nm
mask = (w_mod < outer_limits[1]) & (w_mod > outer_limits[0])
w_mod = w_mod[mask]
flux = flux[mask]

# Normalize by median largest 100 points.
flux /= np.median(flux)
maxes = flux.argsort()[-100:][::-1]
flux /= np.median(flux[maxes])
flux[flux < 0] = 0

# Convolve to R=50000 PyAstronomy (my convolution code also works)
flux = pyasl.instrBroadGaussFast(w_mod, flux, R)

mask2 = (w_mod <= inner_limits[1]) & (
    w_mod >= inner_limits[0]
)  # Mask only CRIRES region
w_mod2 = w_mod[mask2]
flux2 = flux[mask2]

# Iterate over RV offsets.
rvs = np.arange(-10, 10, 0.001)
model_rv_amp = np.zeros_like(rvs)
model_rv_amp1 = np.zeros_like(rvs)  # chips 1-4
model_rv_amp2 = np.zeros_like(rvs)
model_rv_amp3 = np.zeros_like(rvs)
model_rv_amp4 = np.zeros_like(rvs)

for i, rv in enumerate(rvs):
    # Doppler shift model and calculate the maximum line amplitude.
    nflux_rv, __ = pyasl.dopplerShift(
        w_mod, flux, rv, edgeHandling=None, fillValue=None
    )
    diff = flux - nflux_rv
    wav_nan = w_mod[~np.isnan(diff)]
    diff = diff[~np.isnan(diff)]

    model_rv_amp[i] = np.max(
        np.abs(diff[(wav_nan <= inner_limits[1]) & (wav_nan >= inner_limits[0])])
    )
    # Limit for each chip
    model_rv_amp1[i] = np.max(
        np.abs(diff[(wav_nan <= chip_limits[0][1]) & (wav_nan >= chip_limits[0][0])])
    )
    model_rv_amp2[i] = np.max(
        np.abs(diff[(wav_nan <= chip_limits[1][1]) & (wav_nan >= chip_limits[1][0])])
    )
    model_rv_amp3[i] = np.max(
        np.abs(diff[(wav_nan <= chip_limits[2][1]) & (wav_nan >= chip_limits[2][0])])
    )
    model_rv_amp4[i] = np.max(
        np.abs(diff[(wav_nan <= chip_limits[3][1]) & (wav_nan >= chip_limits[3][0])])
    )

theoretical_gauss_list = []
theoretical_lorentz_list = []
for chip in range(1, 6):
    theory_gaussian = np.zeros_like(rvs)
    theory_lorentz = np.zeros_like(rvs)

    mask = (w_mod <= chip_limits[chip - 1][1]) & (
        w_mod >= chip_limits[chip - 1][0]
    )  # mask chip
    w_chip = w_mod[mask]
    wav0 = np.median(w_chip)
    fwhm0 = wav0 / R
    for i, rv in enumerate(rvs):
        wave_shift = rv * wav0 / c
        theory_gaussian[i] = np.max(sprof_gaussian(w_theory, wave_shift, wav0, fwhm0))
        theory_lorentz[i] = np.max(sprof_lorentz(w_theory, wave_shift, wav0, fwhm0))

    theoretical_gauss_list.append(theory_gaussian)
    theoretical_lorentz_list.append(theory_lorentz)


@styler
def g(fig, *args, **kwargs):
    ax = fig.add_subplot(111)
    plt.plot(rvs, model_rv_amp, label="Full PHOENIX-ACES")
    plt.plot(rvs, theoretical_gauss_list[0], "--", label="Gaussian 1")
    plt.plot(rvs, theoretical_gauss_list[1], ":", label="Gaussian 2")
    plt.plot(rvs, theoretical_gauss_list[2], "-.", label="Gaussian 3")
    plt.plot(rvs, theoretical_gauss_list[3], "-", label="Gaussian 4")
    plt.plot(rvs, theoretical_lorentz_list[0], "-.", label="Lorentzian 1")
    plt.plot(rvs, theoretical_lorentz_list[1], ":", label="Lorentzian 2")
    plt.plot(rvs, theoretical_lorentz_list[2], "-.", label="Lorentzian 3")
    plt.plot(rvs, theoretical_lorentz_list[3], "-", label="Lorentzian 4")
    plt.vlines(x=[-1.2, 1.2], ymin=-0.1, ymax=1.1, label="Max RV", alpha=0.5)
    plt.vlines(
        x=[-rv_fwhm, rv_fwhm],
        ymin=-0.1,
        ymax=1.1,
        linestyle="--",
        label=r"$\rm RV_{FWHM}$",
        alpha=0.5,
    )
    ax.set_xlabel("Radial Velocity (km/s)")
    ax.set_ylabel("Relative Amplitude")
    plt.ylim([-0.05, 1.1])
    plt.legend()
    # plt.show()


g(type="A&A", save="../rv_diff_amplitude_chip_test.pdf", tight=True)


@styler
def h(fig, *args, **kwargs):
    ax = fig.add_subplot(111)
    plt.plot(rvs, model_rv_amp, label="PHOENIX-ACES")
    plt.plot(rvs, model_rv_amp1, "--", label="PHOENIX-ACES 1")
    plt.plot(rvs, model_rv_amp2, ":", label="PHOENIX-ACES 2")
    plt.plot(rvs, model_rv_amp3, "-.", label="PHOENIX-ACES 3")
    plt.plot(rvs, model_rv_amp4, "-", label="PHOENIX-ACES 4")
    plt.plot(rvs, theoretical_gauss_list[0], "--", label="Gaussian 1")
    plt.plot(rvs, theoretical_gauss_list[1], ":", label="Gaussian 2")
    plt.plot(rvs, theoretical_gauss_list[2], "-.", label="Gaussian 3")
    plt.plot(rvs, theoretical_gauss_list[3], "-", label="Gaussian 4")
    plt.plot(rvs, theoretical_lorentz_list[0], "-.", label="Lorentzian 1")
    plt.plot(rvs, theoretical_lorentz_list[1], ":", label="Lorentzian 2")
    plt.plot(rvs, theoretical_lorentz_list[2], "-.", label="Lorentzian 3")
    plt.plot(rvs, theoretical_lorentz_list[3], "-", label="Lorentzian 4")
    plt.vlines(x=[-1.2, 1.2], ymin=-0.1, ymax=1.1, label="Max RV", alpha=0.5)
    plt.vlines(
        x=[-rv_fwhm, rv_fwhm],
        ymin=-0.1,
        ymax=1.1,
        linestyle="--",
        label=r"$\rm RV_{FWHM}$",
        alpha=0.5,
    )
    ax.set_xlabel("Radial Velocity (km/s)")
    ax.set_ylabel("Relative Amplitude")
    plt.ylim([-0.05, 1.1])
    plt.legend()
    # plt.show()


h(type="A&A", save="../rv_diff_amplitude_chip_test2.pdf", tight=True)

#########################################

# Detector 1 looks good for paper
phoenix_amp = model_rv_amp1
theory_gaussian_amp = theoretical_gauss_list[0]
theory_lorentz_amp = theoretical_lorentz_list[0]

# Scale amplitudes to 1 at edges
# Median Value between 7 and 12 km/s
rvs_mask = (abs(rvs) >= 7) & (abs(rvs) <= 12)

phoenix_scale = np.median(phoenix_amp[rvs_mask])
theory_gaussian_scale = np.median(theory_gaussian_amp[rvs_mask])
theory_lorentz_scale = np.median(theory_lorentz_amp[rvs_mask])


# Make the figure
@styler
def f(fig, *args, **kwargs):
    ax = fig.add_subplot(111)
    plt.plot(rvs, phoenix_amp / phoenix_scale, label="PHOENIX-ACES")
    # plt.plot(rvs, theory_amp / theory_scale, label="Theory")
    plt.plot(rvs, theory_gaussian_amp / theory_gaussian_scale, "--", label="Gaussian")
    plt.plot(rvs, theory_lorentz_amp / theory_lorentz_scale, "-.", label="Lorentzian")
    plt.vlines(x=[-1.2, 1.2], ymin=-0.1, ymax=1.1, label="Max RV", alpha=0.5)
    plt.vlines(
        x=[-rv_fwhm, rv_fwhm],
        ymin=-0.1,
        ymax=1.1,
        linestyle="--",
        label=r"$\rm RV_{FWHM}$",
        alpha=0.5,
    )
    ax.set_xlabel("Radial Velocity (km/s)")
    ax.set_ylabel("Relative Amplitude")
    plt.ylim([-0.05, 1.1])
    plt.legend()
    # plt.show()


f(type="A&A", save="../rv_diff_amplitude_figure_final.pdf", tight=True)
print("Done")

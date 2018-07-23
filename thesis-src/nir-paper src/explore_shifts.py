import copy
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
from TelluricSpectra import Obtain_Telluric as obt
from astropy.io import fits
from spectrum_overload.spectrum import Spectrum

""" Use bary corrected spectra.
    and the RV class.
"""

sys.path.append(".")
from styler import colors_named, styler

print(colors_named)
from utils.rv_utils import RV


def load_telluric(name, limits, normalize=False):
    data, hdr = obt.load_telluric("", name)
    wl = data[0]
    I = data[1]
    # Normalize
    if normalize:
        maxes = I[(I < 1.2)].argsort()[-50:][::-1]
        I = I / np.median(I[maxes])
    wlmask = (wl > limits[0]) & (wl < limits[-1])
    wl = wl[wlmask]
    I = I[wlmask]
    return wl, I, hdr


def load_spectrum(name, limits=None, normalize=False):
    data = fits.getdata(name)
    hdr = fits.getheader(name)
    wl = data["Wavelength"]
    I = data["Flux"]
    if normalize:
        maxes = I[(I < 1.2)].argsort()[-50:][::-1]
        I = I / np.median(I[maxes])
    if limits is not None:
        wlmask = (wl > limits[0]) & (wl < limits[-1])
        wl = wl[wlmask]
        I = I[wlmask]
    return wl, I, hdr


handy_path = "/home/jneal/.handy_spectra/"
raw_path_template = "/home/jneal/Phd/data/Crires/BDs-DRACS/2017/{0}/Combined_Nods/"
wavecal_template = "{0}CRIRE*_{1}.nod.ms.norm.mixavg.wavecal.fits"
tellcorr_template = "{0}{1}-{2}-mixavg-{3}tellcorr_{4}.fits"
bervcorr_template = "{0}{1}-{2}-mixavg-{3}tellcorr_{4}_bervcorr.fits"


# def prepare_differential(star, chip, obs_num, ref_num, use_h2o=False):
#    return raw_obs, raw_ref, obs, ref_obs, telluric


@styler
def differential_plot(fig, *, spec_1=None, spec_2=None, telluric=None, diff=None, masks_1=None, masks_2=None,
                      masks_3=None, **kwargs):
    pltkwargs = {"lw": kwargs.get("lw", 1)}

    for ii, variable in enumerate((spec_1, spec_2, telluric, masks_1, masks_2, diff, masks_3)):
        assert variable is not None, f"variable {ii} is None "

    # obs_1 = kwargs.get("obs")
    # obs_2 = kwargs.get("ref_obs")
    obs_2 = ref_obs
    raw_obs_1 = kwargs.get("raw_obs")
    raw_obs_2 = kwargs.get("raw_ref")
    # diff = kwargs.get("diff")
    # telluric = kwargs.get("telluric")

    # FIRST PLOT
    ax1 = plt.subplot(311)
    ax1.hlines(1, spec_1.xaxis[0], spec_1.xaxis[-1], colors='k', linestyles='dashed', label='', alpha=0.5)
    # Spectrum with telluric lines present
    spec_1.plot(axes=ax1, label="Observed", **pltkwargs)
    # Telluric lines
    telluric.plot(axes=ax1, label="Telluric", **pltkwargs)

    ax1.set_ylabel("Norm\nFlux")
    plt.legend()

    ax2 = plt.subplot(312)
    ax2.hlines(1, spec_2.xaxis[0], spec_2.xaxis[-1], colors='k', linestyles='dashed', label='', alpha=0.5)
    # obs_1.plot(axes=ax2, label="", **pltkwargs)
    # raw_obs_2.plot(axes=ax2, label="Observed", **pltkwargs)
    spec_2.plot(axes=ax2, label="", **pltkwargs)
    ax2.set_ylabel("Norm\nFlux")

    # telluric.plot(axes=ax2, label="telluric", **pltkwargs)
    # plt.legend()

    ax3 = plt.subplot(313)
    # diff = obs_1-obs_2
    diff.plot(label="Difference", **pltkwargs)
    ax3.set_ylabel("$\Delta$ Flux")

    # Line masking
    mask_value = 0.96  # 4 %
    ref_wl = spec_2.xaxis
    tap_wl = telluric.xaxis
    star_mask = spec_2.flux < mask_value
    tapas_mask = telluric.flux < mask_value
    plt.legend()

    for ax_, low, high in zip([ax1, ax2, ax3], [0.5, 0.5, -0.03], [1.1, 1.1, 0.02]):
        ax_.fill_between(ref_wl, low * np.ones_like(ref_wl), high * np.ones_like(ref_wl), where=star_mask, alpha=0.7,
                         color="C3", linewidth=0)
        # color=colors_named["lightblue"], linewidth=0)
        ax_.fill_between(tap_wl, low * np.ones_like(tap_wl), high * np.ones_like(tap_wl), where=tapas_mask, alpha=0.7,
                         color="C2", linewidth=0)
        # color=colors_named["green"], linewidth=0)


def get_target_rv(star, obs_jd):
    param_file = '/home/jneal/Phd/data/parameter_files/{0}_params.dat'.format(star)

    rv = RV.from_file(param_file)
    rv_jd = rv.rv_at_times(obs_jd)
    return rv_jd[0]


if __name__ == "__main__":
    # Parameters to alter to change spectra seen
    star = "HD30501"
    chip_num = 1
    obs_num = "1"
    ref_num = "3"
    use_h2o = False
    assert obs_num != ref_num, "Same object selected to compare."
    # paper_plot = main(star, chip_num, obs_num, ref_num, use_h2o)

    fig = plt.figure()

    # paper_plot(fig, type='A&A', save='New_Spectrum_and_diff_simulated_{0}.pdf'.format(_id), tight=True,
    #           figsize=(None, 1))

    # raw_obs, raw_ref, obs, ref_obs, telluric = prepare_differential(star, chip_num, obs_num, ref_num, use_h2o)

    # Determine file names
    assert obs_num != ref_num, "obs_num and ref_num must be different."
    h2o = "h2o" if use_h2o else ""

    target = "{}-{}".format(star, obs_num)
    reference_target = "{}-{}".format(star, ref_num)  # should be different from target

    target_path = raw_path_template.format(target)
    ref_target_path = raw_path_template.format(reference_target)
    raw_target_name = glob.glob(wavecal_template.format(target_path, chip_num))[0]
    raw_ref_target_name = glob.glob(wavecal_template.format(ref_target_path, chip_num))[0]

    corr_name = glob.glob(tellcorr_template.format(handy_path, star, obs_num, h2o, chip_num))[0]
    ref_corr_name = glob.glob(tellcorr_template.format(handy_path, star, ref_num, h2o, chip_num))[0]

    # Load in the spectra
    raw_wl, raw_I, raw_hdr = load_spectrum(raw_target_name, normalize=True)
    raw_obs = Spectrum(xaxis=raw_wl, flux=raw_I, header=raw_hdr)

    raw_ref_wl, raw_ref_I, raw_ref_hdr = load_spectrum(raw_ref_target_name, normalize=True)
    raw_ref = Spectrum(xaxis=raw_ref_wl, flux=raw_ref_I, header=raw_ref_hdr)

    obs_wl, obs_I, obs_hdr = load_spectrum(corr_name)
    obs = Spectrum(xaxis=obs_wl, flux=obs_I, header=obs_hdr)

    ref_wl, ref_I, ref_hdr = load_spectrum(ref_corr_name)
    ref_obs = Spectrum(xaxis=ref_wl, flux=ref_I, header=ref_hdr)

    limits = [raw_wl[0] - 0.25, raw_wl[-1] + 0.25]
    tapas_name = glob.glob("{0}/../Telluric_files/tapas_*ReqId_10*".format(ref_target_path))[0]
    tap_wl, tap_I, tap_hdr = load_telluric(tapas_name, limits)
    telluric = Spectrum(xaxis=tap_wl, flux=tap_I, header=tap_hdr)
    # Make sure to copy so that they can each be changed
    telluric_1 = telluric.copy()
    telluric_1.header = copy.copy(obs.header)
    telluric_2 = telluric.copy()
    telluric_2.header = copy.copy(ref_obs.header)

    # BERVCORR the spectrum
    from mingle.utilities.crires_utilities import barycorr_crires_spectrum

    bary_obs = barycorr_crires_spectrum(obs)
    bary_telluric_1 = barycorr_crires_spectrum(telluric_1)
    bary_ref_obs = barycorr_crires_spectrum(ref_obs)
    bary_telluric_2 = barycorr_crires_spectrum(telluric_2)

    ## Calculate RV of hosts
    jd_1 = float(bary_obs.header["MJD-OBS"]) + 2400000
    jd_2 = float(bary_ref_obs.header["MJD-OBS"]) + 2400000

    obs_rv = get_target_rv(star, jd_1)
    ref_rv = get_target_rv(star, jd_2)
    print("RVs", obs_rv, ref_rv)

    ## RV shift hosts to zero RV
    bary_obs.doppler_shift(obs_rv)
    bary_telluric_1.doppler_shift(obs_rv)

    bary_ref_obs.doppler_shift(ref_rv)
    bary_telluric_2.doppler_shift(ref_rv)

    telluric_diff_mask = bary_telluric_1.copy()
    obs_diff_mask = obs


    # Remove nans before diff
    bary_obs = bary_obs.remove_nans()
    bary_ref_obs=bary_ref_obs.remove_nans()

    diff = bary_obs - bary_ref_obs
    print("diff ", diff)
    print("diff flux", diff.flux[~np.isnan(diff.flux)])
    print("diff xaxis", diff.xaxis[~np.isnan(diff.flux)])
    print("ref_obs", ref_obs)
    differential_plot(type='A&A', tight=True, figsize=(None, 1), lw=1, dpi=300,
                      spec_1=raw_ref, spec_2=ref_obs, diff=diff,
                      masks_1=[obs, telluric],
                      masks_2=[obs, telluric],
                      masks_3=[obs, telluric],
                      telluric=telluric)
    plt.show()
    # save='New_differential_{0}_{2}-{3}_{1}.pdf'.format(star, chip_num, obs_num, ref_num),
    print("done")

# TODO: Pass the newly shifted values into the differential plot.

# TODO: a 8 panel with incremental rv shifts about 0.
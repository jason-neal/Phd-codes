import copy
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
from TelluricSpectra import Obtain_Telluric as obt
from astropy.io import fits
from mingle.utilities.crires_utilities import barycorr_crires_spectrum
from spectrum_overload.spectrum import Spectrum
from utils.rv_utils import RV

sys.path.append(".")
from styler import colors_named, styler

print(colors_named)


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


@styler
def differential_plot(fig, *, spec_1=None, spec_2=None, telluric=None, diff=None, masks_1=None, masks_2=None,
                      masks_3=None, **kwargs):
    pltkwargs = {"lw": kwargs.get("lw", 1)}
    mask_value = kwargs.get("mask_value", 0.96)  # 4 %
    for ii, variable in enumerate((spec_1, spec_2, telluric, masks_1, masks_2, diff, masks_3)):
        assert variable is not None, f"variable {ii} is None "

    # FIRST PLOT
    ax1 = plt.subplot(311)
    ax1.hlines(1, spec_1.xaxis[0] - 0.3, spec_1.xaxis[-1] + 0.3, colors='k', linestyles='dashed', label='', alpha=0.5)
    plt.plot(spec_1.xaxis, spec_1.flux, label="Observed", **pltkwargs)
    plt.plot(telluric.xaxis, telluric.flux, "--", label="Telluric", **pltkwargs)
    ax1.set_ylabel("Norm Flux")

    # masks_1
    low, high = .52, 1.1
    assert len(masks_1) == 2
    mask_obs, mask_tell = masks_1[0], masks_1[1]
    bound = np.ones_like(mask_obs.xaxis)
    bound_tell = np.ones_like(mask_tell.xaxis)
    star_mask = mask_obs.flux < mask_value
    tell_mask = mask_tell.flux < mask_value
    ax1.fill_between(mask_obs.xaxis, low * bound, high * bound, where=star_mask, alpha=0.7,
                     color=colors_named["lightblue"], linewidth=0)
    ax1.fill_between(mask_tell.xaxis, low * bound_tell, high * bound_tell, where=tell_mask, alpha=0.7,
                     color=colors_named["green"], linewidth=0)
    plt.legend()

    # Second Plot
    ax2 = plt.subplot(312, sharex=ax1)
    ax2.hlines(1, spec_2.xaxis[0] - 0.3, spec_2.xaxis[-1] + 0.3, colors='k', linestyles='dashed', label='', alpha=0.5)
    # spec_2.plot(axes=ax2, label="", **pltkwargs)  # Telluric corrected spectrum
    plt.plot(spec_2.xaxis, spec_2.flux, label="", **pltkwargs)

    ax2.set_ylabel("Norm Flux")

    # masks_2
    low, high = .52, 1.1
    assert len(masks_2) == 2
    mask_obs, mask_tell = masks_2[0], masks_2[1]
    bound = np.ones_like(mask_obs.xaxis)
    bound_tell = np.ones_like(mask_tell.xaxis)
    star_mask = mask_obs.flux < mask_value
    tell_mask = mask_tell.flux < mask_value
    ax2.fill_between(mask_obs.xaxis, low * bound, high * bound, where=star_mask, alpha=0.7,
                     color=colors_named["lightblue"], linewidth=0)
    ax2.fill_between(mask_tell.xaxis, low * bound_tell, high * bound_tell, where=tell_mask, alpha=0.7,
                     color=colors_named["green"], linewidth=0)

    # Third Plot
    ax3 = plt.subplot(313, sharex=ax1)
    # diff = obs_1-obs_2
    ax3.hlines(0, diff.xaxis[0] - 0.3, diff.xaxis[-1] + 0.3, colors='k', linestyles='dashed', label='', alpha=0.5)

    diff_label = kwargs.get("diff_label", "Difference")
    # diff.plot(label=diff_label, **pltkwargs)
    plt.plot(diff.xaxis, diff.flux, label=diff_label, **pltkwargs)

    synthetic_diff = kwargs.get("synth_diff", False)
    if synthetic_diff:
        synthetic_diff.plot(label="Simulated", linestyle="dashed", color="C1", **pltkwargs)
        plt.legend(ncol=1, loc="upper left")
    ax3.set_ylabel("$\Delta$ Flux")

    # masks_2
    low, high = -0.027, 0.04
    assert len(masks_3) == 4
    mask_obs_a, mask_obs_b, mask_tell_a, mask_tell_b = masks_3[0], masks_3[1], masks_3[2], masks_3[3]
    for mask_obs, mask_tell in zip((mask_obs_a, mask_obs_b), (mask_tell_a, mask_tell_b)):
        bound = np.ones_like(mask_obs.xaxis)
        bound_tell = np.ones_like(mask_tell.xaxis)
        star_mask = mask_obs.flux < mask_value
        tell_mask = mask_tell.flux < mask_value
        ax3.fill_between(mask_obs.xaxis, low * bound, high * bound, where=star_mask, alpha=0.7,
                         color=colors_named["lightblue"], linewidth=0)
        ax3.fill_between(mask_tell.xaxis, low * bound_tell, high * bound_tell, where=tell_mask, alpha=0.7,
                         color=colors_named["green"], linewidth=0)

    # Line masking

    # ref_wl = spec_2.xaxis
    # tap_wl = telluric.xaxis
    # star_mask = spec_2.flux < mask_value
    # tapas_mask = telluric.flux < mask_value
    ## TODO add masking properly
    # for ax_, low, high in zip([ax1, ax2, ax3], [0.52, 0.52, -0.03], [1.1, 1.1, 0.03]):
    #    ax_.fill_between(ref_wl, low * np.ones_like(ref_wl), high * np.ones_like(ref_wl), where=star_mask, alpha=0.7,
    #                     #color="C2", linewidth=0)
    #                     color=colors_named["lightblue"], linewidth=0)
    #    ax_.fill_between(tap_wl, low * np.ones_like(tap_wl), high * np.ones_like(tap_wl), where=tapas_mask, alpha=0.7,
    #                     # color="C4", linewidth=0)
    #                     color=colors_named["green"], linewidth=0)


@styler
def incremental_diff(fig, *, diffs=None, rvs=None, **kwargs):
    pltkwargs = {"lw": kwargs.get("lw", 1)}

    for ii, (shift_rv, diff) in enumerate(zip(rvs, diffs)):
        ax = plt.subplot(len(diffs), 1, ii + 1)
        diff.plot(label=f"RV={shift_rv:0.03f} km/s", **pltkwargs)
        plt.legend()
        ax.hlines(0, diff.xaxis[0], diff.xaxis[-1], colors='k', linestyles='dashed', label='', alpha=0.5)
        ax.set_ylabel("$\Delta$ Flux")


def get_target_rv(star, obs_jd, mass_ratio=None):
    param_file = '/home/jneal/Phd/data/parameter_files/{0}_params.dat'.format(star)

    host = RV.from_file(param_file)
    rv_jd = host.rv_at_times(obs_jd)

    comp = host.create_companion(mass_ratio=mass_ratio)
    rv_comp = comp.rv_at_times(obs_jd)
    return rv_jd[0], rv_comp[0]


if __name__ == "__main__":
    # Parameters to alter to change spectra seen
    star = "HD30501"
    chip_num = 1
    obs_num = "1"
    ref_num = "3"
    mass_ratio = 9.4  # Set for M1 = 0.81 Msol, M2 = 89.6 Mjup
    use_h2o = False
    assert obs_num != ref_num, "Same object selected to compare."

    h2o = "h2o" if use_h2o else ""

    # Determine file names
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

    bary_obs = barycorr_crires_spectrum(obs)
    bary_telluric_1 = barycorr_crires_spectrum(telluric_1)
    bary_ref_obs = barycorr_crires_spectrum(ref_obs)
    bary_telluric_2 = barycorr_crires_spectrum(telluric_2)

    ## Calculate RV of hosts
    jd_1 = float(bary_obs.header["MJD-OBS"]) + 2400000
    jd_2 = float(bary_ref_obs.header["MJD-OBS"]) + 2400000

    obs_rv, obs_comp_rv = get_target_rv(star, jd_1, mass_ratio=mass_ratio)
    ref_rv, ref_comp_rv = get_target_rv(star, jd_2, mass_ratio=mass_ratio)
    print("RVs", obs_rv, ref_rv)
    print("RVs", obs_comp_rv, ref_comp_rv)

    ## RV shift hosts to zero RV
    bary_obs.doppler_shift(obs_rv + 0.3)  # Small hack of 300m/s to completely cancel out host lines for plot
    bary_telluric_1.doppler_shift(obs_rv)

    bary_ref_obs.doppler_shift(ref_rv)
    bary_telluric_2.doppler_shift(ref_rv)

    telluric_diff_mask = bary_telluric_1.copy()
    obs_diff_mask = obs

    # Remove nans before diff
    bary_obs = bary_obs.remove_nans()
    bary_ref_obs = bary_ref_obs.remove_nans()

    diff = bary_obs - bary_ref_obs

    # Calculate synthetic diff
    pathwave = "/home/jneal/Phd/data/phoenixmodels/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    specpath = "/home/jneal/Phd/data/phoenixmodels/HD30501b-lte02500-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    w_mod = fits.getdata(pathwave)
    w_mod /= 10  # turn into nm
    mod_flux = fits.getdata(specpath)
    model = Spectrum(xaxis=w_mod, flux=mod_flux)
    model.wav_select(2111.9, 2123.7)
    model = model.instrument_broaden(R=50000)
    model = model.normalize(method="linear")
    model *= 0.01  # Scale by flux ratio
    model_1 = model.copy()
    model_2 = model.copy()

    shift_1 = obs_comp_rv - obs_rv
    shift_2 = ref_comp_rv - ref_rv
    print(f"Synthetic rv shifts = {shift_1:5.04f},  {shift_2:5.04f} km/s")
    print(f"Difference used in model shift {shift_1 - shift_2:5.04f} km/s")
    model_1.doppler_shift(shift_1)
    model_2.doppler_shift(shift_2)
    synthetic_diff = model_1 - model_2

    differential_plot(type='A&A', save="../final/differential.pdf",
                      tight=True, figsize=(None, 1), lw=1, dpi=400,
                      spec_1=raw_ref, spec_2=ref_obs, diff=diff, axislw=0.5,
                      masks_1=[ref_obs, telluric],
                      masks_2=[ref_obs, telluric],
                      masks_3=[bary_obs, bary_ref_obs, telluric_1, telluric_2],
                      telluric=telluric, synth_diff=synthetic_diff, diff_label="Observed")

    # differential_plot(type='A&A',
    #                   tight=True, figsize=(None, 1), lw=1, dpi=400,
    #                   spec_1=raw_ref, spec_2=ref_obs, diff=diff, axislw=0.5,
    #                   masks_1=[ref_obs, telluric],
    #                   masks_2=[ref_obs, telluric],
    #                   masks_3=[bary_obs, bary_ref_obs, telluric_1, telluric_2],
    #                   telluric=telluric, synth_diff=synthetic_diff, diff_label="Observed")

    print(f"Maximum simulated diff amplitude = {np.nanmax(np.abs(synthetic_diff.flux)):5.04f} km/s")
    # Increment the RV around 0 to see how they evolve
    incremental_rvs = np.linspace(-2, 2.1, 8)
    diffs = []
    for rv in incremental_rvs:
        bary_obs_shift = bary_obs.copy()
        bary_obs_shift.doppler_shift(rv)
        diffs.append(bary_obs_shift - bary_ref_obs)

    # incremental_diff(type='A&A', tight=True, figsize=(None, 3), lw=1, dpi=300,
    #                 rvs=incremental_rvs, diffs=diffs)
    incremental_rvs2 = np.linspace(-0.5, 0.5, 8)
    diffs2 = []
    for rv in incremental_rvs2:
        bary_obs_shift = bary_obs.copy()
        bary_obs_shift.doppler_shift(rv)
        diffs2.append(bary_obs_shift - bary_ref_obs)

    # incremental_diff(type='A&A', tight=True, figsize=(None, 3), lw=1, dpi=300,
    #                 rvs=incremental_rvs2, diffs=diffs2)

    # TODO : minimize residuals to find rv?

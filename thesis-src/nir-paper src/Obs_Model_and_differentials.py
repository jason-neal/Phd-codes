import datetime
import glob
import sys
import time

from TelluricSpectra import Obtain_Telluric as obt
import matplotlib.pyplot as plt
import numpy as np
from PyAstronomy import pyasl
from astropy.io import fits

from scipy.interpolate import interp1d

sys.path.append(".")
from styler import colors_named, styler
from mingle.utilities.param_file import parse_paramfile
from utils.old import RV_from_params, companion_amplitude
from utils.rv_utils import RV

from mingle.utilities.crires_utilities import barycorr_crires

# def barycorr_CRIRES(wavelength, flux, header, extra_offset=None):
#     # """
#     # Calculate Heliocenteric correction values and apply to spectrum.
#     # """
#     longitude = float(header["HIERARCH ESO TEL GEOLON"])
#     latitude = float(header["HIERARCH ESO TEL GEOLAT"])
#     altitude = float(header["HIERARCH ESO TEL GEOELEV"])
#
#     ra = header["RA"]  # CRIRES RA already in degrees
#     dec = header["DEC"]  # CRIRES hdr DEC already in degrees
#
#     # Pyastronomy helcorr needs the time of observation in julian Days
#     ##########################################################################################
#     Time = header["DATE-OBS"]  # Observing date  '2012-08-02T08:47:30.8425'
#     # Get Average time **** from all raw files!!!  #################################################################
#
#     wholetime, fractionaltime = Time.split(".")
#     Time_time = time.strptime(wholetime, "%Y-%m-%dT%H:%M:%S")
#     dt = datetime.datetime(*Time_time[:6])  # Turn into datetime object
#     # Account for fractions of a second
#     seconds_fractionalpart = float("0." + fractionaltime) / (24 * 60 * 60)  # Divide by seconds in a day
#
#     # Including the fractional part of seconds changes pyasl.helcorr RV by the order of 1cm/s
#     jd = pyasl.asl.astroTimeLegacy.jdcnv(dt) + seconds_fractionalpart
#
#     # Calculate helocentric velocity
#     helcorr = pyasl.helcorr(longitude, latitude, altitude, ra, dec, jd, debug=False)
#
#     if extra_offset is not None:
#         print("Warning!!!! have included a manual offset for testing")
#         helcorr_val = helcorr[0] + extra_offset
#     else:
#         helcorr_val = helcorr[0]
#     # Apply doopler shift to the target spectra with helcorr correction velocity
#     nflux, wlprime = pyasl.dopplerShift(wavelength, flux, helcorr_val, edgeHandling=None, fillValue=None)
#
#     print(" RV size of heliocenter correction for spectra", helcorr_val)
#     return nflux, wlprime


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


def main(star, chip, obs, ref, use_h20=False):
    h2o = "h2o" if use_h2o else ""

    _id = "{0}_{1}-{2}_{3}".format(star, obs_num, ref_num, chip)
    target = "{}-{}".format(star, obs_num)
    reference_target = "{}-{}".format(star, ref_num)  # should be different from target

    handy_path = "/home/jneal/.handy_spectra/"
    target_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/2017/{0}/Combined_Nods/".format(target)
    ref_target_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/2017/{0}/Combined_Nods/".format(reference_target)
    target_name = glob.glob("{0}CRIRE*_{1}.nod.ms.norm.mixavg.wavecal.fits".format(target_path, chip_num))[0]
    ref_target_name = glob.glob("{0}CRIRE*_{1}.nod.ms.norm.mixavg.wavecal.fits".format(ref_target_path, chip_num))[0]
    tellcorr_name = glob.glob("{0}{1}-{2}-mixavg-{3}tellcorr_{4}.fits".format(handy_path, star, obs_num,
                                                                              h2o, chip_num))[0]
    reftellcorr_name = glob.glob("{0}{1}-{2}-mixavg-{3}tellcorr_{4}.fits".format(handy_path, star,
                                                                                 ref_num, h2o, chip_num))[0]
    refh20tellcorr_name = glob.glob("{0}{1}-{2}-mixavg-h2otellcorr_{3}.fits".format(handy_path, star,
                                                                                    ref_num, chip_num))[0]
    reforgtellcorr_name = glob.glob("{0}{1}-{2}-mixavg-tellcorr_{3}.fits".format(handy_path, star, ref_num, chip_num))[
        0]

    tapas_name = glob.glob("{0}/../Telluric_files/tapas_*ReqId_10*".format(ref_target_path))[0]

    # Wavelength calibrated spectrum
    wl, I, hdr = load_spectrum(ref_target_name, normalize=True)
    limits = [wl[0] - 0.25, wl[-1] + 0.25]

    tap_wl, tap_I, tap_hdr = load_telluric(tapas_name, limits)

    obs_wl, obs_I, obs_hdr = load_spectrum(tellcorr_name)
    ref_wl, ref_I, ref_hdr = load_spectrum(reftellcorr_name)

    h20ref_wl, h20ref_I, h20ref_hdr = load_spectrum(refh20tellcorr_name)
    orgref_wl, orgref_I, orgref_hdr = load_spectrum(reforgtellcorr_name)

    @styler
    def correction_comparison(fig, *args, **kwargs):
        ax = fig.add_subplot(111)
        ax.plot(orgref_wl, orgref_I, label=r"Telluric correction")
        ax.plot(h20ref_wl, h20ref_I, label=r"$H_{2}O$", linestyle="-.")
        plt.legend()
        plt.title("Telluric correction")
        plt.xlabel("Wavlength (nm)")

    # fig1 = plt.figure()
    # correction_comparison(fig1, type='A&A', save='./tellcorr/Tellric_corrections_{0}.pdf'.format(_id), tight=True)

    plt.figure()
    plt.subplot(121)
    plt.plot(ref_wl, ref_I, label="Reference")
    plt.plot(obs_wl, obs_I, label="Target")
    plt.title("Not BERV Corrected")
    plt.xlabel("Wavelength(nm)")

    # Barycentric correct observations
    # obs_I_old, ref_I_old = obs_I, ref_I
    __, obs_I = barycorr_crires(obs_wl, obs_I, obs_hdr, extra_offset=None)
    __, ref_I = barycorr_crires(ref_wl, ref_I, ref_hdr, extra_offset=None)

    # baryshift not tellcorrect and tapas also
    __, tap_I = barycorr_crires(tap_wl, tap_I, hdr, extra_offset=None)
    __, I = barycorr_crires(wl, I, hdr, extra_offset=None)

    plt.subplot(122)
    plt.plot(ref_wl, ref_I, label="Reference")
    plt.plot(obs_wl, obs_I, label="Target")
    plt.title("BERV Corrected")
    plt.xlabel("Wavelength(nm)")
    figname = "BERV_check_{0}.pdf".format(_id)
    plt.savefig(figname)
    # plt.show()
    plt.close()
    # Doppler shift to RV reference star!

    param_file = '/home/jneal/Phd/data/parameter_files/{0}_params.dat'.format(star)
    parameters = parse_paramfile(param_file)
    print("Parameters", parameters)
    parameters["tau"] = parameters["tau"]  # + 2400000
    # radial_velocity(gamma, k, ta, omega, ecc)
    obs_jd = float(obs_hdr["MJD-OBS"])  # + 2400000
    ref_jd = float(ref_hdr["MJD-OBS"])  # + 2400000
    print(obs_jd)
    print(ref_jd)
    # obs_jd = datetime2jd(obs_time)
    # ref_jd = datetime2jd(ref_time)
    from warnings import warn
    warn.warning("There is a bug in RV_from_params! Its values are wrong.")
    obs_host_rv = RV_from_params(obs_jd, parameters, companion=False, ignore_mean=False)
    ref_host_rv = RV_from_params(ref_jd, parameters, companion=False, ignore_mean=False)
    obs_host_rv_sys = RV_from_params(obs_jd, parameters, companion=False, ignore_mean=True)
    ref_host_rv_sys = RV_from_params(ref_jd, parameters, companion=False, ignore_mean=True)
    print("Checking the RV difference = {}km/s".format(ref_host_rv - obs_host_rv))
    print("Checking the RV difference = {}km/s".format(ref_host_rv_sys - obs_host_rv_sys))

    # Hack for adding k2
    if "k2" in parameters.keys():
        pass
    else:
        if ('m_true' in parameters.keys()):
            # Use true mass if given
            if not parameters["m_true"] == "":
                mass_used = parameters['m_true']
                true_mass_flag = True  # parameter to indicate if the true mass was used or not
            else:
                mass_used = parameters["msini"]
                true_mass_flag = False
        else:
            mass_used = parameters["msini"]
            true_mass_flag = False

        parameters['k2'] = companion_amplitude(parameters['k1'],
                                               parameters['m_star'],
                                               mass_used)
        parameters["true_mass_flag"] = true_mass_flag  # True if true mass used

    t = np.linspace(parameters["tau"], parameters["tau"] + 2 * parameters["period"], 200)
    plt.plot(t, RV_from_params(t, parameters, companion=True, ignore_mean=True))
    plt.plot(obs_jd, RV_from_params(obs_jd, parameters, companion=True, ignore_mean=True), "ro")
    plt.plot(ref_jd, RV_from_params(ref_jd, parameters, companion=True, ignore_mean=True), "k*")
    plt.title("check rv curve")
    plt.close()
    # plt.show()

    print("true_mass_flag", parameters["true_mass_flag"])
    obs_comp_rv_sys = RV_from_params(obs_jd, parameters, companion=True, ignore_mean=True)
    ref_comp_rv_sys = RV_from_params(ref_jd, parameters, companion=True, ignore_mean=True)
    print("Checking the Companion RV difference sys = {}km/s".format(ref_comp_rv_sys - obs_comp_rv_sys))
    obs_comp_rv = RV_from_params(obs_jd, parameters, companion=True, ignore_mean=False)
    ref_comp_rv = RV_from_params(ref_jd, parameters, companion=True, ignore_mean=False)
    print("Checking the Companion RV difference with gamma = {}km/s".format(ref_comp_rv - obs_comp_rv))

    # obs_I_old, ref_I_old = obs_I, ref_I
    obs_I, __ = pyasl.dopplerShift(obs_wl, obs_I, -obs_host_rv)
    ref_I, __ = pyasl.dopplerShift(ref_wl, ref_I, -ref_host_rv)
    obs_I, __ = pyasl.dopplerShift(obs_wl, obs_I, + ref_host_rv_sys - obs_host_rv_sys)
    # obs_I, __ = barycorr_crires(obs_wl, obs_I, obs_hdr, extra_offset=None)
    # ref_I, __ = barycorr_crires(ref_wl, ref_I, ref_hdr, extra_offset=None)
    print("RV values - ")
    print("obs_host_rv", obs_host_rv)
    print("ref_host_rv", ref_host_rv)
    print("obs_host_rv_sys", obs_host_rv_sys)
    print("ref_host_rv_sys", ref_host_rv_sys)
    print("obs_comp_rv_sys", obs_comp_rv_sys)
    print("ref_comp_rv_sys", ref_comp_rv_sys)
    print("obs_comp_rv", obs_comp_rv)
    print("ref_comp_rv", ref_comp_rv)

    # Match wavelengths to reference spectrum
    obs_interp = interp1d(obs_wl, obs_I, kind="linear", bounds_error=False, fill_value="extrapolate")
    new_obs_I = obs_interp(ref_wl)
    # new_obs_I = wl_interpolation(obs_wl, obs_I, ref_wl)

    I_diff = ref_I - new_obs_I  # This fixed the bug and removed stellar lines very well!!!!
    print(ref_I)
    print(I_diff)
    plt.figure()
    plt.plot(ref_wl, I_diff)
    plt.hlines(0, np.min(ref_wl), np.max(ref_wl), colors='k', linestyles='dashed', label='')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Recovered Difference")
    plt.title("Difference between {0} and {1}".format(target, reference_target))
    plt.show()

    # Spectrum Simulation
    pathwave = "/home/jneal/Phd/data/phoenixmodels/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    specpath = "/home/jneal/Phd/data/phoenixmodels/HD30501b-lte02500-5.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    w_mod = fits.getdata(pathwave)
    w_mod /= 10  # turn into nm
    mod_flux = fits.getdata(specpath)
    # mod_hdr = fits.getheader(specpath)

    mask = (w_mod < 2150) & (w_mod > 2100)
    w_mod = w_mod[mask]
    mod_flux = mod_flux[mask]

    # Conlvove to 50000
    mod_flux = pyasl.instrBroadGaussFast(w_mod, mod_flux, 50000, edgeHandling=None, fullout=False, maxsig=None)
    # Normalize
    mod_flux /= np.median(mod_flux)
    maxes = mod_flux.argsort()[-100:][::-1]
    mod_flux /= np.median(mod_flux[maxes])
    mod_flux[mod_flux < 0] = 0

    print("Difference used in model shift {0}".format(obs_comp_rv_sys - ref_comp_rv_sys))
    nflux_rvobs, __ = pyasl.dopplerShift(w_mod, mod_flux, obs_comp_rv_sys - ref_comp_rv_sys, edgeHandling=None,
                                         fillValue=None)
    nflux_rvref, __ = mod_flux, __
    mod_diff = (nflux_rvobs - nflux_rvref) * 0.01

    mask = (w_mod < 2123.5) & (w_mod > 2112)
    w_mod = w_mod[mask]
    mod_diff = mod_diff[mask]

    mask_value = 0.96
    wl = wl[~np.isnan(I)]
    I = I[~np.isnan(I)]

    ref_wl = ref_wl[~np.isnan(ref_I)]
    I_diff = I_diff[~np.isnan(ref_I)]
    ref_I = ref_I[~np.isnan(ref_I)]

    tap_wl = tap_wl[~np.isnan(tap_I)]
    tap_I = tap_I[~np.isnan(tap_I)]

    star_mask = ref_I < mask_value
    tapas_mask = tap_I < mask_value

    @styler
    def paper_plot(fig, *args, **kwargs):
        ax = fig.add_subplot(311)
        ax.plot(wl, I, label="Observed")
        ax.plot(tap_wl, tap_I, label="Telluric", linestyle="-.")
        ax.set_ylabel("Norm\nFlux")
        ax.fill_between(ref_wl, 0.5 * np.ones_like(ref_wl), 1.1 * np.ones_like(ref_wl), where=star_mask, alpha=0.8,
                        color=colors_named["lightblue"], linewidth=0)
        ax.fill_between(tap_wl, 0.5 * np.ones_like(tap_wl), 1.1 * np.ones_like(tap_wl), where=tapas_mask, alpha=0.8,
                        color=colors_named["green"], linewidth=0)
        plt.legend()
        # ax.set_xlim = [min(ref_wl), max(ref_wl)]

        ax = fig.add_subplot(312)
        ax.plot(ref_wl, ref_I, label="Telluric Corrected")
        ax.hlines(1, np.min(ref_wl), np.max(ref_wl), colors='k', linestyles='dashed', label='', alpha=0.5)
        # plt.xlabel("Wavelength (nm)")
        ax.set_ylabel("Norm\nFlux")
        # ax.set_title("Telluric Correctd")
        # ax.set_xlim = [min(ref_wl), max(ref_wl)]
        ax.fill_between(ref_wl, 0.5 * np.ones_like(ref_wl), 1.1 * np.ones_like(ref_wl), where=star_mask, alpha=0.8,
                        color=colors_named["lightblue"], linewidth=0)
        ax.fill_between(tap_wl, 0.5 * np.ones_like(tap_wl), 1.1 * np.ones_like(tap_wl), where=tapas_mask, alpha=0.8,
                        color=colors_named["green"], linewidth=0)
        plt.legend()

        ax = fig.add_subplot(313)
        ax.plot(ref_wl, I_diff, label="Observed")
        ax.plot(w_mod, mod_diff, label="Simulated", linestyle="-.")
        # ax.hlines(0, np.min(ref_wl), np.max(ref_wl), colors='k', linestyles='dashed', alpha=0.5)
        plt.legend()
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(r"$\Delta$ Flux")
        ax.fill_between(ref_wl, -0.03 * np.ones_like(ref_wl), 0.02 * np.ones_like(ref_wl), where=star_mask, alpha=0.8,
                        color=colors_named["lightblue"], linewidth=0)
        ax.fill_between(tap_wl, -0.03 * np.ones_like(tap_wl), 0.02 * np.ones_like(tap_wl), where=tapas_mask, alpha=0.8,
                        color=colors_named["green"], linewidth=0)
        # ax.set_title("Difference between {0} and {1}".format(target, reference_target))
        # ax.set_xlim = [min(ref_wl), max(ref_wl)]

    fig = plt.figure()
    paper_plot(fig, type='A&A', save='New_Spectrum_and_diff_simulated_{0}.pdf'.format(_id), tight=True,
               figsize=(None, 1))


if __name__ == "__main__":
    # Parameters to alter to change spectra seen
    star = "HD30501"
    chip_num = 1
    obs_num = "1"
    ref_num = "3"
    use_h2o = True
    assert obs_num != ref_num, "Same object selected to compare."
    main(star, chip_num, obs_num, ref_num, use_h2o)
    print("done")

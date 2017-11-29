import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mingle.utilities.crires_utilities import barycorr_crires
from mingle.utilities.spectrum_utils import load_spectrum
from spectrum_overload import Spectrum

import TelluricSpectra.Obtain_Telluric as obt
from TelluricSpectra.TellRemoval import new_export_correction_2fits

base_dir = os.path.join("/", "home", "jneal", "Phd", "data", "Crires", "BDs-DRACS", "2017")


def decompose_parameters(fname):
    path, name = os.path.split(fname)
    star = name.split("-")[0]
    obsnum = name.split("-")[1]
    chip = name.split(".fits")[0][-1]
    return star, obsnum, chip


def main(fname, apply_berv=False, export=False, show=False):
    star, obsnum, chip = decompose_parameters(fname)
    observation = load_spectrum(fname)
    path = os.path.join(base_dir, "{0}-{1}".format(star, obsnum))

    # Find telluric file
    telluric_file = glob.glob(path + "/Telluric_files*/*_10_*.ipac")
    assert len(telluric_file) == 1
    teluric, tell_header = obt.load_telluric("", telluric_file[0])
    tell_spec = Spectrum(xaxis=teluric[0], flux=teluric[1], header=observation.header)

    if apply_berv:
        wlprime_obs, __ = barycorr_crires(observation.xaxis, observation.flux, observation.header)
        wlprime_tell, __ = barycorr_crires(tell_spec.xaxis, tell_spec.flux, observation.header)
        observation.xaxis = wlprime_obs
        tell_spec.xaxis = wlprime_tell

    observation.remove_nans()
    tell_spec.remove_nans()
    assert len(tell_spec.xaxis) != len(observation.xaxis)
    tell_spec.spline_interpolate_to(observation)

    assert len(tell_spec.xaxis) == len(observation.xaxis)
    assert np.allclose(tell_spec.xaxis, observation.xaxis)
    mask_value = 0.02
    tell_mask = tell_spec.flux < 1 - mask_value  # 5%

    masked_obs = Spectrum(xaxis=observation.xaxis[tell_mask], flux=observation.flux[tell_mask])
    masked_tell = Spectrum(xaxis=tell_spec.xaxis[tell_mask], flux=tell_spec.flux[tell_mask])

    maskedin_obs = Spectrum(xaxis=observation.xaxis[~tell_mask], flux=observation.flux[~tell_mask])
    maskedin_tell = Spectrum(xaxis=tell_spec.xaxis[~tell_mask], flux=tell_spec.flux[~tell_mask])

    pixels_removed = sum(tell_mask)
    fraction_removed = pixels_removed / len(observation.xaxis)

    if show:
        observation.plot(label="obs")
        tell_spec.plot(label="telluric")
        maskedin_obs.plot(label="masked obs >5", color="r", linestyle="", marker=".")
        plt.legend()
        plt.title("Masking Telluric lines deeper than {}".format(mask_value))
        plt.show()

    if export:
        obs_name = fname.replace(".fits", "_bervd.fits")
        new_export_correction_2fits(obs_name, observation.xaxis, observation.flux, observation.header, ["BervDone"],
                                    [True],tellhdr=tell_spec.header)

        tell_name = "{}-{}_berved_telluric_model_{}.fits".format(star, obsnum, chip)
        new_export_correction_2fits(tell_name, tell_spec.xaxis, tell_spec.flux, tell_spec.header, ["BervDone"], [True],tellhdr=tell_spec.header)

        obs_mask = fname.replace(".fits", "_bervd_tellmasked.fits")
        new_export_correction_2fits(obs_mask, maskedin_obs.xaxis, maskedin_obs.flux, observation.header,
                                    ["BervDone", "Tellmask", "pix_mask", "ratio_mask"],
                                    [True, (mask_value, "telluric line depth limit"),
                                     (pixels_removed, "Number of pixels masked out"),
                                     (fraction_removed, "Fraction of pixels masked out")],tellhdr=tell_spec.header)


def parse_args(args):
    """Take care of all the argparse stuff.

    :returns: the args
    """
    parser = argparse.ArgumentParser(description='Telluric Removal')
    parser.add_argument('fname', help='Input fits file')
    parser.add_argument('-b', '--apply_berv', action='store_true',
                        help='Add berv correction to spectra and maskfile.')
    parser.add_argument('-e', '--export', action='store_true',
                        help='Export/save results to fits file')
    parser.add_argument("-s", "--show", action='store_true',
                        help="Show plots")  # Does not work without a display

    return parser.parse_args(args)


if __name__ == "__main__":
    args = vars(parse_args(sys.argv[1:]))
    fname = args.pop('fname')
    opts = {k: args[k] for k in args}

    main(fname, **opts)

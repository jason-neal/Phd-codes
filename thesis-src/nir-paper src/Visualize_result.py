"""Visualize result for paper"""

import matplotlib.pyplot as plt
import numpy as np
from mingle.models.broadcasted_models import inherent_alpha_model
from mingle.utilities.db_utils import SingleSimReader, DBExtractor
# from mingle.utilities.errors import spectrum_error
from mingle.utilities.masking import spectrum_masking
from mingle.utilities.phoenix_utils import load_starfish_spectrum
from mingle.utilities.spectrum_utils import load_spectrum
from simulators.iam_module import iam_helper_function
from spectrum_overload import Spectrum

from HD211847_result_contour import base, chi2_val
from styler import styler

star = "HD211847"
obsnum = 2

sim_example = SingleSimReader(base=base, name="HD211847", mode="iam", chi2_val=chi2_val, obsnum=2, suffix="*")

extractor = DBExtractor(sim_example.get_table())

df_min = extractor.minimum_value_of(chi2_val)

cols = ['teff_2', 'logg_2', 'feh_2', 'rv', 'gamma',
        chi2_val, 'teff_1', 'logg_1', 'feh_1']

teff_1 = df_min["teff_1"].values[0]
teff_2 = df_min["teff_2"].values[0]
logg_1 = df_min["logg_1"].values[0]
logg_2 = df_min["logg_2"].values[0]
gamma = df_min["gamma"].values[0]
rv = df_min["rv"].values[0]
feh_1 = 0.0
feh_2 = 0.0
print(teff_1, teff_2, logg_1, logg_2, gamma, rv, )
print("df min", df_min)


@styler
def f(fig, *args, **kwargs):
    for chip in range(1, 5):
        plt.subplot(2, 2, chip)
        ax = plt.gca()
        # Get observation data
        obs_name, params, output_prefix = iam_helper_function(star, obsnum, chip)

        # Load observed spectrum
        obs_spec = load_spectrum(obs_name)

        # Mask out bad portion of observed spectra
        obs_spec = spectrum_masking(obs_spec, star, obsnum, chip)
        obs_spec_masked = spectrum_masking(obs_spec, star, obsnum, chip, stricter=True)

        # Create model with given parameters
        host = load_starfish_spectrum([teff_1, logg_1, feh_1],
                                      limits=[2110, 2165], area_scale=True, hdr=True)
        if teff_2 is None:
            assert (logg_2 is None) and (feh_2 is None) and (rv == 0), "All must be None for bhm case."
            companion = Spectrum(xaxis=host.xaxis, flux=np.zeros_like(host.flux))
        else:
            companion = load_starfish_spectrum([teff_2, logg_2, feh_2],
                                               limits=[2110, 2165], area_scale=True, hdr=True)

        joint_model = inherent_alpha_model(host.xaxis, host.flux,
                                           companion.flux, gammas=gamma, rvs=rv)

        # Assuming 3200K is correct
        assumed_companion = load_starfish_spectrum([3200, logg_2, feh_2],
                                               limits=[2110, 2165], area_scale=True, hdr=True)
        assummed_model = inherent_alpha_model(host.xaxis, host.flux,
                                           assumed_companion.flux, gammas=gamma, rvs=rv)

        model_spec = Spectrum(xaxis=host.xaxis, flux=joint_model(host.xaxis).squeeze())
        model_spec = model_spec.remove_nans()
        model_spec = model_spec.normalize("exponential")

        assummed_spec = Spectrum(xaxis=host.xaxis, flux=assummed_model(host.xaxis).squeeze())
        assummed_spec = assummed_spec.remove_nans()
        assummed_spec = assummed_spec.normalize("exponential")

        lw = kwargs.get("lw", 1)

        # Spectrum chunks
        wave_diffs = np.diff(obs_spec.xaxis)
        indx = np.where(wave_diffs > 0.05)[0] + 1
        if len(indx) > 0:
            wave_chunks = np.split(obs_spec.xaxis, indx)
            flux_chunks = np.split(obs_spec.flux, indx)
        else:
            wave_chunks = [obs_spec.xaxis]
            flux_chunks = [obs_spec.flux]
        for ii, (wchunk, fchunk) in enumerate(zip(wave_chunks, flux_chunks)):
            # plt.plot(obs_spec.xaxis, obs_spec.flux + 0.05, label="HD211847-1", lw=0.6)
            label = "Observed" if ii == 0 else ""
            plt.plot(wchunk, fchunk, "C0", label=label, lw=lw)
        # obs_spec.plot(axis=ax, label="Observed", lw=lw)

        model_spec.plot(axis=ax, linestyle="--", label="Recovered", lw=lw, color="C1")
        assummed_spec.plot(axis=ax, linestyle="--", label="$Teff_2$=3200K", lw=lw, color="C2")
        ax.set_xlim([obs_spec.xmin() - 0.5, obs_spec.xmax() + 0.5])

        # ax.set_title("Detector {}".format(chip))
        plt.annotate("Detector {}".format(chip), (.8, 0.05), xycoords='axes fraction')

        # Set xaxis limits
        xticks = ax.get_xticks()
        ax.set_xticks(xticks[xticks % 2 == 0])  # multiple of 2 axis labels
        ax.set_xlim([obs_spec.xmin() - 0.5, obs_spec.xmax() + 0.5])

        if chip in [3, 4]:
            plt.xlabel("Wavelength (nm)")
        if chip in [1, 3]:
            plt.ylabel("Normalized Flux")

        # Masking
        wave_diffs = np.diff(obs_spec_masked.xaxis)
        indx = np.where(wave_diffs > 0.05)[0] + 1
        if len(indx) > 0:
            wave_chunks = np.split(obs_spec_masked.xaxis, indx)
        else:
            wave_chunks = [obs_spec_masked.xaxis]

        if len(wave_chunks) > 1:
            for ii in range(len(wave_chunks) - 1):
                mask_label = "Excluded" if ii == 0 else ""

                plt.axvspan(wave_chunks[ii][-1], wave_chunks[ii + 1][0], facecolor="k", alpha=0.3, label=mask_label)
            if not np.allclose(obs_spec.xaxis[0], wave_chunks[0][0]):
                plt.axvspan(obs_spec.xaxis[0], wave_chunks[0][0], facecolor="k", alpha=0.3, label=mask_label)
            if not np.allclose(obs_spec.xaxis[-1], wave_chunks[-1][-1]):
                plt.axvspan(wave_chunks[-1][-1], obs_spec.xaxis[-1], facecolor="k", alpha=0.3, label=mask_label)
        elif len(wave_chunks[0]) > 0:
            mask_label = "Excluded"
            if not np.allclose(obs_spec.xaxis[0], obs_spec_masked.xaxis[0]):
                plt.axvspan(obs_spec.xaxis[0], obs_spec_masked.xaxis[0], facecolor="k", alpha=0.3, label=mask_label)
                mask_label = ""  # Reset if been through this conditional
            if not np.allclose(obs_spec.xaxis[-1], obs_spec_masked.xaxis[-1]):
                print("boo2")
                plt.axvspan(obs_spec_masked.xaxis[-1], obs_spec.xaxis[-1], facecolor="k", alpha=0.3, label=mask_label)

        else:
            mask_label = "Excluded"
            plt.axvspan(obs_spec.xaxis[0], obs_spec.xaxis[-1], facecolor="k", alpha=0.3, label=mask_label)

        ax.legend()
        print("Number of pixels in masked section", len(obs_spec_masked), "chip=", chip)
    plt.tight_layout()
    # plt.show()


# f(type="two", tight=True, dpi=500, figsize=(None, .70), axislw=0.5, formatcbar=False, formatx=False, formaty=False,
#     grid=True)
f(type="two", tight=True, dpi=500, save="../final/visualize_result.pdf", figsize=(None, .70), axislw=0.5,
  formatcbar=False, formatx=False, formaty=False, grid=False)
print("Done")

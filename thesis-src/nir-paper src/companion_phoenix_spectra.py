import matplotlib.pyplot as plt
import numpy as np
import simulators
from simulators.iam_module import prepare_iam_model_spectra, inherent_alpha_model
from spectrum_overload.spectrum import Spectrum
from styler import styler

area_scale = False
wav_scale = False
# area_scale = True
wav_scale = True

if "CIFIST" in simulators.starfish_grid["hdf5_path"]:
    phoenix = "BT-SETTL"
else:
    phoenix = "ACES"
rv_limits = [[2112.5418, 2123.4991], [2127.7725, 2137.4103],
             [2141.9468, 2151.7606], [2155.2510, 2164.9610]]

host_temp = 5500
comp_temps = np.array([2500, 3000, 3500, 3800, 4000, 4500, 5000])
gammas = [0]
rvs = [50]
obs_spec = np.linspace(2111, 2163, 6072)


@styler
def f(fig, *args, **kwargs):
    lw = kwargs.get("lw", 1)
    ax = plt.subplot(111)
    for ii, ctemp in enumerate(comp_temps):
        _, mod2_spec = prepare_iam_model_spectra([host_temp, 4.5, 0], [ctemp, 5.0, 0.0], limits=(2111, 2166),
                                                 area_scale=area_scale, wav_scale=area_scale)
        mod2_spec /= 1e6
        mod2_spec.plot(axis=ax, label="{} K".format(ctemp), lw=lw)

    for limits in rv_limits:
        plt.axvline(limits[0], color="black", alpha=0.8, ls="--", lw=lw)
        plt.axvline(limits[-1], color="black", alpha=0.8, ls="--", lw=lw)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Flux ($10^6$ erg/s/cm$^2$/cm)")
    plt.show()


if __name__ == "__main__":
    f(type="one", tight=True, dpi=500, formatcbar=False, formatx=False,
      save="../final/companion_spectra.pdf", axislw=0.5, formaty=False,
      grid=True, lw=0.8, figsize=(None, .80))
    import os

    print("saved", os.getcwd(), "../final/companion_spectra.pdf")

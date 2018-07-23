import itertools
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
# Load in the injection recovery files and plot
from baraffe_tables.teff2mass import main as temp2mass
from matplotlib import rc

from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

comp_logg = 4.5
error = None
# fname = f"/home/jneal/Phd/Analysis/injector/noise test/test_results_logg={comp_logg}_error={error}.txt"
# fname_template = "/home/jneal/Phd/Analysis/injection/{}_injector_results_logg={}_error={}.txt"
fname_template = "./data/{}_injector_results_logg={}_error={}.txt"

colors = ["C0", "C1", "C2", "C3", "C4"]
markers = [".", "^", "o", "*", ">"]


def temp_2jupmass(temp: Union[int, float], logg: float):
    result = temp2mass(temp, logg, plot=False)
    return result["M/Mjup"]


def temp_2_age(temp: Union[int, float], logg: float):
    result = temp2mass(temp, logg, plot=False)
    return result["age"]


fname1 = fname_template.format("HD211847", 5.0, None)
input_, _, _, _ = np.loadtxt(fname1, skiprows=2, unpack=True)

input_5000 = input_[input_ <= 5000]
temp_err = 100 * np.ones_like(input_)
masses4p5 = [temp_2jupmass(t, 4.5) for t in input_5000]
masses5 = [temp_2jupmass(t, 5.0) for t in input_5000]
print("4.5", masses4p5)
print("5", masses5)
ages = [temp_2_age(t, comp_logg) for t in input_5000]

# targets = ("HD211847", "HD30501")
targets = ("HD211847",)
loggs = (4.5, 5.0)


@styler
def f(fig, *args, **kwargs):
    # plot the injection-recovery
    show_mass = kwargs.get("show_mass", False)
    ms = kwargs.get("ms", 2)
    lw = kwargs.get("lw", 1)
    # ax1 = plt.subplot(211)
    ax1 = plt.subplot(111)

    for ii, (star, logg) in enumerate(
            itertools.product(targets, loggs)):
        print(ii)
        fname = fname_template.format(star, logg, error)
        input_, output, rv1, rv2 = np.loadtxt(fname, skiprows=2, unpack=True)
        # temp_err = 100 * np.ones_like(input_)
        output[output >= 5000] = np.nan
        assert len(input_) == len(temp_err)
        ax1.errorbar(input_, output, yerr=temp_err, marker=markers[ii], color=colors[ii], ms=ms, lw=lw, ls="",
                     label="logg = {0}".format(logg))
    ax1.plot(input_, input_, "k--", alpha=0.7, lw=lw)
    plt.xlabel("Injected Temperature (K)")
    ax1.set_ylabel("Recovered Temperature (km/s)")
    ax1.set_xlim(2450, 5050)
    ax1.legend(title="Companion")

    if show_mass:
        ax2 = plt.twinx(ax1)
        ax2.plot(input_5000, masses4p5, "g-.", lw=lw, label="logg = 4.5")
        ax2.plot(input_5000, masses5, "m:", lw=lw, label="logg = 5.0")
        ax2.set_ylabel("Mass (M$_{Jup}$)")
        ax2.set_xlim(2450, 5050)

        ax2.axhline(temp_2jupmass(3600, 4.5), xmin=place_line(ax1, 3600, "x"), alpha=0.7, color="C0", lw=lw)
        ax2.axhline(temp_2jupmass(3900, 5), xmin=place_line(ax2, 3900, "x"), alpha=0.7, color="C1", lw=lw)
        ax2.axvline(3600, ymax=place_line(ax2, temp_2jupmass(3600, 4.5), "y"),
                    alpha=0.7, color="C0", lw=lw)
        ax2.axvline(3900, ymax=place_line(ax2, temp_2jupmass(3900, 5), "y"),
                    alpha=0.7, color="C1", lw=lw)

        plt.legend(loc=4, title="Evolution Models")
    plt.tight_layout()
    # plt.show()


def temp_2jupmass(temp: Union[int, float], logg: float):
    result = temp2mass(temp, logg, plot=False)
    return result["M/Mjup"]


def temp_2_age(temp: Union[int, float], logg: float):
    result = temp2mass(temp, logg, plot=False)
    return result["age"]


def place_line(ax, value, axis="y"):
        """Calculate axis fraction to place line"""
        if axis == "y":
            bottom, top = ax.get_ylim()
        elif axis == "x":
            bottom, top = ax.get_xlim()
        else:
            raise ValueError("axis=({}) is not 'x' or 'y'".format(axis))
        fraction = (value - bottom) / (top - bottom)
        return fraction


if __name__ == "__main__":
    # f(type="one", tight=True, dpi=400, figsize=(None, .70),
    #   axislw=0.5, ms=1, lw=0.7,
    #   formatcbar=False, formatx=False, formaty=False)
    show_mass = False
    f(type="one", tight=True, dpi=400, figsize=(None, .70),
      axislw=0.5, ms=1.8, lw=0.7, save="../final/inject_recovery.pdf",
      formatcbar=False, formatx=False, formaty=False, show_mass=show_mass)
    print("Done")

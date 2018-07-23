"""HD211847 Example companion of sun like star"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from mingle.utilities.chisqr import chi2_at_sigma
from mingle.utilities.db_utils import SingleSimReader, DBExtractor
from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

chi2_val = "chi2_123"

# dir_base = "/home/jneal/Phd/Analysis/injection/INJECTORSIMS_analysis"
# dir_base = "/home/jneal/Phd/Writing-in-Progress/nir-paper/images/src/data/injector_shape/analysis"
dir_base = "./data/injector_shape/analysis"

ms = 10


@styler
def f(fig, *args, **kwargs):
    name = kwargs.get("name")
    comp_temp = kwargs.get("comp_temp", False)
    height_ratios = kwargs.get("height_ratios", (3, 1))
    lw = kwargs.get("lw", 0.8)
    grid = kwargs.get("grid", False)
    print("comp_temp", comp_temp)
    injected_point = {"teff_1": 5800, "logg_2": 4.5, "feh_1": 0.0, "gamma": 0,
                      "teff_2": comp_temp, "logg_2": 5.0, "feh_2": 0.0, "rv": 100, "obsnum": 1}

    kwargs = {"correct": injected_point, "dof": 4, "grid": grid, "ms": ms, "sigma": [3]}

    ##################
    # ### IAM RESULTS
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": height_ratios})

    for comp_temp in [2500, 3000, 3500, 3800, 4000, 4500, 5000]:
        try:
            name = f"INJECTORSIMS{comp_temp}"
            sim_example = SingleSimReader(base=dir_base, name=name, mode="iam", suffix="*", chi2_val=chi2_val)
            extractor = DBExtractor(sim_example.get_table())
            df_min = extractor.minimum_value_of(chi2_val)
            df = extractor.simple_extraction(columns=["teff_2", chi2_val])
            chi2s = []
            teffs = []
            for teff_2 in df["teff_2"]:
                teffs.append(teff_2)
                chi2s.append(np.min(df[df["teff_2"] == teff_2][chi2_val]))

            ax1.plot(teffs, chi2s - min(chi2s), "-.", label=comp_temp, lw=lw)

            ax2.axvline(teffs[np.argmin(chi2s)], lw=lw, color="grey", alpha=0.9, linestyle="--")
            ax2.plot(teffs, chi2s - min(chi2s), "-.", label=comp_temp, lw=lw)
            # plt.legend(title="Injected", , fontsize="small")
        except:
            print("Failed with temp=", comp_temp)
    # ax1.set_xlabel("Recovered Temp (K)")
    ax1.set_ylabel("$\Delta \chi^2$")
    ax1.set_ylim(bottom=-50, top=5000)
    ax1.set_xlim(right=5300)
    ax1.legend(title="Injected", fontsize="small")

    sigma1 = chi2_at_sigma(1, dof=2)
    sigma2 = chi2_at_sigma(2, dof=2)
    sigma3 = chi2_at_sigma(3, dof=2)
    ax2.axhline(sigma2, linestyle="-", alpha=0.5, color="k", label="$2-\sigma$", lw=lw - 0.2)
    ax2.axhline(sigma1, linestyle="-", alpha=0.5, color="k", label="$1-\sigma$", lw=lw - 0.2)
    ax2.axhline(sigma3, linestyle="-", alpha=0.5, color="k", label="$3-\sigma$", lw=lw - 0.2)
    ax2.set_ylim(bottom=0, top=15)
    ax2.set_xlabel("Model Companion Temperature (K)")
    ax2.set_ylabel("$\Delta \chi^2$")

    ax1.axvline(2300, color="k", lw=lw)
    ax2.axvline(2300, color="k", lw=lw)
    plt.tight_layout()


if __name__ == "__main__":
    f(type="one", tight=True, dpi=500, save="../final/chi2_shape_investigation_sigmas.pdf", figsize=(None, .80),
      axislw=0.5, formatcbar=False, formatx=False, formaty=False, grid=True, lw=0.8)

    # Print errorbars
    # Get the errorbars
    sigma1 = chi2_at_sigma(1, dof=2)
    sigma2 = chi2_at_sigma(2, dof=2)
    for comp_temp in [2500, 3000, 3500, 3800, 4000, 4500, 5000]:
        chi2s = []
        teffs = []
        try:
            name = f"INJECTORSIMS{comp_temp}"
            sim_example = SingleSimReader(base=dir_base, name=name, mode="iam", suffix="*", chi2_val=chi2_val)
            extractor = DBExtractor(sim_example.get_table())
            df_min = extractor.minimum_value_of(chi2_val)
            df = extractor.simple_extraction(columns=["teff_2", chi2_val])
            chi2s = []
            teffs = []
            sorted_teff_set = list(set(df["teff_2"]))
            sorted_teff_set.sort()
            for teff_2 in sorted_teff_set:
                teffs.append(teff_2)
                this_min_chi2 = np.min(df[df["teff_2"] == teff_2][chi2_val])
                chi2s.append(this_min_chi2)

            from scipy.optimize import fsolve
            from scipy.interpolate import interp1d

            value = teffs[np.argmin(chi2s)]
            ts = np.linspace(value - 300, value + 300, 250)
            value = teffs[np.argmin(chi2s)]
            chi2_func = interp1d(teffs, chi2s, kind="cubic",
                                 bounds_error=False, fill_value=5000)
            root_func2 = interp1d(teffs, chi2s - sigma1 - np.min(chi2_func(ts)), kind="cubic",
                                  bounds_error=False, fill_value=5000)

            min_t_value = ts[np.argmin(chi2_func(ts))]
            lower_root = fsolve(root_func2, x0=min_t_value - 100)
            upper_root = fsolve(root_func2, x0=min_t_value + 10)

            print(f"{comp_temp} result = {min_t_value:5.0f}\t+{(upper_root-min_t_value)[0]:5.01f}\t{(lower_root-min_t_value)[0]:5.01f} ")

            plt.plot(ts, root_func2(ts) + sigma1, label="minimizer")
            plt.plot(teffs, chi2s - min(chi2s), label="values")
            plt.axhline(sigma1, linestyle=":", color="m", label="sigma1")
            plt.axhline(sigma2, linestyle="--", label="sigma2")
            plt.plot(value, 0, "rx")
            plt.plot(upper_root, sigma1, "r<")
            plt.plot(lower_root, sigma1, "m>")
            plt.xlabel("Temp")
            plt.ylabel("$\chi^2$")
            plt.ylim([np.min(root_func2(ts) + sigma1), 20])
            plt.xlim([value - 200, value + 200])
            plt.legend()
            plt.show()

        except:
            print(teffs, chi2s)
            raise

print("Done")

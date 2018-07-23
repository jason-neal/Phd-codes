"""HD211847 Example companion of sun like star"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
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
    grid = kwargs.get("grid", False)
    print("comp_temp", comp_temp)
    injected_point = {"teff_1": 5800, "logg_2": 4.5, "feh_1": 0.0, "gamma": 0,
                      "teff_2": comp_temp, "logg_2": 5.0, "feh_2": 0.0, "rv": 100, "obsnum": 1}

    kwargs = {"correct": injected_point, "dof": 4, "grid": grid, "ms": ms, "sigma": [3]}

    ##################
    # ### IAM RESULTS
    for comp_temp in [3000, 3500, 3800, 4000, 4500]:
        name = f"INJECTORSIMS{comp_temp}"
        sim_example = SingleSimReader(base=dir_base, name=name, mode="iam", suffix="*", chi2_val=chi2_val)

        extractor = DBExtractor(sim_example.get_table())

        df_min = extractor.minimum_value_of(chi2_val)
        # print(df_min)
        df = extractor.simple_extraction(columns=["teff_2", chi2_val])
        # df = extractor.ordered_extraction(order_by=chi2_val, columns=["teff_2", chi2_val], limit=1)
        # df = extractor.fixed_extraction(columns=["teff_2", chi2_val], limit=1)

        # print(df.head)
        chi2s = []
        teffs = []
        for teff_2 in df["teff_2"]:
            teffs.append(teff_2)
            chi2s.append(np.min(df[df["teff_2"] == teff_2][chi2_val]))

        # plt.subplot(211)
        # plt.plot(df["teff_2"], df["chi2_123"], ".", label=comp_temp)
        plt.plot(teffs, chi2s, "-.", label=comp_temp, lw=1)
        plt.xlabel("Recovered Temp (K)")
        plt.ylabel("$\chi^2$")
        plt.legend(title="Injected", loc="outside_right", fontsize="small")


if __name__ == "__main__":
    f(type="one", tight=True, dpi=500, save="../final/chi2_shape_investigation.pdf", figsize=(None, .70),
      axislw=0.5, formatcbar=False, formatx=False, formaty=False, grid=True, lw=1.5)
print("Done")

from mingle.utilities.param_file import parse_paramfile

"""HD211847 Example companion of sun like star"""

import matplotlib.pyplot as plt

from matplotlib import rc
from mingle.utilities.db_utils import SingleSimReader, DBExtractor, df_contour
import os
from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

starname = "HD211847"
rvs = [6.6137, 6.6137]
gammas = [7.171, 7.167]

chi2_val = "chi2_123"

# base = "/home/jneal/Phd/Analysis/Paper_results/paper-{}_wider_rv".format(starname)
base = "/home/jneal/Phd/Analysis/Paper_results/paper-{}_wavmask2_more-host_tmp".format(starname)
base = "/home/jneal/Phd/Analysis/Paper_results/paper-{}_wavmask2_more-host_tmp_rv2".format(starname)
base = "/home/jneal/Phd/Analysis/Paper_results/paper-{}_wavmask2_more-host_tmp_rv3".format(starname)
base = "/home/jneal/Phd/Analysis/Paper_results/paper_{}_lowerrvbound2".format(starname)

# base = "/home/jneal/Phd/Analysis/Paper_results/paper-{}_wavmask2".format(starname)
# base = "/home/jneal/Phd/Analysis/Paper_results/paper-{}_wavmask2".format(starname)


assert os.path.exists(base)
correct = parse_paramfile("/home/jneal/Phd/data/parameter_files/HD211847_params.dat")
correct.update({"teff_1": correct["temp"], "teff_2": correct["comp_temp"],
                "logg_1": correct["logg"], "logg_2": correct["comp_logg"],
                "feh_1": correct["fe_h"], "feh_2": correct["comp_fe_h"],
                "gamma": 6.613, "rv": 7.169 - 6.613})
errors = {"teff_1": 24, "teff_2": None,
          "logg_1": None, "logg_2": None,
          "feh_1": None, "feh_2": None,
          "gamma": 0.306, "rv": 1.968}
ms = 10


# grid = True


@styler
def f(fig, *args, **kwargs):
    ### BHM ANALYSIS
    grid = kwargs.get("grid", False)
    bhm_sim_example = SingleSimReader(base=base, name="HD211847", mode="bhm", chi2_val=chi2_val, obsnum=2, suffix="*")

    bhm_extractor = DBExtractor(bhm_sim_example.get_table())

    bhm_df_min = bhm_extractor.minimum_value_of(chi2_val)
    print("HD211847 single Result min values.")
    print(bhm_df_min.head())
    cols = ['gamma', chi2_val, 'teff_1', 'logg_1', 'feh_1']

    fixed = {key: bhm_df_min[key].values[0] for key in ["logg_1", "feh_1"]}

    bhm_df = bhm_extractor.fixed_extraction(cols, fixed, limit=-1)
    kwargs = {"correct": correct, "dof": 2, "grid": grid, "ms": ms, "errorbars": errors, "sigma": [3]}
    plt.subplot(221)
    df_contour(bhm_df, "teff_1", "gamma", chi2_val, bhm_df_min, [],
               xlim=[5600, 6000], ylim=[5, 8.9], **kwargs)

    plt.annotate("$C^1$", (.01, 0.95), xycoords='axes fraction')

    ##################
    # ### IAM RESULTS
    sim_example = SingleSimReader(base=base, name="HD211847", mode="iam", chi2_val=chi2_val, obsnum=2, suffix="*")

    extractor = DBExtractor(sim_example.get_table())

    df_min = extractor.minimum_value_of(chi2_val)

    cols = ['teff_2', 'logg_2', 'feh_2', 'rv', 'gamma',
            chi2_val, 'teff_1', 'logg_1', 'feh_1']

    fixed = {key: df_min[key].values[0] for key in ["logg_1", "logg_2", "feh_1", "feh_2"]}

    df = extractor.fixed_extraction(cols, fixed, limit=-1)

    kwargs.update({"dof": 4})

    plt.subplot(222)
    df_contour(df, "teff_2", "teff_1", chi2_val, df_min, ["gamma", "rv"],
               xlim=[2800, 3800], ylim=[5600, 6000], **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')
    plt.subplot(223)
    df_contour(df, "teff_1", "gamma", chi2_val, df_min, ["teff_2", "rv"],
               xlim=[5600, 6000], ylim=[5, 8.9], **kwargs)
    #    xlim=[5600, 5900], ylim=[6, 7], **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')
    plt.subplot(224)
    df_contour(df, "teff_2", "rv", chi2_val, df_min, ["gamma", "teff_1"],
               xlim=[2800, 3800], ylim=[-20, 20], **kwargs)
    print("HD211847 binary Result min values.")
    print(df_min.head())
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')


if __name__ == "__main__":
    print("should had made plots by now")
    f(type="two", tight=True, dpi=400, save="../final/HD211847_result_contours.pdf", figsize=(None, .70), axislw=0.5,
      formatcbar=False, formatx=False, formaty=False, grid=False)
    f(type="two", tight=True, dpi=400, save="../final/HD211847_result_pcolors.pdf", figsize=(None, .70), axislw=0.5,
      formatcbar=False, formatx=False, formaty=False, grid=True)

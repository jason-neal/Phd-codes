"""HD211847 Example companion of sun like star"""

import matplotlib.pyplot as plt
from matplotlib import rc
from mingle.utilities.db_utils import SingleSimReader, DBExtractor, df_contour

from styler import styler

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

chi2_val = "chi2_123"

ex9_base = "/home/jneal/Phd/Analysis/Paper_examples/Example9_highres/analysis"
# ex8_base ="/home/jneal/Documents/data/Paper_examples/Example9/analysis"

correct9 = None
correct9 = {"teff_1": 5700, "logg_2": 4.5, "feh_1": 0.0, "gamma": 6.6,
            "teff_2": 3200, "logg_2": 5.0, "feh_2": 0.0, "rv": 0.5, "obsnum": 1}

ms = 10


# grid = True


@styler
def f(fig, *args, **kwargs):
    grid = kwargs.get("grid", False)
    ### BHM ANALYSIS

    # EX 9 bhm recovery  on iam sim. snr=150
    bhm_sim_example = SingleSimReader(base=ex9_base, name="Example9150RES", mode="bhm", suffix="*", chi2_val=chi2_val)

    bhm_extractor = DBExtractor(bhm_sim_example.get_table())

    bhm_df_min = bhm_extractor.minimum_value_of(chi2_val)
    print("HD211847 single example min values.")
    print(bhm_df_min.head())
    cols = ['gamma', chi2_val, 'teff_1', 'logg_1', 'feh_1']

    fixed = {key: bhm_df_min[key].values[0] for key in ["logg_1", "feh_1"]}

    bhm_df = bhm_extractor.fixed_extraction(cols, fixed, limit=-1)
    kwargs = {"correct": correct9, "dof": 2, "grid": grid, "ms": ms, "sigma": [3]}
    plt.subplot(221)
    df_contour(bhm_df, "teff_1", "gamma", chi2_val, bhm_df_min, [], xlim=[5600, 5800],
               ylim=[6, 6.95], **kwargs)
    plt.annotate("$C^1$", (.01, 0.95), xycoords='axes fraction')

    ##################
    # ### IAM RESULTS

    sim_example = SingleSimReader(base=ex9_base, name="Example9150RES", mode="iam", suffix="*", chi2_val=chi2_val)

    extractor = DBExtractor(sim_example.get_table())

    df_min = extractor.minimum_value_of(chi2_val)

    cols = ['teff_2', 'logg_2', 'feh_2', 'rv', 'gamma',
            chi2_val, 'teff_1', 'logg_1', 'feh_1']

    fixed = {key: df_min[key].values[0] for key in ["logg_1", "logg_2", "feh_1", "feh_2"]}

    df = extractor.fixed_extraction(cols, fixed, limit=-1)

    kwargs.update({"dof": 4})

    plt.subplot(222)
    df_contour(df, "teff_2", "teff_1", chi2_val, df_min, ["gamma", "rv"],
               xlim=[2800, 3600], ylim=[5600, 5800], **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')

    plt.subplot(223)
    df_contour(df, "teff_1", "gamma", chi2_val, df_min, ["teff_2", "rv"],
               xlim=[5600, 5800], ylim=[6, 6.95], **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')

    plt.subplot(224)
    df_contour(df, "teff_2", "rv", chi2_val, df_min, ["gamma", "teff_1"],
               xlim=[2800, 3600], ylim=[-3, 3], **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')
    print("HD211847 example min values.")
    print(df_min.head())


if __name__ == "__main__":
    f(type="two", tight=True, dpi=500, save="../final/HD211847_example_contours.pdf", figsize=(None, .70), axislw=0.5,
      formatcbar=False,
      formatx=False, formaty=False, grid=False)
    f(type="two", tight=True, dpi=500, save="../final/HD211847_example_pcolors.pdf", figsize=(None, .70), axislw=0.5,
      formatcbar=False,
      formatx=False, formaty=False, grid=True)

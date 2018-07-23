"""M-DWARF companion of sun like star"""

import matplotlib.pyplot as plt
from matplotlib import rc
from mingle.utilities.db_utils import SingleSimReader, DBExtractor, df_contour

from styler import styler

chi2_val = "chi2_123"
ex6_base = "/home/jneal/Phd/Analysis/Paper_examples/Example6_highres/analysis/"
# correct6 = {"teff_1": 5800, "logg_2": 4.5, "feh_1": 0.0, "gamma": -15,
#            "teff_2": 4000, "logg_2": 5.0, "feh_2": 0.0, "rv": 20}

ex6_base = "/home/jneal/Phd/Analysis/Paper_examples/Example6_highrest_smaller_rv2/analysis/"
# correct6 = {"teff_1": 5800, "logg_2": 4.5, "feh_1": 0.0, "gamma": -15,
#            "teff_2": 4000, "logg_2": 5.0, "feh_2": 0.0, "rv": 7.5}


correct6 = {"teff_1": 5800, "logg_2": 4.5, "feh_1": 0.0, "gamma": 0,
            "teff_2": 4000, "logg_2": 5.0, "feh_2": 0.0, "rv": 10}

# ex6_base = "/home/jneal/Phd/Analysis/Paper_examples/Example_6_0_10_logg45/analysis/"
ex6_base = "/home/jneal/Phd/Analysis/Paper_examples/Example_6_0_10/analysis/"

runname = "Example6150RES"

ms = 10
# grid = True
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


@styler
def f(fig, *args, **kwargs):
    ### BHM ANALYSIS
    grid = kwargs.get("grid", False)

    # EX 6 bhm recovery  on iam sim. snr=150
    bhm_sim_example = SingleSimReader(base=ex6_base, name=runname, mode="bhm", chi2_val=chi2_val)

    bhm_extractor = DBExtractor(bhm_sim_example.get_table())

    bhm_df_min = bhm_extractor.minimum_value_of(chi2_val)
    print("M-Dwarf single Result min values.")
    print(bhm_df_min.head())
    cols = ['gamma', chi2_val, 'teff_1', 'logg_1', 'feh_1']

    fixed = {key: bhm_df_min[key].values[0] for key in ["logg_1", "feh_1"]}

    bhm_df = bhm_extractor.fixed_extraction(cols, fixed, limit=-1)

    kwargs = {"correct": correct6, "dof": 2, "grid": grid, "ms": ms, "sigma": [3]}
    plt.subplot(221)
    df_contour(bhm_df, "teff_1", "gamma", chi2_val, bhm_df_min, [], xlim=[5700, 5900],
               ylim=[-0.5, 0.505], **kwargs)
    plt.annotate("$C^1$", (.01, 0.95), xycoords='axes fraction')

    ################################
    # ### IAM RESULTS
    sim_example = SingleSimReader(base=ex6_base, name=runname, mode="iam", chi2_val=chi2_val)

    extractor = DBExtractor(sim_example.get_table())

    df_min = extractor.minimum_value_of(chi2_val)

    cols = ['teff_2', 'logg_2', 'feh_2', 'rv', 'gamma',
            chi2_val, 'teff_1', 'logg_1', 'feh_1']

    fixed = {key: df_min[key].values[0] for key in ["logg_1", "logg_2", "feh_1", "feh_2"]}

    df = extractor.fixed_extraction(cols, fixed, limit=-1)

    kwargs.update({"dof": 4})

    plt.subplot(222)
    df_contour(df, "teff_2", "teff_1", chi2_val, df_min, ["gamma", "rv"], xlim=[3400, 4400], ylim=[5700, 5900],
               **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')

    plt.subplot(223)
    df_contour(df, "teff_1", "gamma", chi2_val, df_min, ["teff_2", "rv"], xlim=[5700, 5900], ylim=[-0.5, 0.505],
               **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')

    plt.subplot(224)
    df_contour(df, "teff_2", "rv", chi2_val, df_min, ["gamma", "teff_1"], xlim=[3400, 4400], ylim=[5, 14.95], **kwargs)
    plt.annotate("$C^2$", (.01, 0.95), xycoords='axes fraction')
    print("Mdwarf binary min results")
    print(df_min.head())


if __name__ == "__main__":
    f(type="two", tight=True, dpi=400, save="../final/Mdwarf_contours.pdf", figsize=(None, .70), axislw=0.5,
      formatcbar=False,
      formatx=False, formaty=False, grid=False)
    f(type="two", tight=True, dpi=400, save="../final/Mdwarf_pcolors.pdf", figsize=(None, .70), axislw=0.5,
      formatcbar=False,
      formatx=False, formaty=False, grid=True)

"""Make quick-look style plot for paper showing bad pixel effect.

- Plot all 8 normalized reduced spectra
- Plot combined optimal and mixed combined
- Plot Difference optimal-mixed combined.
"""
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from octotribble.Get_filenames import get_filenames
from os.path import join, exists

# Update the thesis style to change the plot style
matplotlib.style.use("thesis")

image_path = join(
    "/",
    "home",
    "jneal",
    "Phd",
    "Writing-in-Progress",
    "thesis",
    "figures",
    "reduction",
    "bp_plots",
)
assert exists(image_path)


def bpplot(path, target, chip, **kwargs):
    """Plot the bad pixel plots for given reduced DRACS target."""
    path = join(path, target)
    intermediate_path = join(path, "Intermediate_steps")
    combined_path = join(path, "Combined_Nods")

    assert exists(path)
    assert exists(intermediate_path)
    assert exists(combined_path)

    norm_names = get_filenames(
        intermediate_path, "CRIRE*.ms.norm.fits", "*_{0}.*".format(chip), fullpath=True
    )

    norm_data_opt = [fits.getdata(name)[0, 0] for name in norm_names]

    norm_data_non_opt = [fits.getdata(name)[1, 0] for name in norm_names]

    nod_labels = range(1, 9)

    """Create bad pixel plot"""
    fig = plt.figure()

    h_ratio = kwargs.get("height_ratio", [1, 1, 0.5])
    gs = gridspec.GridSpec(3, 1, height_ratios=h_ratio)
    ax1 = fig.add_subplot(gs[0])

    #### Top Plot: Optimal
    for ii, data in enumerate(norm_data_opt):
        data_mask = (data > 4 * 1.2) | (data < 0.0)
        data[data_mask] = np.nan
        ax1.plot(data + ii * -0.1, label=nod_labels[ii], lw=0.8)
    plt.ylabel("Norm Flux")
    plt.annotate("Optimal", (500, 0.05))
    plt.xlim([0, 1024])
    start, end = ax1.get_ylim()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylim([-0.1, 1.05])

    ### Mid plot: Non optimal
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    for ii, data in enumerate(norm_data_non_opt):
        data_mask = (data > 4 * 1.2) | (data < 0.0)
        data[data_mask] = np.nan
        ax2.plot(data + ii * -0.1, label=nod_labels[ii], lw=0.8)
    plt.ylabel("Norm Flux")
    plt.annotate("Rectangular", (500, 0.05))
    plt.xlim([0, 1024])
    plt.ylim([-0.1, 1.05])
    start, end = ax2.get_ylim()
    plt.setp(ax2.get_xticklabels(), visible=False)

    # LOAD in opt combine and mix avg combine
    combined_name = get_filenames(
        combined_path, "CRIRE*norm.sum.fits", "*_{0}.*".format(chip), fullpath=True
    )
    comb_data = fits.getdata(combined_name[0])
    mix_avg_name = get_filenames(
        combined_path, "CRIRE*norm*mixavg.fits", "*_{0}.*".format(chip), fullpath=True
    )
    mix_data = fits.getdata(mix_avg_name[0])

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(comb_data[0, 0] - mix_data, label="Optimal - Mixed", lw=0.8)
    plt.hlines(0, -10, 1050, colors="k", linestyles="--", alpha=0.5, lw=0.5)
    plt.xlim([0, 1024])
    plt.xlabel("Pixel number")
    plt.ylabel("$\Delta$ Flux")

    plt.tight_layout()

    return fig


def resize_plot(path, target, chip, **kwargs):
    """Plot the bad pixel plots for given resized aperature reduced DRACS target."""
    path = join(path, target)
    intermediate_path = join(path, "Intermediate_steps")
    combined_path = join(path, "Combined_Nods")

    assert exists(path)
    assert exists(intermediate_path)
    assert exists(combined_path)

    norm_names = get_filenames(
        intermediate_path, "CRIRE*.ms.norm.fits", "*_{0}.*".format(chip), fullpath=True
    )

    norm_data_opt = [fits.getdata(name)[0, 0] for name in norm_names]

    norm_data_non_opt = [fits.getdata(name)[1, 0] for name in norm_names]

    nod_labels = range(1, 9)

    """Create bad pixel plot"""
    fig = plt.figure()

    h_ratio = kwargs.get("height_ratio", [1, 1])
    gs = gridspec.GridSpec(2, 1, height_ratios=h_ratio)
    ax1 = fig.add_subplot(gs[0])

    #### Top Plot: Optimal
    for ii, data in enumerate(norm_data_opt):
        data_mask = (data > 4 * 1.2) | (data < 0.0)
        data[data_mask] = np.nan
        ax1.plot(data + ii * -0.1, label=nod_labels[ii], lw=0.8)
    plt.ylabel("Norm Flux")
    plt.annotate("Optimal", (500, 0.05))
    plt.xlim([0, 1024])
    start, end = ax1.get_ylim()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylim([-0.1, 1.05])

    ### Mid plot: Non optimal
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    for ii, data in enumerate(norm_data_non_opt):
        data_mask = (data > 4 * 1.2) | (data < 0.0)
        data[data_mask] = np.nan
        ax2.plot(data + ii * -0.1, label=nod_labels[ii], lw=0.8)
    plt.ylabel("Norm Flux")
    plt.annotate("Rectangular", (500, 0.05))
    plt.xlim([0, 1024])
    plt.ylim([-0.1, 1.05])
    start, end = ax2.get_ylim()
    #plt.setp(ax2.get_xticklabels(), visible=False)

    plt.xlabel("Pixel")

    plt.tight_layout()

    return fig


data = join("/", "home", "jneal", "Phd", "data", "Crires", "BDs-DRACS")
if __name__ == "__main__":
    targets = [
        "HD4747-1",
        "HD30501-1",
        "HD30501-2a",
        "HD30501-2b",
        "HD30501-3",
        "HD162020-1",
        "HD162020-2",
        "HD167665-1a",
        "HD167665-1b",
        "HD167665-2",
        "HD168443-1",
        "HD168443-2",
        "HD202206-1",
        "HD202206-2",
        "HD202206-3",
        "HD211847-1",
        "HD211847-2",
    ]
    path = join(data, "2017")

    for target in targets:
        for chip in range(1, 5):
            figure = bpplot(path, target, chip, height_ratio=[1, 1, 0.4])

            figure.savefig(
                join(
                    image_path,
                    "extraction_comparision_{}_chip_{}.pdf".format(target, chip),
                ),
                dpi=500,
            )
            plt.close(figure)

    # Targets with non-re-sized/re-sized width.

    nonresized_path = path = join(data, "2018-sigmacheck", "sigma3")
    resized_path = path = join(data, "2018-sigmacheck", "sigma3resize")

    resized_targets = ["HD202206-1", "HD162020-1", "HD167665-1b"]

    for resize_target in resized_targets:
        for chip in range(1, 5):
            resizefig = resize_plot(
                resized_path, resize_target, chip, height_ratio=[1, 1]
            )
            resizefig.savefig(
                join(image_path, "resized_nods_{}_chip_{}.pdf".format(resize_target, chip)), dpi=500
            )
            plt.close(resizefig)
            non_fig = resize_plot(
                nonresized_path, resize_target, chip, height_ratio=[1, 1]
            )

            non_fig.savefig(
                join(image_path, "non_resized_nods_{}_chip_{}.pdf".format(resize_target, chip)), dpi=500
            )

            plt.close(non_fig)

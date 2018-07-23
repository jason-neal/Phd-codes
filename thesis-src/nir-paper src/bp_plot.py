"""Make quick-look style plot for paper showing bad pixel effect.

- Plot all 8 normalized reduced spectra
- Plot combined optimal and mixed combined
- Plot Difference optimal-mixed combined.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from octotribble.Get_filenames import get_filenames

from utils.styler import styler

if __name__ == '__main__':
    observation_name = "HD30501-1"
    #print(path)
    path = os.path.join("/", "home", "jneal", "Phd", "data", "Crires", "BDs-DRACS", "2017", observation_name)
    intermediate_path = os.path.join(path, "Intermediate_steps")
    combined_path = os.path.join(path, "Combined_Nods")
    image_path = os.path.join("/", "home", "jneal", "Phd", "Writing-in-Progress", "nir-paper", "images", "tmp-images")

    assert os.path.exists(path)
    assert os.path.exists(intermediate_path)
    assert os.path.exists(combined_path)
    assert os.path.exists(image_path)

    # chip = range(1, 5)
    chip = 2

    nod_names = get_filenames(intermediate_path, 'CRIRE*.ms.fits', "*_{0}.*".format(chip), fullpath=True)
    norm_names = get_filenames(intermediate_path, 'CRIRE*.ms.norm.fits', "*_{0}.*".format(chip), fullpath=True)

    optimal_median = []
    optimal_mean = []
    non_optimal_median = []
    non_optimal_mean = []

    ### Mid plot
    ### Non optimal
    # nod_data = [fits.getdata(name)[indx, 0] for name in nod_names]

    norm_data_opt = [fits.getdata(name)[0, 0] for name in norm_names]

    norm_data_non_opt = [fits.getdata(name)[1, 0] for name in norm_names]

    nod_labels = range(1, 9)

    import matplotlib.gridspec as gridspec
    @styler
    def f(fig, *args, **kwargs):
        """Create bad pixel plot"""
        h_ratio = kwargs.get("height_ratio", [1, 1, 0.5])
        gs = gridspec.GridSpec(3, 1, height_ratios=h_ratio)
        ax1 = fig.add_subplot(gs[0])
        #### Top Plot: Optimal
        for ii, data in enumerate(norm_data_opt):
            data_mask = (data > 4 * 1.2) | (data < 0.0)
            data[data_mask] = np.nan
            ax1.plot(data + ii * -0.1, label=nod_labels[ii], lw=0.8)
        plt.ylabel("Norm Flux")
        # plt.title("Optimal Extraction")
        plt.annotate("Optimal", (500, 0.1))
        plt.xlim([0, 1024])
        start, end = ax1.get_ylim()
        # plt.legend(ncol=8)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylim([-0.1, 1.05])

        #ax1.xaxis.set_major_formatter(plt.NullFormatter())
        # ax2.yaxis.set_ticks(np.arange(start, end, 0.1))

        ### Mid plot: Non optimal
        ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
        for ii, data in enumerate(norm_data_non_opt):
            data_mask = (data > 4 * 1.2) | (data < 0.0)
            data[data_mask] = np.nan
            ax2.plot(data + ii * -0.1, label=nod_labels[ii], lw=0.8)
        plt.ylabel("Norm Flux")
        plt.annotate("Rectangular", (500, 0.1))
        plt.xlim([0, 1024])
        plt.ylim([-0.1, 1.05])
        start, end = ax2.get_ylim()
        plt.setp(ax2.get_xticklabels(), visible=False)
        # ax2.xaxis.set_major_formatter(plt.NullFormatter())
        # plt.legend(ncol=8)
        # ax2.yaxis.set_ticks(np.arange(start, end, 0.1))

        # LOAD in opt combine and mix avg combine
        combined_name = get_filenames(combined_path, 'CRIRE*norm.sum.fits', "*_{0}.*".format(chip), fullpath=True)
        comb_data = fits.getdata(combined_name[0])
        mix_avg_name = get_filenames(combined_path, 'CRIRE*norm*mixavg.fits', "*_{0}.*".format(chip), fullpath=True)
        mix_data = fits.getdata(mix_avg_name[0])

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(comb_data[0, 0] - mix_data, label="Optimal - Mixed", lw=0.8)
        plt.hlines(0, -10, 1050, colors="k", linestyles="--", alpha=0.5, lw=0.5)
        plt.xlabel("Pixel number")
        plt.ylabel("$\Delta$ Flux")
        # plt.legend()
        # plt.show()
        #ax1.get_xaxis().set_ticklabels([])
        #ax2.get_xaxis().set_ticklabels([])

    # fig.savefig(image_path + "combine_diff_{0}_{1}_reduction_opt_minus_nonopt.png".format(observation_name, chip_num))

    #f(type='A&A', tight=True, dpi=300, axislw=0.5, figsize=(None, 1.), formatx=False, height_ratio=[1,1,0.3])
    f(type='A&A', tight=True, dpi=300, save=os.path.join(image_path, "Bad_pixel_replacement.pdf"), axislw=0.5, figsize=(None, 1), formatx=False, height_ratio=[1,1,0.3])
    f(type='A&A', tight=True, dpi=300, save=os.path.join(image_path, "Bad_pixel_replacement.png"), axislw=0.5, figsize=(None, 1), formatx=False, height_ratio=[1,1,0.3])

    # f(type='A&A', save='TEST_output_figure.pdf', tight=True, verbose=True)
    # f(type='A&AFW', save='2TEST_output_figureFW.pdf', tight=True, verbose=True)
    sys.exit(0)

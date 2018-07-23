# /usr/bin/env python
# Compare ESO-DRACS extracted spectra results

import glob
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from spectrum_overload.norm import continuum
import numpy as np

from os.path import join, exists

# Update the thesis style to change the plot style
matplotlib.style.use("thesis")

dracs_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/2017"

gasgano_path = "/home/jneal/Phd/data/Crires/Gasgano-BDs"

image_path = join("/home/jneal/Phd/Writing-in-Progress/thesis/",
                  "figures/reduction/pipeline_compare/"
                  )

assert exists(image_path)


def load_dracs_mix(target, chip):
    filename = glob.glob(join(dracs_path, target,
                              "Combined_Nods/CRIRE*_{0}.nod.ms.norm.mixavg.fits".format(chip)))[0]
    spec = fits.getdata(filename)
    return spec


def load_dracs(target, chip, opt=True):
    filename = glob.glob(join(dracs_path, target,
                              "Combined_Nods/CRIRE*_{0}.nod.ms.norm.sum.fits".format(chip)))[0]

    data = fits.getdata(filename)
    print("dracs data", data)

    if opt:
        spec = data[0]
    else:
        spec = data[1]
    return spec[0]


def load_eso(target, chip, opt=True):
    filename = join(gasgano_path, target, "Reduced/crires_spec_jitter_extracted_0001.fits")
    data = fits.getdata(filename, chip)
    # print("gasgano data", data)
    print(data.shape)
    print(data.columns)
    if opt:
        spec = data["Extracted_OPT"]
    else:
        spec = data["Extracted_RECT"]
    eso_wav = data["Wavelength"]
    print("eso spec, chip", chip, "=", spec)
    return eso_wav, spec


if __name__ == "__main__":
    for opt in [True]:
        for target in ["HD30501-1", "HD202206-1"]:
            for chip in range(1, 5):
                fig = plt.figure(figsize=(6, 3))

                dracs = load_dracs(target, chip, opt=opt)
                wav, eso = load_eso(target, chip, opt=opt)
                dracs_mix = load_dracs_mix(target, chip)

                # Continuum normalize eso
                eso_cont = continuum(np.arange(1024), eso, method="quadratic", nbins=20)
                dracs_cont = continuum(np.arange(1024), dracs, method="quadratic", nbins=20)
                dracs_mix_cont = continuum(np.arange(1024), dracs_mix, method="quadratic", nbins=20)


                plt.plot(wav, eso / eso_cont, label="ESO - Optimal")
                plt.plot(wav, dracs/dracs_cont, "--", label="DRACS - Optimal")
                plt.plot(wav, dracs_mix / dracs_mix_cont, "-.", label="DRACS - Combination")
                plt.ylabel("Normalized Flux")
                plt.xlabel("Wavelength (nm)")

                plt.title(r"{} \#{}".format(target, chip))
                plt.legend()
                plt.tight_layout()
                plt.savefig(join(image_path, "pipeline_compare_{}_chip_{}".format(target, chip)))
                plt.savefig(join(image_path, "pipeline_compare_{}_chip_{}.pdf".format(target, chip)))

                plt.show()
                plt.close(fig)
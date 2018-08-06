# Check wavelength differences between my wavelength calibration and eso calibration

import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from os.path import join, exists
from spectrum_overload.norm import continuum

# Update the thesis style to change the plot style
matplotlib.style.use("thesis")
dracs_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/2017"

gasgano_path = "/home/jneal/Phd/data/Crires/Gasgano-BDs"

image_path = join(
    "/home/jneal/Phd/Writing-in-Progress/thesis/",
    "figures/reduction/wavelength_compare/",
)

assert exists(image_path)


def load_dracs_mix_wav(target, chip):
    filename = glob.glob(
        join(
            dracs_path,
            target,
            "Combined_Nods/CRIRE*_{0}.nod.ms.norm.mixavg.wavecal.fits".format(chip),
        )
    )[0]
    spec = fits.getdata(filename)
    return spec["Wavelength"], spec["Flux"]


def load_eso(target, chip, opt=True):
    filename = join(
        gasgano_path, target, "Reduced/crires_spec_jitter_extracted_0001.fits"
    )
    data = fits.getdata(filename, chip)
    print(data.shape)
    print(data.columns)
    if opt:
        spec = data["Extracted_OPT"]
    else:
        spec = data["Extracted_RECT"]
    eso_wav = data["Wavelength"]
    print("eso spec, chip", chip, "=", spec)
    return eso_wav, data["Wavelength_model"], spec


if __name__ == "__main__":
    for target in ["HD30501-1", "HD202206-1"]:
        for chip in range(1, 5):
            fig = plt.figure(figsize=(6, 3))
            ax1 = plt.subplot(211)
            eso_wav, wav_model, eso = load_eso(target, chip, opt=True)
            my_wav, dracs_mix = load_dracs_mix_wav(target, chip)
            print("MODEL", wav_model)
            print(eso_wav)

            # Continuum normalize eso
            eso_cont = continuum(np.arange(1024), eso, method="quadratic", nbins=20)
            dracs_mix_cont = continuum(
                np.arange(1024), dracs_mix, method="quadratic", nbins=20
            )

            plt.plot(eso_wav, eso / eso_cont, label="ESO - Optimal")
            model_nonzero = wav_model > 2000
            plt.plot(
                wav_model[model_nonzero],
                eso[model_nonzero] / eso_cont[model_nonzero],
                label="ESO - Optimal - wav model",
            )

            plt.plot(
                my_wav, dracs_mix / dracs_mix_cont, "-.", label="DRACS - Combination"
            )
            plt.ylabel("Normalized Flux")


            plt.title(r"{} \#{}".format(target, chip))

            plt.legend()

            ax2 = plt.subplot(212, sharex=ax1)
            ax2.plot(my_wav, my_wav-eso_wav, label="my_wav-eso_wav")

            plt.xlabel("Wavelength (nm)")
            plt.tight_layout()
            plt.savefig(
                join(image_path, "wave_compare_{}_chip_{}".format(target, chip))
            )
            plt.savefig(
                join(image_path, "wave_compare_{}_chip_{}.pdf".format(target, chip))
            )

            plt.show()
            plt.close(fig)

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
    filename = glob.glob(
        join(gasgano_path, target, "Reduced/crires_spec_jitter_extracted_000*.fits")
    )
    print("glob filenames", filename)
    filename.sort()
    filename = filename[-1]  # get highest number crires
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
    wavs = []
    wav_diffs = []
    plots = False
    for target in [
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
    ]:
        for chip in range(1, 5):
            fig = plt.figure(figsize=(6, 3))
            try:


                eso_wav, wav_model, eso = load_eso(target, chip, opt=True)
                my_wav, dracs_mix = load_dracs_mix_wav(target, chip)
                print("MODEL", wav_model)
                print(eso_wav)

                # Continuum normalize eso
                eso_cont = continuum(np.arange(1024), eso, method="quadratic", nbins=20)
                dracs_mix_cont = continuum(
                    np.arange(1024), dracs_mix, method="quadratic", nbins=20
                )
                if plots:
                    ax1 = plt.subplot(211)
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
                    ax2.plot(my_wav, my_wav - eso_wav, label="my_wav-eso_wav")

                    plt.xlabel("Wavelength (nm)")
                    plt.tight_layout()
                    plt.savefig(
                        join(image_path, "wave_compare_{}_chip_{}_dist.png".format(target, chip)),
                             dpi=400
                    )

                    plt.savefig(
                        join(image_path, "wave_compare_{}_chip_{}_dist.pdf".format(target, chip))
                    )


                wavs.append(my_wav)
                print("waves length", len(wavs), wavs)
                wav_diffs.append(my_wav - eso_wav)
            except IndexError:
                print("NO gasgano reduced spectra for ")
                print("Target", target)
                print("chip", chip)

            plt.close(fig)

    print("waves", wavs)
    print("wave diffs", wav_diffs)

    wavs = np.asarray(wavs)
    wav_diffs = np.asarray(wav_diffs)

    print("waves", wavs)
    print("wave diffs", wav_diffs)

    plt.figure()
    plt.hist(wav_diffs.flatten(), bins=100, label="Dracs-ESO")
    plt.title("Wavelength differences")
    plt.legend()
    plt.xlabel(r"$\Delta \lambda$ (nm)")
    plt.show()

    print("mean difference", np.nanmean(wav_diffs.flatten()))
    print("abs mean difference", np.nanmean(np.abs(wav_diffs.flatten())))

    print("std difference", np.nanstd(wav_diffs.flatten()))

    print("max difference", np.nanmax(wav_diffs.flatten()))



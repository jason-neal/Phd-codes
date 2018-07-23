# coding: utf-8
# Using the styler

# Plot one spectrum from each observation and each detector
import glob
import os
from astropy.io import fits
import matplotlib.pyplot as plt

import numpy as np
from styler import styler
from spectrum_overload import Spectrum


# Load telluric function

def load_telluric(path, filename):
    """Load in TAPAS telluric data and header.

    If just want the data then call as load_telluric()[0]
    or data, __ = load_telluric()

    Likewise just the header as hdr = load_telluric()[1].

    """
    ext = filename[-4:]
    file_ = path + filename
    if ext == "ipac":
        tell_hdr = fits.Header()
        with open(file_) as f:
            col1 = []
            col2 = []
            for line in f:
                if line.startswith("\\"):
                    # Get the Tapas Header
                    line = line[1:]  # remove the leading \
                    line = line.strip()
                    items = line.split("=")

                    tell_hdr[items[0]] = items[1]  # Add to header

                elif line.startswith("|"):
                    # Obtain wavelength scale from piped lines
                    if "in air" in line:
                        tell_hdr["WAVSCALE"] = "air"
                    elif "nm|" in line:
                        tell_hdr["WAVSCALE"] = "vacuum"
                    # Need extra condition to deal with wave number
                else:
                    line = line.strip()
                    val1, val2 = line.split()
                    col1.append(float(val1))
                    col2.append(float(val2))

    elif ext == "fits":
        i_tell = fits.getdata(file_, 1)
        tell_hdr = fits.getheader(file_, 1)
        # TODO ... Need to get wavelength scale (air/wavelength) from fits file somehow...
        col1 = i_tell["wavelength"]
        col2 = i_tell["transmittance"]

    else:
        raise ValueError("Tapas file '{0}' Does not have an extension of '.ipac' or '.fits'".format(filename))

    # Sort in ascending wavelength
    if col1[-1] - col1[0] < 0:  # wl is backwards
        col1 = col1[::-1]
        col2 = col2[::-1]
    tell_data = np.array([col1, col2], dtype="float64")

    return tell_data, tell_hdr


if __name__ == "__main__":
    # dir
    folder = "/home/jneal/Phd/data/Crires/BDs-DRACS/2017/"
    observations = ["HD167665-1a", "HD162020-2", "HD168443-1", "HD202206-1", "HD4747-1", "HD211847-2", "HD30501-1"]
    chips = [1, 2, 3, 4]

    cm2inches = 1 / 2.54
    plt.figure(figsize=(15 * cm2inches, 20 * cm2inches))


    @styler
    def f(fig, *args, **kwargs):
        figtype = kwargs.get('type', 'A&A')
        if figtype.lower() in ('a&afw', 'twocolumns', 'two', 'full'):
            pltnum = 220
        else:
            pltnum = 410

        text_offset = 1.05
        offset_slope = 0.25
        offset_value = 6 * offset_slope
        #### CHIP 1
        chip = 1
        pltnum += 1
        ax1 = fig.add_subplot(pltnum)
        spectrum = "CRIRE.2012*_{0}.nod.ms.norm.mixavg.wavecal.fits".format(chip)
        for ii, observation in enumerate(observations):
            file = glob.glob(os.path.join(folder, observation, "Combined_Nods", spectrum))
            data = fits.getdata(file[0])

            telluric = glob.glob(os.path.join(folder, observation,
                                              "Telluric_files",
                                              "tapas*_ReqId_10_R-50000_sratio-10_barydone-NO*.ipac"))

            assert len(telluric) == 1
            telluric_data, tell_header = load_telluric("", telluric[0])

            tell_spec = Spectrum(xaxis=telluric_data[0], flux=telluric_data[1])
            tell_spec.wav_select(data["Wavelength"][0], data["Wavelength"][-1])
            offset = offset_value - offset_slope * ii
            plt.plot(data["Wavelength"], data["Flux"] + offset, label=observation.split("-")[0], lw=1)
            plt.plot(tell_spec.xaxis, tell_spec.flux + offset, "k--", lw=.8)

            xoffset = 0
            plt.annotate(observation.replace("a", ""), (data["Wavelength"][0] + xoffset, text_offset + offset))
        plt.ylabel("Norm Flux")

        plt.annotate("Detector {}".format(chip), (.8, 0.05), xycoords='axes fraction')

        #### CHIP 2
        chip = 2
        pltnum += 1
        ax2 = fig.add_subplot(pltnum, sharey=ax1)
        spectrum = "CRIRE.2012*_{0}.nod.ms.norm.mixavg.wavecal.fits".format(chip)
        for ii, observation in enumerate(observations):
            file = glob.glob(os.path.join(folder, observation, "Combined_Nods", spectrum))
            data = fits.getdata(file[0])

            telluric = glob.glob(os.path.join(folder, observation,
                                              "Telluric_files",
                                              "tapas*_ReqId_10_R-50000_sratio-10_barydone-NO*.ipac"))

            assert len(telluric) == 1
            telluric_data, tell_header = load_telluric("", telluric[0])

            tell_spec = Spectrum(xaxis=telluric_data[0], flux=telluric_data[1])
            tell_spec.wav_select(data["Wavelength"][0], data["Wavelength"][-1])
            offset = offset_value - offset_slope * ii
            plt.plot(data["Wavelength"], data["Flux"] + offset, label=observation.split("-")[0], lw=1)
            plt.plot(tell_spec.xaxis, tell_spec.flux + offset, "k--", lw=.8)

            xoffset = 1.5

            plt.annotate(observation.replace("a", ""), (data["Wavelength"][0] + xoffset, text_offset + offset))
        if pltnum == 412:
            plt.ylabel("Norm Flux")
        plt.annotate("Detector {}".format(chip), (.8, 0.05), xycoords='axes fraction')

        #### CHIP 3
        chip = 3
        pltnum += 1
        ax3 = fig.add_subplot(pltnum, sharey=ax1)
        spectrum = "CRIRE.2012*_{0}.nod.ms.norm.mixavg.wavecal.fits".format(chip)
        for ii, observation in enumerate(observations):
            file = glob.glob(os.path.join(folder, observation, "Combined_Nods", spectrum))
            data = fits.getdata(file[0])

            telluric = glob.glob(os.path.join(folder, observation,
                                              "Telluric_files",
                                              "tapas*_ReqId_10_R-50000_sratio-10_barydone-NO*.ipac"))

            assert len(telluric) == 1
            telluric_data, tell_header = load_telluric("", telluric[0])

            tell_spec = Spectrum(xaxis=telluric_data[0], flux=telluric_data[1])
            tell_spec.wav_select(data["Wavelength"][0], data["Wavelength"][-1])
            offset = offset_value - offset_slope * ii
            plt.plot(data["Wavelength"], data["Flux"] + offset, label=observation.split("-")[0], lw=1)
            plt.plot(tell_spec.xaxis, tell_spec.flux + offset, "k--", lw=.8)

            xoffset = 0
            plt.annotate(observation.replace("a", ""), (data["Wavelength"][0] + xoffset, text_offset + offset))
        if pltnum == 223:
            plt.xlabel("Wavelength (nm)")
        plt.ylabel("Norm Flux")
        plt.annotate("Detector {}".format(chip), (.8, 0.05), xycoords='axes fraction')

        #### CHIP 4
        chip = 4
        pltnum += 1
        print(pltnum)
        ax4 = fig.add_subplot(pltnum, sharey=ax1)
        spectrum = "CRIRE.2012*_{0}.nod.ms.norm.mixavg.wavecal.fits".format(chip)
        for ii, observation in enumerate(observations):
            file = glob.glob(os.path.join(folder, observation, "Combined_Nods", spectrum))
            data = fits.getdata(file[0])

            telluric = glob.glob(os.path.join(folder, observation,
                                              "Telluric_files",
                                              "tapas*_ReqId_10_R-50000_sratio-10_barydone-NO*.ipac"))

            assert len(telluric) == 1
            telluric_data, tell_header = load_telluric("", telluric[0])

            tell_spec = Spectrum(xaxis=telluric_data[0], flux=telluric_data[1])
            tell_spec.wav_select(data["Wavelength"][0], data["Wavelength"][-1])
            offset = offset_value - offset_slope * ii
            plt.plot(data["Wavelength"], data["Flux"] + offset, label=observation.split("-")[0], lw=1)
            plt.plot(tell_spec.xaxis, tell_spec.flux + offset, "k--", lw=.8)

            xoffset = 0
            plt.annotate(observation.replace("a", ""), (data["Wavelength"][0] + xoffset, text_offset + offset))
        if pltnum == 414:
            plt.ylabel("Norm Flux")
        plt.annotate("Detector {}".format(chip), (.8, 0.05), xycoords='axes fraction')

        plt.xlabel("Wavelength (nm)")
        plt.plot(2160, 2.6)
        # plt.legend(ncol=4, loc=(-20, 0))
        plt.xlabel("Wavelength (nm)")


    print(os.getcwd())
    #f(type="two", tight=True, dpi=300, figsize=(None, .70), axislw=0.5)
    f(type="two", tight=True, dpi=300, save="../final/Spectra_examples.pdf", figsize=(None, .70), axislw=0.5)
    f(type="two", tight=True, dpi=300, save="../final/Spectra_examples.png", figsize=(None, .70), axislw=0.5)

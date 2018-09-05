#!/usr/bin/env python3
# -*- coding: utf8 -*-

""" Obtain_Telluric.py

Module that has a function that can obtain the telluric spectra relevant to the observations we have
## ie. match date, time, RA/DEC
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from octotribble.Get_filenames import get_filenames


def get_telluric_name(path, date, time, ext="*"):
    """Find telluric spectra that matches the input conditions of the observation.

    Note:
        Tapas produces error of 1 hour in timing of observation
         so need to add +1 to the hour.
        This seems to be for manual pull down from the website but not the xml code.
    """

    # ext can be specified .ipac or .fits or left as * for either
    # tapas_time = str(int(time[0:2]) + 1)  # including offset
    tapas_time = str(int(time[0:2]))  # no time offset
    print("Hour of observation, ", tapas_time)
    str1 = "tapas_*" + date + "*"
    if int(tapas_time) > 9:
        str2 = "*" + tapas_time + ":*:*"
    else:
        str2 = "*T0" + tapas_time + ":*:*" + ext
    print("Match Filenames to", str1, "and", str2)
    match = get_filenames(path, str1, str2)
    if not match and int(time[3:5]) > 40:  # Check without an hour specification

        tapas_time = str(int(tapas_time) + 1)
        if int(tapas_time) > 9:
            str2 = "*" + tapas_time + ":*:*"
        else:
            str2 = "*T0" + tapas_time + ":*:*" + ext
        print("Match Filenames to", str1, "and", str2)
        match = get_filenames(path, str1, str2)
    return match


def list_telluric(path):
    match = get_filenames(path, "tapas_*")
    print("List all " "tapas_*" " files in directory")
    return match


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
        raise ValueError(
            "Tapas file '{0}' Does not have an extension of '.ipac' or '.fits'".format(
                filename
            )
        )

    # Sort in ascending wavelength
    if col1[-1] - col1[0] < 0:  # wl is backwards
        col1 = col1[::-1]
        col2 = col2[::-1]
    tell_data = np.array([col1, col2], dtype="float64")

    return tell_data, tell_hdr


def plot_telluric(data, name, labels=True, show=False):
    plt.plot(data[0], data[1], label=name)
    plt.legend()
    if labels:
        plt.title("Telluric Spectra")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmittance")
    if show:
        plt.show()
    pass


if __name__ == "__main__":
    tapas_path = "/home/jneal/Phd/data/Tapas/"

    print("Beginning Telluric request")

    test_target = "HD30501-1"
    test_date = "2012-04-07"  # date yy-mm-dd
    test_time = "00:20:20"  # hh:mm:ss
    test_ra = "04:45:38"
    test_dec = "-50:04:38"

    test_result = get_telluric_name(tapas_path, test_date, test_time)
    print("TEST Result", test_result)

    list_files = list_telluric(tapas_path)
    print(list_files)

    for filename in list_files:
        data = load_telluric(tapas_path, filename)
        if filename[17:25] == "08:10:00":
            plt.plot(data[0], data[1], label=filename)
    plt.legend()
    plt.show()

    for filename in list_files:
        data = load_telluric(tapas_path, filename)
        plot_telluric(data, filename)
    plt.show()

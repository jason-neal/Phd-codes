#!/usr/bin/env python
# -*- coding: utf8 -*-

## Module that has a function that can obtain the telluric spectra relevant to the observations we have
## ie. match date, time, RA/DEC
import matplotlib.pyplot as plt
from Get_filenames import get_filenames
import IOmodule
from astropy.io import fits
import numpy as np

def get_telluric_name(date, time):
    """Find telluric spectra that matches the input conditions of the obersvation """
    """ Tapas produces error of 1 hour in timing of observation so need to add +1 to the hour"""
    tapas_time = str(int(time[0:2]) + 1) + time[2:]
    #print("date :",date)
    str1 = "tapas_" + date + "*"
    str2 = "*" + tapas_time + "*"
    match = get_filenames(tapas_path, str1 , str2)
    return match 


def list_telluric(path):
    match = get_filenames(path, "tapas_*")
    print("List all ""tapas_*"" files in directory")
    return match

def load_telluric(tapas_path, filename):
    ext = filename[-4:] 
    file = tapas_path + filename
    if ext == "ipac":
        with open(file) as f:
            col1 = []
            col2 = []
            for line in f:
                firstchar = line[0]
            #print("first char =", firstchar)
                if line[0] == "\\" or line[0] == "|":
                    pass #print("Match a header line")
                else:
                    line.strip()
                    val1, val2 = line.split()
                    col1.append(val1)
                    col2.append(val2)
        tell = np.array([col1,col2], dtype="float64")
    elif ext == "fits":
        i_tell = (fits.getdata(file,0))
        col1 = i_tell["wavelength"]
        col2 = i_tell["transmittance"]
        #print("i_tell", i_tell)
        #print("type(i_tell)", type(i_tell))
        tell = np.array([col1,col2], dtype="float64")
    else:
        print(" Could not load file", filename," with extention", ext)
        return None
    #print(tell)
    return tell    
    
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

if __name__== "__main__" :
    tapas_path = "/home/jneal/Phd/data/Tapas/"
    #tapas_2012-08-02T10:01:44-2452
    print("Begining Telluric request")
    test_target = "HD30501-1"
    test_date = "2012-04-07" # date yy-mm-dd
    test_time = "00:20:20"  # hh:mm:ss
    test_ra =    "04:45:38"
    test_dec = "-50:04:38"

    test_result = get_telluric_name(test_date, test_time)
    print("TEST Result", test_result)
    
    list_files = list_telluric(tapas_path)
    print(list_files)

    for filename in list_files:
        data = load_telluric(tapas_path, filename)
        #if filename[-9:-5] == "2672":
        if filename[17:25] == "08:10:00":
            plt.plot(data[0], data[1], label=filename)
    plt.legend()    
    plt.show()

    for filename in list_files:
        data = load_telluric(tapas_path, filename)
        plot_telluric(data, filename)
    plt.show()

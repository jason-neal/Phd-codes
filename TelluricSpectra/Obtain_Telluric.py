#!/usr/bin/env python
# -*- coding: utf8 -*-

## Module that has a function that can obtain the telluric spectra relevant to the observations we have
 ## ie. match date, time, RA/DEC
import matplotlib.pyplot as plt
from Get_filenames import get_filenames
import IOmodule
from astropy.io import fits
import numpy as np

def Get_TelluricName(date, time):
    """Find telluric spectra that matches the input conditions of the obersvation """
    """ Tapas produces error of 1 hour in timing of observation so need to add +1 to the hour"""
    tapas_time = str(int(time[0:2]) + 1) + time[2:]
    #print("date :",date)
    str1 = "tapas_" + date + "*"
    str2 = "*" + tapas_time + "*"
    match = get_filenames(Tapas_path, str1 , str2)
    return match 


def List_Telluric(path):
    match = get_filenames(path, "tapas_*")
    print("List all ""tapas_*"" files in directory")
    return match

def Load_Telluric(Tapas_path,filename):
    ext = filename[-4:] 
    File = Tapas_path + filename
    if ext == "ipac":
        with open(File) as f:
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
        Tell = np.array([col1,col2], dtype="float64")
    elif ext == "fits":
        I_Tell = (fits.getdata(File,0))
        col1 = I_Tell["wavelength"]
        col2 = I_Tell["transmittance"]
        #print("I_Tell", I_Tell)
        #print("type(I_Tell)", type(I_Tell))
        Tell = np.array([col1,col2], dtype="float64")
    else:
        print(" Could not load file", filename," with extention", ext)
        return None
    #print(Tell)
    return Tell    
    
def Plot_Telluric(Data, Name, labels=True, show=False):
    plt.plot(Data[0],Data[1], label=Name)
    plt.legend()
    if labels:
        plt.title("Telluric Spectra")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Transmittance")
    if show:
        plt.show()
    pass

if __name__== "__main__" :
    Tapas_path = "/home/jneal/Phd/data/Tapas/"
    #tapas_2012-08-02T10:01:44-2452
    print("Begining Telluric request")
    Test_target = "HD30501-1"
    Test_date = "2012-04-07" # date yy-mm-dd
    Test_time = "00:20:20"  # hh:mm:ss
    Test_RA =    "04:45:38"
    Test_DEC = "-50:04:38"

    Test_Result = Get_TelluricName(Test_date, Test_time)
    print("TEST Result", Test_Result)
    
    list_files = List_Telluric(Tapas_path)
    print(list_files)

    for filename in list_files:
        Data=Load_Telluric(Tapas_path, filename)
        #if filename[-9:-5] == "2672":
        if filename[17:25] == "08:10:00":
            plt.plot(Data[0],Data[1], label=filename)
    plt.legend()    
    plt.show()
    for filename in list_files:
        Data=Load_Telluric(Tapas_path, filename)
        Plot_Telluric(Data, filename)
    plt.show()
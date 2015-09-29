#!/usr/bin/env python
# -*- coding: utf8 -*-

"""Combine Normalized DRACS Specta

Script that take one imput parameter which is the path to the normobs directory

Needs functions/modules to get the filenames, export to a fits tablefile, change fits hdr 
"""

import time
import argparse
import os
import fnmatch
import matplotlib.pyplot as plt
from astropy.io import fits
import IOmodule
import numpy as np
import scipy as sp 


def get_filenames(path, regexp, regexp2=False):
    """ regexp must be a regular expression as a string
            eg '*.ms.*', '*_2.*', '*.ms.norm.fits*'
        resexp2 is if want to match two expressions such as 
            '*_1*' and '*.ms.fits*'
    """
    os.chdir(path)
    filelist = []
    for file in os.listdir('.'):
        if regexp2:  # Match two regular expresions
            if fnmatch.fnmatch(file, regexp) and fnmatch.fnmatch(file, regexp2):
                #print file
                filelist.append(file)
        else:
            if fnmatch.fnmatch(file, regexp):
                #print file
                filelist.append(file)
    filelist.sort()
    return filelist

def SumNods(Spectra, Headers, Pos="All", Norm="None"):
    """ Add together the nod postitions of Crires spectra"""
    #if Headers[i]["HIERARCH ESO SEQ NODPOS"] == Pos:
    NodSum = np.zeros_like(Spectra[0])
    if Pos.upper() == "ALL":
        # Sum all
        NodNum = 8
        for i in range(8):
            NodSum += np.array(Spectra[i])
    elif Pos.upper() == "A":
        # Sum A nods
        NodNum = 4
        for i in [0,3,4,7]:
            NodSum += np.array(Spectra[i])
    elif Pos.upper() == "B":   
        # Sum B nods
        NodNum = 4
        for i in [1,2,5,6]:
            NodSum += np.array(Spectra[i])
    if Norm.upper() == "MEDIAN":
      NodSum /= np.median(NodSum)
    elif Norm.upper() == "DIVIDE":
      NodSum /= NodNum  
    return NodSum 



def ExportToFits(Outputfile, Combined, NodA, NodB, hdr, hdrkeys, hdrvals):

    col1 = fits.Column(name="Combined", format="E", array=Norm_All) # colums of data
    col2 = fits.Column(name="Nod A", format="E", array=NodA)
    col3 = fits.Column(name="Nod B", format="E", array=NodB)
    cols = fits.ColDefs([col1, col2, col3])
    tbhdu = fits.BinTableHDU.from_columns(cols) # binary tbale hdu
    prihdr = append_hdr(hdr, hdrkeys, hdrvals)
    prihdu = fits.PrimaryHDU(header=prihdr)
    prihdu.verify()
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(Outputfile)

    return None



def append_hdr(hdr, keys, values ,item=0):
    ''' Apend/change parameters to fits hdr, 
    can take list or tuple as input of keywords 
    and values to change in the header 
    Defaults at changing the header in the 0th item 
    unless the number the index is givien,
    If a key is not found it adds it to the header'''
    # open fits file
    #hdulist = fits.open(output)
    #hdr = hdulist[item].header
    #print repr(hdr[0:10])
    #assert type(keys) == type(values), 'keys and values do not match'
    if type(keys) == str:           # To handle single value
        hdr[keys] = values
    else:
        assert len(keys) == len(values), 'Not the same number of keys as values' 
        for i in range(len(keys)):
            hdr[keys[i]] = values[i]
            print repr(hdr[0:10])
    return hdr





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine CRIRES Nod positions \
                                     Normized values')
    parser.add_argument('inputpath', help='Path to normobs directory')
    #parser.add_argument('-s', '--sun', help='Plot with spectra of the Sun (1)',
    #                     default=False)
    ##parser.add_argument('-t', '--telluric', help='Plot telluric with spectrum\
    #                    (1)', default=False)
    #parser.add_argument('-s', '--stellar', help='Stellar spectra\
    #                    (1)', default=False)
    #parser.add_argument('-r', '--rv', help='RV correction to the stellar spectra in\
    #                    km/s', default=False, type=float)
    #parser.add_argument('-l', '--lines',
    #                    help='Lines to plot on top (multiple lines is an\
    #                    option). If multiple lines needs to be plotted, then\
    #                    separate with , and without any spaces.', default=False, type=list)
    args = parser.parse_args()

    print args
    path = args.inputpath
# find norm values in this directory:

    chips = range(4)
#for chip in chips:
    while True:
        chip = 1
        org_vals = get_filenames(path, "CRIRE*.ms.fits", "*_" + str(chip + 1) + "*")
        norm_vals = get_filenames(path, "CRIRE*.ms.norm.fits", "*_" + str(chip + 1) + "*")

        print(org_vals)
        I_dracs = []
        I_norm = []
        I_norm_hdrs = []
        I_dracs_hdrs = []
        for name in org_vals:    
                #print("name", name)
                ThisFile = path + name
                I_dracs.append(fits.getdata(ThisFile,0))
                I_dracs_hdrs.append(fits.getheader(ThisFile,0))
        
        dracs_All = SumNods(I_dracs, I_dracs_hdrs, Pos="All", Norm="Divide")
        dracs_A = SumNods(I_dracs, I_dracs_hdrs, Pos="A", Norm="Divide")
        dracs_B = SumNods(I_dracs, I_dracs_hdrs, Pos="B", Norm="Divide")
                #print(type(fits.getdata(ThisFile,0)))
        #load Dracs files
        ## Speed testing  open verse getdata/getheader
        # for name in norm_vals:    
        #         #print("name", name)
        #         ThisFile = path + name
        #         I_norm.append(fits.getdata(ThisFile))
        #         I_norm_hdrs.append(fits.getheader(ThisFile))
        #         #print(type(I_norm))

        print(norm_vals)
        for name in norm_vals:    
                #print("name", name)
                ThisFile = path + name
                #print ("This file = ")
                ThisNorm = fits.open(ThisFile)
                Last_normhdr = ThisNorm[0].header
                I_norm_hdrs.append(Last_normhdr)
                I_norm.append(ThisNorm[0].data)
                ThisNorm.close()

        #print("Inorm",I_norm)

        #Last_normhdr.verify()

        Norm_All = SumNods(I_norm, I_norm_hdrs, Pos="All", Norm="Divide")
        Norm_A = SumNods(I_norm, I_norm_hdrs, Pos="A", Norm="Divide")
        Norm_B = SumNods(I_norm, I_norm_hdrs, Pos="B", Norm="Divide")

        plt.plot(dracs_All, label="dracs All")
        plt.plot(dracs_A, label="dracs A")
        plt.plot(dracs_B, label="dracs B")
        plt.legend()
        plt.show()

        plt.plot(Norm_All, label="All")
        plt.plot(Norm_A, label="A")
        plt.plot(Norm_B, label="B")
        plt.legend()
        plt.show()

        # write ouput to fits file
        testhdr = fits.Header()
        testhdr['TESTVal'] = 'Edwin Hubble'
        T_Now = str(time.gmtime()[0:6])
        testhdr['Date'] = (T_Now, ' Time fits was last changed')        
        #fits.writeto(output, hdulist[item].data, hdr, clobber=True)
        outputfile = path + "test_fits_ouput.fits"
        #fits.writeto(outputfile, Norm_All, I_norm_hdrs[1])
        ExportToFits(outputfile,Norm_All,Norm_A,Norm_B,Last_normhdr,["Test heder key","Test hdr key 2"],["Value 1", "Test Value 2"])
        print("Wrote to fits Succesfully")

        break














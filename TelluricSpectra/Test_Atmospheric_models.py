#!/usr/lib/python3
import numpy as np
import GaussianFitting as gf
import Obtain_Telluric as obt

import matplotlib.pyplot as plt
from astropy.io import fits

path = "/home/jneal/Phd/data/ESO-Skydata/"

filenames = ["HD30501-1-R50000-gaussianconvolution-2FWHM.dat", \
            "HD30501-1-R50000-gaussianconvolution-5FWHM.dat", \
            "HD30501-1-R50000-gaussianconvolution-8FWHM.dat", \
            "HD30501-1-R50000-noconvolution-.dat", \
            "HD30501-1-R100000-gaussianconvolution-5FWHM.dat", \
            "HD30501-1-R100000-noconvolution-.dat"]

Fits_skytables = ["HD30501-1-100000-no-convolution.fits", \
                    "HD30501-1-50000-no-convolution.fits"]
    

# See what is in the Skytables
plt.figure()
for name in Fits_skytables:
	res = name.split("-")[2]
	data = fits.getdata(path+name)
	wl = data["lam"]
	Tr = data["trans"]

	plt.plot(wl, Tr, label=res)
    
plt.legend()
#plt.show()



no_conv_filenames = ["HD30501-1-R50000-noconvolution-.dat", \
            		"HD30501-1-R100000-noconvolution-.dat"]
plt.figure()
for name in no_conv_filenames:
    wl, Tr = np.loadtxt(path+name, unpack=True)
    res = name.split("-")[2]
    plt.plot(wl, Tr, label=res)
    
plt.legend()
plt.title("Resolution Change Effect with No Convolution")
#plt.show()

R5_filenames = ["HD30501-1-R50000-noconvolution-.dat", \
			"HD30501-1-R50000-gaussianconvolution-2FWHM.dat", \
            "HD30501-1-R50000-gaussianconvolution-5FWHM.dat", \
            "HD30501-1-R50000-gaussianconvolution-8FWHM.dat"]
plt.figure()
for name in R5_filenames:
    wl, Tr = np.loadtxt(path+name, unpack=True)
    split = name.split("-")
    res = split[2]
    conv = split[3]

    if conv[0:8] == "gaussian":
    	print(conv)
    	fwhm = split[-1].split(".")[0]
    else:
    	fwhm = ""
    plt.plot(wl, Tr, label=res+" "+conv+" "+fwhm)

plt.legend()
plt.title("50000 Resolution Effect of Line Profile")
#plt.show()


R10_filenames = ["HD30501-1-R100000-gaussianconvolution-5FWHM.dat", \
            "HD30501-1-R100000-noconvolution-.dat"]
plt.figure()
for name in R10_filenames:
    wl, Tr = np.loadtxt(path+name, unpack=True)
    split = name.split("-")
    res = split[2]
    conv = split[3]
    if conv[0:8] == "gaussian":
    	print(conv)
    	fwhm = split[-1].split(".")[0]
    else:
    	fwhm = ""

    plt.plot(wl, Tr, label=res+" "+conv+" "+fwhm)
   
plt.legend()
plt.title("R=100000 Convolution Effect")
plt.show()
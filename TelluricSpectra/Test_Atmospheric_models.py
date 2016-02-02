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
#plt.show()



#####  Tapas at R 50000

path = "/home/jneal/Phd/data/tapas-testing/"

tapas1_filenames = ["tapas_2012-04-07T01:20:20-HD30501-1-R50000-sample-10.ipac", \
            "tapas_2012-04-07T01:20:20-HD30501-1-R50000-sample-5.ipac", \
            "tapas_2012-04-07T01:20:20-HD30501-1-R50000-sample-15.ipac"]
plt.figure()
for name in tapas1_filenames:
    data, hdr = obt.load_telluric(path, name)
      
    res = float(hdr["resPower"])
    sampRati = hdr["sampRati"]
    airmass = hdr["airmass"]
    plt.plot(data[0], data[1], label="R =" + str(int(res)) + ", sampling =" + sampRati + " airmass=" + str(airmass))
   
    #plt.plot(wl, Tr, label=res+" "+conv+" "+fwhm)
plt.legend()
plt.title("Tapas R=50000 sampling Effect")
#plt.show()




tapas2_filenames = ["tapas_test2_1.ipac", \
            "tapas_test2_2.ipac", \
            "tapas_test2_3.ipac", \
            "tapas_test2_4.ipac"]
plt.figure()
for name in tapas2_filenames:
    data, hdr2 = obt.load_telluric(path, name)
    for key, val in hdr2.iteritems():
        print("Key", key, "Value", val)
    
    try:
        res = hdr2["RESPOWER"]
        res = float(res)
        sampRati = hdr2["sampRati"]
        airmass = hdr2["airmass"]
    except:
        print("No RESPOWER for", name)
        res = 0
        sampRati = 0
        airmass = hdr2["airmass"]
   
    plt.plot(data[0], data[1], label="R =" + str(int(res)) + ", sampling =" + str(sampRati) + " airmass=" + str(airmass))
    #plt.plot(wl, Tr, label=res+" "+conv+" "+fwhm)
plt.legend()
plt.title("Tapas Instrument Effect")
#plt.show()




# Effect of 1 hour time increments on tapas spectra 
#(to see the effect of the 1hr timing difference given by tapas)
plt.figure()
tapas3_filenames = ["tapas_test3_1_timing_0.ipac", \
            "tapas_test3_1_timing_0.ipac", \
            "tapas_test3_1_timing_0.ipac", \
            "tapas_test3_1_timing_0.ipac", \
            "tapas_test3_1_timing_0.ipac", \
            "tapas_test3_1_timing_0.ipac"]

for name in tapas3_filenames:
    data, hdr = obt.load_telluric(path, name)
    #wl, Tr = np.loadtxt(path+name, unpack=True)
   
    res = float(hdr["resPower"])
    sampRati = hdr["sampRati"]
    airmass = hdr["airmass"]
   
   
    plt.plot(data[0], data[1], label="R =" + str(int(res)) + ", sampling =" + str(sampRati) + " airmass=" + str(airmass))
    #plt.plot(wl, Tr, label=res+" "+conv+" "+fwhm)
plt.legend()
plt.title("Tapas Timing Effects")
plt.show()





# Effect BEERV COrrection increments on tapas spectra 
#(to see the effect of the 1hr timing difference given by tapas)


tapas4_filenames = ["tapas_test3_timing_0_no_berv_corr.ipac", \
            "tapas_test3_1_timing_0.ipac"]

for name in tapas4_filenames:
    data, hdr = obt.load_telluric(path, name)
    #wl, Tr = np.loadtxt(path+name, unpack=True)
   
    res = float(hdr["resPower"])
    sampRati = hdr["sampRati"]
    airmass = hdr["airmass"]
   
   
    plt.plot(data[0], data[1], label="R =" + str(int(res)) + ", sampling =" + str(sampRati))
    #plt.plot(wl, Tr, label=res+" "+conv+" "+fwhm)
plt.legend()
plt.title("Tapas BERV Effects")
plt.show()
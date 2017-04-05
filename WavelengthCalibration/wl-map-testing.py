#!/usr/bin/env python

# Testing difference between the norm.sum and sum.norm  wavelenght calcualations

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def wavelength_map(x, params):
	wl = [params[0]*(xi**2) + params[1]*xi + params[2] for xi in x]
	return wl


path = "/home/jneal/Phd/data/Crires/BDs-DRACS/HD30501-1/Fullreductionr-test-1dec2015/"

file1 = "CRIRE.2012-04-07T00:08:29.976_1.nod.ms.sum.norm.wavecal.fits"
file2 = "CRIRE.2012-04-07T00:08:29.976_1.nod.ms.norm.sum.wavecal.fits"


# Load fits
hdr1 = fits.getheader(path+file1)
print(hdr1)
hdr2 = fits.getheader(path+file2)
print(hdr2)

data1 = fits.getdata(path+file1)
wl1 = data1["Wavelength"]
I1 = data1["Extracted_DRACS"]
P1 = data1["Pixel"]
data2 = fits.getdata(path+file2)
wl2 = data2["Wavelength"]
I2 = data2["Extracted_DRACS"]
P2 = data2["Pixel"]
print(data2[0])
params1 = [hdr1["HIERARCH PIXELMAP PARAM1"], hdr1["HIERARCH PIXELMAP PARAM2"], hdr1["HIERARCH PIXELMAP PARAM3"]]
params2 = [hdr2["HIERARCH PIXELMAP PARAM1"], hdr2["HIERARCH PIXELMAP PARAM2"], hdr2["HIERARCH PIXELMAP PARAM3"]]

plt.figure()
plt.plot(wl1, I1, "r", label="sum.norm")
plt.plot(wl2, I2, "k", label="norm.sum")
plt.title("Switch of normalization")
plt.legend()

plt.figure()
plt.plot(P1, I1-I2, "r", label="I1-I2")
plt.plot(P1, I1/I2 -1, "k", label="I1/I2 -1")
plt.title("Comparisions")
plt.legend()

plt.figure()
plt.plot(P1, wl1, "r.", label="sum.norm")
plt.plot(P2, wl2, "g.", label="sum.norm")
# plt.plot(wl2,I2, "k", label="norm.sum")
plt.title("pxl - wl")
plt.legend()

plt.figure()
plt.plot(P1, I1, "r.", label="pxls sum.norm")
plt.plot(P2, I2, "b.", label="pxls sum.norm")
# plt.plot(wl2,I2, "k", label="norm.sum")
plt.title("Switch of normalization pxls ")
plt.legend()


pxl = np.array(range(1024))
wlmap1 = np.array(wavelength_map(pxl, params1))
wlmap2 = np.array(wavelength_map(pxl, params2))
wlmap_diff = wlmap1 - wlmap2

plt.figure()
plt.plot(P1, wl1-wl2, "r", label="wl diff")
plt.plot(pxl, wlmap_diff, "k", label="wl equation diff")
plt.legend()
plt.title("Wavelength calibrations")
# plt.show()

plt.figure()
plt.plot(pxl, wlmap_diff, "k", label=" wl map equation diff")
plt.plot(pxl[:-1], wlmap1[1:]-wlmap1[:-1], "g", label="Inter pixel differnces")
plt.legend()
plt.title("wl differences")


plt.figure()
plt.plot(pxl[1:], wlmap_diff[1:]/(wlmap1[:-1]-wlmap1[1:]), "k", label="percent  wl map equation diff")
# plt.plot(pxl,  "g", label="Inter pixel differnces")
plt.legend()
plt.title("Percentage difference")
plt.show()

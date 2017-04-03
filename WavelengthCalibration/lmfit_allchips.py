#!/usr/bin/env python3
#lmfit_allchips.py

#Non-Linear Wavelength Mapping:

import sys
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from lmfit import minimize, Parameters
import lmfit

def residual(params, pixels, wl_data):
    # Polynomial of form q*x**2 + m*x+b
    q = params["q"].value
    m = params["m"].value
    b = params["b"].value
    Gap1 = params["Gap1"].value
    Gap2 = params["Gap2"].value
    Gap3 = params["Gap3"].value

    # Change spacing to add pixel gaps to chip pixel numbering
    new_pixels = np.array(pixels)
    chipmap4 = new_pixels > 3*1024
    chipmap3 = new_pixels > 2*1024
    chipmap2 = new_pixels > 1*1024
    new_pixels += Gap3*chipmap4
    new_pixels += Gap2*chipmap3
    new_pixels += Gap1*chipmap2

    model = q*new_pixels**2 + m*new_pixels + b

    return (wl_data - model)

def residual_individual(params, pixels, wl_data):
    # Polynomial of form q*x**2 + m*x+b
    q = params["q"].value
    m = params["m"].value
    b = params["b"].value

    # Change spacing to add pixel gaps to chip pixel numbering
    new_pixels = np.array(pixels)
    model = q*new_pixels**2 + m*new_pixels + b

    return (wl_data - model)

def residual_individaul_depthweighted(params, pixels, wl_data, depths):
    # Polynomial of form q*x**2 + m*x+b
    q = params["q"].value
    m = params["m"].value
    b = params["b"].value

    # Change spacing to add pixel gaps to chip pixel numbering
    new_pixels = np.array(pixels)
    model = q*new_pixels**2 + m*new_pixels + b

    return (wl_data - model) / depths

def residual_depthweighted(params, pixels, wl_data, depths):
    # Polynomial of form q*x**2 + m*x+b
    q = params["q"].value
    m = params["m"].value
    b = params["b"].value
    Gap1 = params["Gap1"].value
    Gap2 = params["Gap2"].value
    Gap3 = params["Gap3"].value

    # Change spacing to add pixel gaps to chip pixel numbering
    new_pixels = np.array(pixels)
    chipmap4 = new_pixels > 3*1024
    chipmap3 = new_pixels > 2*1024
    chipmap2 = new_pixels > 1*1024
    new_pixels += Gap3*chipmap4
    new_pixels += Gap2*chipmap3
    new_pixels += Gap1*chipmap2

    model = q*new_pixels**2 + m*new_pixels + b

    return (wl_data - model) / depths


# Load data
Chipnames = ["Coordinates_CRIRE.2012-04-07T00:08:29.976_1.nod.ms.norm.sum.txt","Coordinates_CRIRE.2012-04-07T00:08:29.976_2.nod.ms.norm.sum.txt","Coordinates_CRIRE.2012-04-07T00:08:29.976_3.nod.ms.norm.sum.txt","Coordinates_CRIRE.2012-04-07T00:08:29.976_4.nod.ms.norm.sum.txt"]
PATH = "/home/jneal/Dropbox/PhD/"
#PATH = "/home/jneal/Dropbox/PhD/"
#"/home/jneal/Dropbox/PhD/"
pix1, wlen1, dpth1 = np.loadtxt(PATH+Chipnames[0], skiprows=1, unpack=True)
pix2, wlen2, dpth2 = np.loadtxt(PATH+Chipnames[1], skiprows=1, unpack=True)
pix3, wlen3, dpth3 = np.loadtxt(PATH+Chipnames[2], skiprows=1, unpack=True)
pix4, wlen4, dpth4 = np.loadtxt(PATH+Chipnames[3], skiprows=1, unpack=True)

#### Plot  telluric line First

import Obtain_Telluric as obt
from astropy.io import fits
import GaussianFitting as gf
from TellRemoval import airmass_scaling
Path = "/home/jneal/Phd/data/Crires/BDs-DRACS/HD30501-1/Fullreductionr-test-1dec2015/Combined_Nods/"

Chipnames = ["CRIRE.2012-04-07T00:08:29.976_1.nod.ms.norm.sum.fits", "CRIRE.2012-04-07T00:08:29.976_2.nod.ms.norm.sum.fits", "CRIRE.2012-04-07T00:08:29.976_3.nod.ms.norm.sum.fits", "CRIRE.2012-04-07T00:08:29.976_4.nod.ms.norm.sum.fits"]
#Pixel_offsets = [0, Gap1+1024, Gap1+Gap2+2*1024, Gap1+Gap2+Gap3+3*1024]  # Pixel values offset for each chip

## Telluric

hdr = fits.getheader(Path+Chipnames[0])
wl_lower = hdr["HIERARCH ESO INS WLEN STRT"]
datetime = hdr["DATE-OBS"]

wl_upper = fits.getheader(Path+Chipnames[3])["HIERARCH ESO INS WLEN END"]

obsdate, obstime = datetime.split("T")
obstime, __ = obstime.split(".")

tellpath = "/home/jneal/Phd/data/Tapas/"
tellname = obt.get_telluric_name(tellpath, obsdate, obstime) # to within the hour
print("Telluric Name", tellname)

tell_data, tell_header = obt.load_telluric(tellpath, tellname[0])

# Scale telluric lines to airmass
start_airmass = hdr["HIERARCH ESO TEL AIRM START"]
end_airmass = hdr["HIERARCH ESO TEL AIRM END"]
obs_airmass = (start_airmass + end_airmass) / 2

#print(tell_header)
tell_airmass = float(tell_header["airmass"])
#print(obs_airmass, type(obs_airmass))
#print(tell_airmass, type(tell_airmass))
tell_data[1] = airmass_scaling(tell_data[1], tell_airmass, obs_airmass)

# Sliced to wavelength measurement of detector
calib_data = gf.slice_spectra(tell_data[0], tell_data[1], wl_lower, wl_upper)

plt.plot(calib_data[0], calib_data[1], "-", label="Telluric")





Test_pxl1 = [pxl for pxl in pix1]
Test_pxl2 = [pxl + 1*1024 for pxl in pix2]
Test_pxl3 = [pxl + 2*1024 for pxl in pix3]
Test_pxl4 = [pxl + 3*1024 for pxl in pix4]
Test_wl1 = [wl for wl in wlen1]
Test_wl2 = [wl for wl in wlen2]
Test_wl3 = [wl for wl in wlen3]
Test_wl4 = [wl for wl in wlen4]
Test_dpth1 = [d for d in dpth1]
Test_dpth2 = [d for d in dpth2]
Test_dpth3 = [d for d in dpth3]
Test_dpth4 = [d for d in dpth4]

Combined_pxls = np.concatenate((pix1, pix2 + 1*1024, pix3 + 2*1024, pix4 + 3*1024))
Combined_wls = np.concatenate((wlen1, wlen2, wlen3, wlen4))
Combined_depths = np.concatenate((dpth1, dpth2, dpth3, dpth4))

#Combined_pixels = Test_pxl1 + Test_pxl2 + Test_pxl3 + Test_pxl4
#Combined_wls = Test_wl1 + Test_wl2 + Test_wl3 + Test_wl4
#Combined_depths = dpth1


params = Parameters()
params.add('q', value=0.0000001)
params.add('m', value=0.01)
params.add('b', value=2110)
params.add('Gap1', value=283)
params.add('Gap2', value=278)
params.add('Gap3', value=275)

out = minimize(residual, params, args=(Combined_pxls, Combined_wls))
outreport = lmfit.fit_report(out)


#Combined_map = [q, m, b]
Combined_map = [out.params["q"].value, out.params["m"].value, out.params["b"].value]
Gap1 = out.params["Gap1"].value
Gap2 = out.params["Gap2"].value
Gap3 = out.params["Gap3"].value
print("Fitted Gaps \n Gap1 = {}\nGap2 = {}\nGap3 = {}\n".format(Gap1, Gap2, Gap3))

### Dividing by Depths

out_with_depths = minimize(residual_depthweighted, params, args=(Combined_pxls, Combined_wls, Combined_depths))
outreport_with_depths  = lmfit.fit_report(out_with_depths)

Combined_map_depths = [out.params["q"].value, out.params["m"].value, out.params["b"].value]
Gap1_depths = out_with_depths.params["Gap1"].value
Gap2_depths = out_with_depths.params["Gap2"].value
Gap3_depths = out_with_depths.params["Gap3"].value
print("Fitted Gaps with Depths\n Gap1 = {}\nGap2 = {}\nGap3 = {}\n".format(Gap1_depths, Gap2_depths, Gap3_depths))



# Individaul fittings
params_chip1 = Parameters()
params_chip1.add('q', value=0.0000001)
params_chip1.add('m', value=0.01)
params_chip1.add('b', value=2110)
out1 = minimize(residual_individual, params_chip1, args=(Test_pxl1, Test_wl1))
outreport1 = lmfit.fit_report(out1)
Chip1_params = [out1.params["q"].value, out1.params["m"].value, out1.params["b"].value]
print("Chip1 Parameters = {}".format(Chip1_params))

params_chip2 = Parameters()
params_chip2.add('q', value=0.0000001)
params_chip2.add('m', value=0.01)
params_chip2.add('b', value=2110)
out2 = minimize(residual_individual, params_chip2, args=(Test_pxl2, Test_wl2))
outreport2 = lmfit.fit_report(out2)
Chip2_params = [out2.params["q"].value, out2.params["m"].value, out2.params["b"].value]
print("Chip2 Parameters = {}".format(Chip2_params))

params_chip3 = Parameters()
params_chip3.add('q', value=0.0000001)
params_chip3.add('m', value=0.01)
params_chip3.add('b', value=2110)
out3 = minimize(residual_individual, params_chip3, args=(Test_pxl3, Test_wl3))
outreport3 = lmfit.fit_report(out3)
Chip3_params = [out3.params["q"].value, out3.params["m"].value, out3.params["b"].value]
print("Chip3 Parameters = {}".format(Chip3_params))

params_chip4 = Parameters()
params_chip4.add('q', value=0.0000001)
params_chip4.add('m', value=0.01)
params_chip4.add('b', value=2110)
out4 = minimize(residual_individual, params_chip4, args=(Test_pxl4, Test_wl4))
outreport4 = lmfit.fit_report(out4)
Chip4_params = [out4.params["q"].value, out4.params["m"].value, out4.params["b"].value]
print("Chip4 Parameters = {}".format(Chip4_params))




GapOffsets = [0, Gap1, Gap1+Gap2, Gap1+Gap2+Gap3]
GapOffsets_depths = [0, Gap1_depths, Gap1_depths+Gap2_depths, Gap1_depths+Gap2_depths+Gap3_depths]
Test_pixels = [Test_pxl1, Test_pxl2, Test_pxl3, Test_pxl4]

wlmaps = [Chip1_params, Chip2_params, Chip3_params, Chip4_params]

for num, cname in enumerate(Chipnames):

    data = fits.getdata(Path+cname)
    Chip_data = data
    chip_pixel = np.array(range(1,1025))
    chip_pixel += num*1024   # pixels of chips to the left
    chip_pixel_gapped = chip_pixel + GapOffsets[num]  # pixels from detector gaps
    chip_pixel_gapped_depths = chip_pixel + GapOffsets_depths[num]
    Chip_wl_individual = np.polyval(wlmaps[num], chip_pixel)
    plt.plot(Chip_wl_individual, Chip_data, "--", label="Individual Map Chip {}".format(num+1))

    Chip_wl_comb = np.polyval(Combined_map, chip_pixel_gapped)
    plt.plot(Chip_wl_comb, Chip_data, label="Combined Map Chip {}".format(num+1))

    Chip_wl_comb_depths = np.polyval(Combined_map_depths, chip_pixel_gapped_depths)
    plt.plot(Chip_wl_comb_depths, Chip_data, label="Combined depths Map Chip {}".format(num+1))




ax1 = plt.gca()
ax1.get_xaxis().get_major_formatter().set_useOffset(False)
plt.legend(loc=0)


#"Print results"
#print(outreport4)

print("Normal Combined Report\n")
print(outreport)
print("Depths Combined Report\n")
print(outreport_with_depths)

plt.show()

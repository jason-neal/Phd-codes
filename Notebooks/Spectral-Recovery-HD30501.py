
# coding: utf-8

# # Spectral Recovery of HD30501
# First try: March 2016
# Added Bokeh May 2016

# In[2]:

### Load modules and Bokeh
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Seaborn, useful for graphics
import seaborn as sns

# Magic function to make matplotlib inline; other style specs must come AFTER
get_ipython().magic(u'matplotlib inline')

# Import Bokeh modules for interactive plotting
import bokeh.io
import bokeh.mpl
import bokeh.plotting

# This enables SVG graphics inline.  There is a bug, so uncomment if it works.
get_ipython().magic(u"config InlineBackend.figure_formats = {'svg',}")

# This enables high resolution PNGs. SVG is preferred, but has problems
# rendering vertical and horizontal lines
#%config InlineBackend.figure_formats = {'png', 'retina'}

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 1, 
      'axes.labelsize': 12, 
      'axes.titlesize': 14, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style('darkgrid', rc=rc)

# Set up Bokeh for inline viewing
bokeh.io.output_notebook()


# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ### Load in the Telluric corrected spectra
# 
# ##### For now just with the wavecal values 

# In[4]:

#path = 'C:/Users/Jason/Dropbox/PhD/hd30501-Wavecal-march16/'
#name1 = "CRIRE.2012-04-07T00-08-29.976_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits"
#name2 = "CRIRE.2012-08-01T09-17-30.195_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits"
#name3 = "CRIRE.2012-08-02T08-47-30.843_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits"
#name4 = "CRIRE.2012-08-06T09-42-07.888_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits" 

path="/home/jneal/Phd/data/Hd30501-tellcorrected-test/"
#2012-08-06T09-42-07.888_3.nod.ms.norm.sum.wavecal
name1 = "CRIRE.2012-04-07T00:08:29.976_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits"
name2 = "CRIRE.2012-08-01T09:17:30.195_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits"
name3 = "CRIRE.2012-08-02T08:47:30.843_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits"
name4 = "CRIRE.2012-08-06T09:42:07.888_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits" 

# Names for all 4 detectors
name1_chips = [name1[:30]+str(i)+name1[31:] for i in range(1,5)]
name2_chips = [name2[:30]+str(i)+name2[31:] for i in range(1,5)]
name3_chips = [name3[:30]+str(i)+name3[31:] for i in range(1,5)]
name4_chips = [name4[:30]+str(i)+name4[31:] for i in range(1,5)]


# In[5]:

detector = 3   # choose which chip to look at

Obs1 = fits.getdata(path + name1_chips[detector]) 
hdr1 = fits.getheader(path + name1_chips[detector]) 
Obs2 = fits.getdata(path + name2_chips[detector])
hdr2 = fits.getheader(path + name2_chips[detector])
Obs3 = fits.getdata(path + name3_chips[detector])
hdr3 = fits.getheader(path + name3_chips[detector])
Obs4 = fits.getdata(path + name4_chips[detector])
hdr4 = fits.getheader(path + name4_chips[detector])
print("Names of the different data vectors in the fits file")
print("Obs1 Column names = {}".format(Obs1.columns.names))
wl1 = Obs1["Wavelength"]
I1_uncorr = Obs1["Extracted_DRACS"]
I1 = Obs1["Corrected_DRACS"]

#print("Obs2 Column names = {}".format(Obs2.columns.names))
wl2 = Obs2["Wavelength"]
I2_uncorr = Obs2["Extracted_DRACS"]
I2 = Obs2["Corrected_DRACS"]
Tell_2 = Obs2["Interpolated_Tapas"]

#print("Obs3 Column names = {}".format(Obs3.columns.names))
wl3 = Obs3["Wavelength"]
I3_uncorr = Obs3["Extracted_DRACS"]
I3 = Obs3["Corrected_DRACS"]

#print("Obs4 Column names = {}".format(Obs4.columns.names))
wl4 = Obs4["Wavelength"]
I4_uncorr = Obs4["Extracted_DRACS"]
I4 = Obs4["Corrected_DRACS"]

print("Data from Detectors is now loaded")


# Plotting the telluric corrected spectra for the different observations here shows that the BERV corrected tapas is good with all the stellar lines lining up very nicely in the plot below without having to apply a correction myself. I previously showed that the tapas BERV correction is correct and gives the same spectra as if I took the non BERV tapas spectra and applied BERV correction from PyAstronomy. Althought I do not know if there is a significant affect between applying it before or after the pixel mapping. 
# 
# BERV correction aligns the stellar lines.

# In[11]:

print("Colour code")
print("blue = HD30501-1")
print("red = HD30501-2a")
print("green = HD30501-2b")
print("black = HD30501-3")
print("dashed are the tapas models")


# Plot detector 
plt.figure()
plt.plot(wl1 , I1,"b" , label="1" )
plt.plot(wl2 , I2, "r", label="2")
plt.plot(wl3 , I3, "g", label="3")
plt.plot(wl4 , I4, "k", label="4")
#plt.legend(loc=0)
plt.title("All Telluric Corrected observations of HD30501 detector 1")
plt.xlabel("Wavelength")
#plt.show()
# Make it interactive with Bokeh
bokeh.plotting.show(bokeh.mpl.to_bokeh())

# plt spectra and telluric lines to check that there is good wavelength calibration
plt.figure()
plt.plot(wl1 , I1_uncorr, "b", label="1 Obs")
plt.plot(wl1 , Obs1["Interpolated_Tapas"], "--b", label="1 Telluric")
plt.plot(wl2 , I2_uncorr, "r", label="2 Obs")
plt.plot(wl2 , Obs2["Interpolated_Tapas"], "--r", label="2 Telluric")
plt.plot(wl3 , I3_uncorr, "g", label="3 Obs")
plt.plot(wl3 , Obs3["Interpolated_Tapas"], "--g", label="3 Telluric")
plt.plot(wl4 , I4_uncorr, "k", label="4 Obs")
plt.plot(wl4 , Obs4["Interpolated_Tapas"], "--k", label="4 Telluric")
#plt.legend(loc=0)
plt.ylabel("")
plt.xlabel("Wavelength")
plt.title("Uncorrected observations with telluric models to show wl calibration")
#plt.show()
# Make it interactive with Bokeh
bokeh.plotting.show(bokeh.mpl.to_bokeh())


# To be able to subtract the spectra from each other they need to be interpolated to the same wavelength scale. For this target 3 of the wavelengths are very close together while the 1st is (0.1 nm) different at each pixel so will interpolate to the 2nd observations wavelength.

# In[13]:

plt.figure()
plt.plot(wl1, "o-", label="1")
plt.plot(wl2, "+-",label="2")
plt.plot(wl3, "*-", label="3")
plt.plot(wl4, ".-",  label="4")
plt.legend(loc=0)
plt.title("Wavelength values for pixels of each observation of \ndetector 1")
plt.xlabel("pixel")
plt.ylabel("Wavelength")
plt.xlim([600,610])
ylimits = [[2118.5,2119],[2133.4,2133.8],[2147.4,2147.8],[2160.9,2161.2]]
plt.ylim(ylimits[detector])


ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
#plt.show()
# Make it interactive with Bokeh
#bokeh.plotting.show(bokeh.mpl.to_bokeh())


# ### Interpolation to same wavelengths

# In[14]:

from scipy.interpolate import interp1d

# Using bounds_error=False to make it work if outside wavelgnth range
# if fill-value is not given then will replace outisde bounds values with NaNs
interp_1 = interp1d(wl1, I1, kind="linear", bounds_error=False)   
interp_2 = interp1d(wl2, I2, kind="linear", bounds_error=False)
interp_3 = interp1d(wl3, I3, kind="linear", bounds_error=False)
interp_4 = interp1d(wl4, I4, kind="linear", bounds_error=False)

wl = wl2     # Specify here sto easily change the reference wavelength
I1_interp = interp_1(wl) 
I2_interp = interp_2(wl) 
I3_interp = interp_3(wl) 
I4_interp = interp_4(wl) 


# In[15]:

# Plot detector 1
plt.figure()
plt.plot(wl ,I1_interp, label="1")
plt.plot(wl ,I2_interp, label="2")
plt.plot(wl ,I3_interp, label="3")
plt.plot(wl ,I4_interp, label="4")
plt.legend(loc=0)
plt.title("All Telluric Corrected observations of HD30501")
#plt.show()
# Make it interactive with Bokeh
bokeh.plotting.show(bokeh.mpl.to_bokeh())


# ### Subtraction of the different observations

# In[21]:


plt.figure()
plt.suptitle("Subtraction of different observations from detector {}".format(detector), fontsize=16)
plt.subplot(711)
plt.plot(wl, I1_interp-I2_interp, label="Obs 1 - Obs 2")
plt.title("Observation 1 - Observation 2")
#plt.legend(loc=0)

plt.subplot(712)
plt.plot(wl, I1_interp-I3_interp, label="Obs 1 - Obs 3")
plt.title("Observation 1 - Observation 3")
#plt.legend(loc=0)

plt.subplot(713)
plt.plot(wl, I1_interp-I4_interp, label="Obs 1 - Obs 4")
plt.title("Observation 1 - Observation 4")
#plt.legend(loc=0)

plt.subplot(714)
plt.plot(wl, I2_interp-I3_interp, label="Obs 2 - Obs 3")
plt.title("Observation 2 - Observation 3")
#plt.legend(loc=0)

plt.subplot(715)
plt.plot(wl, I2_interp-I4_interp, label="Obs 2 - Obs 4")
plt.title("Observation 2 - Observation 4")
#plt.legend(loc=0)

plt.subplot(716)
plt.plot(wl, I3_interp-I4_interp, label="Obs 3 - Obs 4")
plt.title("Observation 3 - Observation 4")
#plt.legend(loc=0)

plt.subplot(717)
plt.plot(wl2, 1-Tell_2, 'r', label="Tapas")
plt.plot(wl2, 1-I2_uncorr, 'k', label="Exctracted")
plt.title("Telluric line locations (minus 1)")
#plt.legend(loc=0)
#plt.show()
# Make it interactive with Bokeh
bokeh.plotting.show(bokeh.mpl.to_bokeh())


# In[ ]:




# ### Convert ra and dec to decimal degrees
# This is if need to do BERV corrections manually
# 
# To go into PyAstronomy helio-center vecolity calculations.

# In[13]:

def ra2deg(ra):
    split = ra.split(":")
    deg = float(split[0])*15.0 + float(split[1])/4.0 + float(split[2])/240.0 
    return deg

def dec2deg(dec):
	#  degrees ( ° ), minutes ( ' ), and seconds ( " )
	#convert to degrees in decimal
    split = dec.split(":")
    print(split)
    if float(split[0]) < 0:
        deg = abs(float(split[0])) + (float(split[1]) + (float(split[2])/60) )/60
        deg *= -1 
    else:
        deg = float(split[0]) + (float(split[1]) + (float(split[2])/60) )/60 
    return deg


# #### Baycenter Corrections
# http://www.hs.uni-hamburg.de/DE/Ins/Per/Czesla/PyA/PyA/pyaslDoc/aslDoc/baryvel.html
# ##### The telluric spectra should possibly be redone with no barycenter correction so that we can alpply our own... Currently for HD30501 bayrcorr was used i think, check 
# The baryvel() and baryCorr() functions allow to calculate Earth’s helio- and barycentric motion and project it onto a given direction toward a star. The helcorr() includes a barycentric correction including the effect caused by the rotating Earth.
# 
# from PyAstronomy import baryCorr
# baryCorr(jd, ra, dec, deq=0.0)
# ### PyAstronomy.pyasl.baryCorr(jd, ra, dec, deq=0.0)
#     Calculate barycentric correction.
# ###### Parameters :	
# jd : float
#     The time at which to calculate the correction.
# ra : float
#     Right ascension in degrees.
# dec : float
#     Declination in degrees.
# deq : float, optional
#     The mean equinox of barycentric velocity calculation (see bryvel()). If zero, it is assumed to be the same as jd.
# ###### Returns :	
# Projected heliocentric velocity : float
#     Heliocentric velocity toward star [km/s]
# Projected barycentric velocity : float
#     Barycentric velocity toward star [km/s]
# 
# ###     PyAstronomy.pyasl.helcorr(obs_long, obs_lat, obs_alt, ra2000, dec2000, jd, debug=False)
# 
# Calculate barycentric velocity correction.
# 
# This function calculates the motion of an observer in the direction of a star. In contract to baryvel() and baryCorr(), the rotation of the Earth is taken into account.
# 
# 
# ###### Parameters :	
# obs_long : float
#     Longitude of observatory (degrees, eastern direction is positive)
# obs_lat : float
#     Latitude of observatory [deg]
# obs_alt : float
#     Altitude of observatory [m]
# ra2000 : float
#     Right ascension of object for epoch 2000.0 [deg]
# dec2000 : float
# 
#     Declination of object for epoch 2000.0 [deg]
# 
# jd : float
# 
#     Julian date for the middle of exposure.
# 
# ###### Returns :	
# Barycentric correction : float
#     The barycentric correction accounting for the rotation of the Earth, the rotation of the Earth’s center around the Earth-Moon barycenter, and the motion of the Earth-Moon barycenter around the center of the Sun [km/s].
# 
# HJD : float
#     Heliocentric Julian date for middle of exposure.
# 
# 
# 
# Bayrcenter correction velocities from the tapas files to check with my calculations
# 
# #### HD30501-1 
# bayrtcor = 135.854484 ,
# baryvcor = -5.722224
# #### HD30501-2a
# barytcor = -33.867880 ,
# baryvcor = 9.57472
# #### hd30501-2b 
# barytcor = -20.212341, 
# baryvcor = 9.662005
# #### HD30501-3
# barytcor = -31.206540, 
# baryvcor = 9.619015

# In[ ]:

# barycentric correction
from PyAstronomy import baryCorr
from PyAstronomy import pyasl

baryCorr(jd, ra, dec, deq=0.0)




# In[ ]:



# Plot the spectra together 

plt.figure()
plt.plot(wl1, I1, label="Obs1")
plt.plot(wl2, I2, label="Obs2")
plt.plot(wl3, I2, label="Obs3")
plt.plot(wl4, I4, label="Obs4")
plt.show()






# In[15]:

# Subtract 2 spectra and see the differences
#need to interpolate to a common wavelength scale



# In[ ]:




# # Cross Correlation method to find best shift for the spectra

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# ## Examples of using PyAstronomy:
#     

# In[ ]:

from PyAstronomy import pyasl

# Coordinates of European Southern Observatory
# (Coordinates of UT1)
longitude = 289.5967661
latitude = -24.62586583
altitude = 2635.43

# Coordinates of HD 12345 (J2000)
ra2000 = 030.20313477
dec2000 = -12.87498346

# (Mid-)Time of observation
jd = 2450528.2335

# Calculate barycentric correction (debug=True show
# various intermediate results)
corr, hjd = pyasl.helcorr(longitude, latitude, altitude,             ra2000, dec2000, jd, debug=True)

print("Barycentric correction [km/s]: ", corr)
print("Heliocentric Julian day: ", hjd)


# In[ ]:




# In[ ]:




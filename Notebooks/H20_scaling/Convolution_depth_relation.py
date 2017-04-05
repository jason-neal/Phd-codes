
# coding: utf-8

# # Relation between Convoled Spectra and not convolved Spectra of H20.
# 
# 

# In[ ]:

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
get_ipython().magic('matplotlib inline')

# Import Bokeh modules for interactive plotting
import bokeh.io
import bokeh.mpl
import bokeh.plotting

# This enables SVG graphics inline.  There is a bug, so uncomment if it works.
get_ipython().magic("config InlineBackend.figure_formats = {'svg',}")

# This enables high resolution PNGs. SVG is preferred, but has problems
# rendering vertical and horizontal lines
#%config InlineBackend.figure_formats = {'png', 'retina'}

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 1, 
      'axes.labelsize': 14, 
      'axes.titlesize': 16, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style('darkgrid', rc=rc)

# Set up Bokeh for inline viewing
bokeh.io.output_notebook()


# ## Load non convolved telluric

# In[ ]:

import Obtain_Telluric as obt

tapas_h20 = "tapas_2012-04-07T00-24-03_ReqId_12_No_Ifunction_barydone-NO.ipac"

tapas_h20_data, tapas_h20_hdr = obt.load_telluric("", tapas_h20)
tapas_h20_airmass = float(tapas_h20_hdr["airmass"])

print("Telluric Airmass ", tapas_h20_airmass)
try:
    tapas_h20_respower = int(float((tapas_h20_hdr["respower"])))
except:
    tapas_h20_respower = "Nan"
print("Telluric Resolution Power =", tapas_h20_respower)


# In[ ]:

# Load convolved


# In[ ]:

#conv_wav, conv_flux = np.loadtxt("Convolved_50000_tapas_allchips.txt", delimiter="'",unpack=True)

conv_wav = np.loadtxt("Convolved_50000_tapas_wavelength_allchips.txt")
conv_flux = np.loadtxt("Convolved_50000_tapas_transmitance_allchips.txt")

print(conv_wav)
print(conv_flux)


# In[ ]:

orig_flux = np.array([flux for wav, flux in zip(tapas_h20_data[0], tapas_h20_data[1]) if wav in conv_wav])

plt.plot(orig_flux, conv_flux, "o")
pl t.title("Affect of Convolution R=50000")
plt.xlabel("Original Flux")
plt.ylabel("Convolved Flux\nR=50000")

bokeh.plotting.show(bokeh.mpl.to_bokeh())



# In[ ]:

plt.plot(tapas_h20_data[0],tapas_h20_data[1])
plt.plot(conv_wav, conv_flux)
plt.xlabel("Wavelength")
plt.ylabel("Flux")

bokeh.plotting.show(bokeh.mpl.to_bokeh())


# In[ ]:

#Wavelenght density
plt.plot(tapas_h20_data[0][1:],tapas_h20_data[0][1:]-tapas_h20_data[0][:-1])
plt.ylabel("Delta Wavelength")
plt.xlabel("Wavelength (nm)")
plt.title("Distribution of wavelength is not uniform")
bokeh.plotting.show(bokeh.mpl.to_bokeh())


# ## Testing dividing by number of values in gaussian convolution

# In[ ]:

# Convolved_50000_tapas_wavelength_allchips_dividebynumber.txt
# Testing dividing each value by number of points in convolution gaussian
conv_wav_divide = np.loadtxt("Convolved_50000_tapas_wavelength_allchips_dividebynumber.txt")
conv_flux_divide = np.loadtxt("Convolved_50000_tapas_transmitance_allchips_dividebynumber.txt")

# Convolution with division by # of values in convolution did not work well. So proably have to divide by a fitted line.


# In[ ]:

plt.plot(orig_flux, conv_flux_divide, "o")
plt.title("Divided values \n Affect of Convolution R=50000")
plt.xlabel("Original Flux")
plt.ylabel("Convolved Flux\nR=50000")

bokeh.plotting.show(bokeh.mpl.to_bokeh())


# In[ ]:

plt.plot(tapas_h20_data[0],tapas_h20_data[1])
plt.plot(conv_wav_divide, conv_flux_divide)
plt.plot(conv_wav, conv_flux)
plt.xlabel("Wavelenght")
plt.ylabel("Flux with division")

bokeh.plotting.show(bokeh.mpl.to_bokeh())


# In[ ]:




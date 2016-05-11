
# coding: utf-8

# # Relation between Convoled Spectra and not convolved Spectra of H20.
# 
# 

# In[1]:

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

# In[2]:

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


# In[3]:

# Load convolved


# In[5]:

conv_wav, conv_flux = np.loadtxt("Convolved_50000_tapas_allchips.txt", unpack=True)



# In[ ]:




# In[ ]:

orig_flux = np.array([flux for flux in tapas_h20_data[0] if flux in conv_wav])

plt.plot(orig_flux, conv_flux, "o")
plt.title("Affect of Convolution R=50000")
plt.xlabel("Original Flux")
plt.ylabel("Convolved Flux\nR=50000")

bokeh.plotting.show(bokeh.mpl.to_bokeh())


# In[ ]:




# In[ ]:





# coding: utf-8

# # H20 Scaling for Telluric Correction
# 
# Notebook for developing ideas to go into TellRemoval code.
# 
# Need to apply scaling of T^x to transmision of water at full resolving power and then apply a kernal to apply in at resolution of CRIRES.
# 
# Fit to the observed data (Probably with the other lines removed) to fnd the best x to apply for the correction. (Gives flatest result or zero linewidth.) 
# 

# In[3]:

### Load modules and Bokeh
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import numpy as np

# Import pyplot for plotting
import matplotlib.pyplot as plt

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


# ### Load in Observed Data

# In[4]:




# In[ ]:

name1 = "CRIRE.2012-04-07T00-08-29.976_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits"
name2 = "CRIRE.2012-08-01T09-17-30.195_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits"
name3 = "CRIRE.2012-08-02T08-47-30.843_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits"
name4 = "CRIRE.2012-08-06T09-42-07.888_1.nod.ms.norm.sum.wavecal.tellcorr.test.fits" 


# ### Load in the tapas data

# In[ ]:




# ### Plot the data
# Including the 3 tapas models to show they align well and are consistent.
# 

# In[ ]:




# ### Remove non-H20 lines
# (Use telluric removal modules)
# And plot the result.  

# In[ ]:




# ### Convole instrument profile function:
# To use inside fit

# In[ ]:




# ### Fit best scaling power.
# Does each chip need a differnet scaling power?
# 

# In[ ]:




# ### Apply correction with best scaling power:
# 
# And plot the result.

# In[ ]:




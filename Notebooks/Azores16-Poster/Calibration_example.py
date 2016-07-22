
# coding: utf-8

# # Wavelenght Calibration Example for Azores poster.

# In[ ]:

from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import numpy as np
# Import Bokeh modules for interactive plotting
import bokeh.io
import bokeh.mpl
import bokeh.plotting

# Include masking
from bokeh.plotting import figure, show, output_file, gridplot, vplot
from bokeh.models import BoxAnnotation
from bokeh.models import Range1d


# This enables SVG graphics inline.  There is a bug, so uncomment if it works.
get_ipython().magic("config InlineBackend.figure_formats = {'svg',}")

# This enables high resolution PNGs. SVG is preferred, but has problems
# rendering vertical and horizontal lines
#%config InlineBackend.figure_formats = {'png', 'retina'}

# Set up Bokeh for inline viewing
bokeh.io.output_notebook()


# In[ ]:

from astropy.modeling import models, fitting
#from astropy.modeling import SummedCompositeModel
from astropy.modeling.models import Gaussian1D
#from astropy.modeling.models import custom_model_1d

## my functions:
    #x = np.linspace(0, 100, points)
def mk_gaussian_sum(x, amplitudes, means, stddevs, noise=1/200):
    '''Create 3 spectral lines by adding 3 gaussians together'''
    assert len(amplitudes) == len(means), ' Not the same length inputs'
    assert len(amplitudes) == len(stddevs), ' Not the same length inputs'
    y = np.ones_like(x)
    for i in range(len(amplitudes)):
        g = Gaussian1D(amplitude=amplitudes[i], mean=means[i], stddev=stddevs[i])
        # print g(x)
        y -= g(x) # Adding each Gaussian  
    y +=  np.random.normal(0, noise, x.shape)		# Adding some noise to our data

    return y


# In[ ]:

# Wavelength range
wavelength = np.linspace(2110, 2113, 60)
stellar_amp = [.4, .5]
stellar_means = [2111.1, 2112.1]
stellar_stddevs = [.1, .06]
stellar_lines = mk_gaussian_sum(wavelength, stellar_amp, stellar_means, stellar_stddevs,  noise=1/100)
stellar_fit = mk_gaussian_sum(wavelength, stellar_amp, stellar_means, stellar_stddevs,  noise=1/100000000)

telluric_amp = [.1, .3, .25]
telluric_means = [2110.2, 2111.3, 2112.6]
telluric_stddevs = [0.05, .07, .05]
telluric_lines = mk_gaussian_sum(wavelength, telluric_amp, telluric_means, telluric_stddevs, noise=1/100)
telluric_fit = mk_gaussian_sum(wavelength, telluric_amp, telluric_means, telluric_stddevs, noise=1/100000000)
#print(telluric_lines)
#print(stellar_lines)

combined_lines = stellar_lines * telluric_lines


# In[ ]:

# Get different combinations of shifts
shifted_wl1 = wavelength + 0
#shifted_wl2 = wavelength - 0  

shifted_stellar_lines1 = mk_gaussian_sum(shifted_wl1, stellar_amp, stellar_means, stellar_stddevs)
#shifted_stellar_lines2 = mk_gaussian_sum(shifted_wl2, stellar_amp, stellar_means, stellar_stddevs)

shift1 = shifted_stellar_lines1 * telluric_lines
#shift2 = shifted_stellar_lines2 * telluric_lines


# In[ ]:

# using bokeh
from bokeh.io import gridplot, output_file, show
from bokeh.plotting import figure, show, output_file, gridplot, vplot
from bokeh.models import BoxAnnotation
from bokeh.plotting import figure
from bokeh.models.glyphs import Text

s1 = figure(width=500, height=220, title=None)
s1.line(wavelength, telluric_lines, legend="Telluric Model", color="black", line_width=2)
s1.line(wavelength, telluric_fit, legend="Telluric fit", color="red", line_width=2, line_dash="dashed")
s1.xaxis.axis_label = 'Wavelength (nm)'
s1.yaxis.axis_label = 'Transmittance'
s1.legend.location = "bottom_left"
s1.title_text_font_size = "14pt"
s1.xaxis.axis_label_text_font_size = "12pt"
s1.yaxis.axis_label_text_font_size = "12pt"
s1.legend.border_line_color = None

g1 = Text(x=2111, y=0.6, text="wl_1", angle=0, text_color="Black", text_align="center")
s1.add_glyph(g1)
g2 = Text(x=2112, y=0.9, text="wl_2", angle=0, text_color="Black", text_align="center")
s1.add_glyph(g2)
g3 = Text(x=2113, y=0.9, text="wl_3", angle=0, text_color="Black", text_align="center")
s1.add_glyph(g3)


s2 = figure(width=500, height=220, y_range=s1.y_range, title=None)
s2.line(range(len(shift1)), shift1, legend="Observation", color="black", line_width=2)
s2.xaxis.axis_label = 'Pixel Number'
s2.yaxis.axis_label = 'Norm Intensity'
s2.legend.location = "bottom_left"
s2.title_text_font_size = "14pt"
s2.xaxis.axis_label_text_font_size = "12pt"
s2.yaxis.axis_label_text_font_size = "12pt"
s2.legend.border_line_color = None
# Add text around 20, 0.7 

s2.add_glyph(Text(x=3, y=0.9, text="Telluric", angle=0, text_color="Black", text_align="center"))
s2.add_glyph(Text(x=19, y=0.9, text="Blended", angle=0, text_color="Black", text_align="center"))
s2.add_glyph(Text(x=34, y=0.9, text="Stellar", angle=0, text_color="Black", text_align="center")) 
s2.add_glyph(Text(x=50, y=0.6, text="Telluric", angle=0, text_color="Black", text_align="center"))

#glyph = Text(x="x", y="y", text="text", angle=0, text_color="#96deb3")
#plot.add_glyph(source, glyph)

s3 = figure(width=500, height=220, y_range=s1.y_range, title=None)
s3.line(range(len(shift1)), shift1, legend="Observation", color="black", line_width=2)
s3.line(range(len(shift1)), telluric_fit, legend="Telluric fit", color="red", line_width=2, line_dash="dashed")
s3.line(range(len(shift1)), stellar_fit, legend="Stellar fit", color="blue", line_width=2, line_dash="dotdash")
s3.xaxis.axis_label = 'Pixel Number'
s3.yaxis.axis_label = 'Norm Intensity'
s3.legend.location = "bottom_left"
s3.title_text_font_size = "14pt"
s3.xaxis.axis_label_text_font_size = "12pt"
s3.yaxis.axis_label_text_font_size = "12pt"
s3.legend.border_line_color = None

#s3.add_glyph(Text(x=6, y=0.9, text="wl_1", angle=0, text_color="Black"), text_align="center")
#s3.add_glyph(Text(x=27, y=0.7, text="wl_2", angle=0, text_color="Black"), text_align="center")
#s3.add_glyph(Text(x=54, y=0.7, text="wl_3", angle=0, text_color="Black"), text_align="center")

p = gridplot([[s2], [s1], [s3]])

show(p)
print("done")


# # Wavelenght map for poster

# In[ ]:

# Quad fit
#wavelength = np.linspace(2110, 2113, 60)
telluric_means = [2110.2, 2111.3, 2112.6]
#index value of these values
telluric_means = [2110.2, 2111.3, 2112.6]
indexes = [np.argmin(abs(wavelength - mean)) for mean in telluric_means]


map = figure(width=500, height=220)
map.line(np.arange(len(wavelength)), wavelength, line_color="black", line_width=2)
map.circle(indexes, telluric_means, size=8, line_color="blue")

map.xaxis.axis_label = 'Pixel values'
map.yaxis.axis_label = 'Wavelength (nm)'
map.legend.location = "bottom_left"
map.title_text_font_size = "14pt"
map.xaxis.axis_label_text_font_size = "12pt"
map.yaxis.axis_label_text_font_size = "12pt"
map.legend.border_line_color = None

show(map)


# In[ ]:




# In[ ]:




# In[ ]:




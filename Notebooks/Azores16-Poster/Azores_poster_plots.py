
# coding: utf-8

# # Azores poster plots
# 

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import Obtain_Telluric as obt
from Get_filenames import get_filenames
get_ipython().magic('matplotlib inline')


# In[ ]:

# Import Bokeh modules for interactive plotting
import bokeh.io
import bokeh.mpl
import bokeh.plotting
get_ipython().magic("config InlineBackend.figure_formats = {'svg',}")
# Set up Bokeh for inline viewing
bokeh.io.output_notebook()


# In[ ]:

# Parameters to alter to change spectra seen
chip_num = 1
obs_num = "1"
ref_num = "3"
target = "HD30501-" + obs_num
reference_target = "HD30501-"+ ref_num    # should be different from target

if target == reference_target:
   raise ValueError("Reference target should be different from target")


# In[ ]:


### Dracs data
#dracs_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/{0}/Combined_Nods/".format(target)
dracs_path = "C:/Users/Jason/Documents/PhD/Phd-codes/Notebooks/HD30501_data/{0}/".format(obs_num)
#dracs_path = "../HD30501_data/{0}".format(obs_num)
dracs_name = get_filenames(dracs_path, "CRIRE.*","*{}.nod.ms.norm.sum.wavecal.fits".format(chip_num))

dracs_name = dracs_path + dracs_name[0]


# In[ ]:

# Dracs data load

#dracs_data = fits.getdata(dracs_names[Chip_num-1])
#dracs_hdr = fits.getheader(dracs_names[Chip_num-1]) 
dracs_data = fits.getdata(dracs_name)
dracs_hdr = fits.getheader(dracs_name) 

dracs_wl = dracs_data["Wavelength"]
dracs_I = dracs_data["Extracted_DRACS"]

# normalize dracs
maxes = dracs_I[(dracs_I < 1.2)].argsort()[-50:][::-1]
dracs_I = dracs_I / np.median(dracs_I[maxes])


# In[ ]:

# Load tapas file
tapas_path = dracs_path
tapas_name = get_filenames(tapas_path, "tapas_*","*ReqId_10*")[0]

Tapas_data, Tapas_hdr = obt.load_telluric(tapas_path, tapas_name)
tell_wl = Tapas_data[0]
tell_I = Tapas_data[1]

# normalize dracs
maxes = tell_I[(tell_I < 1.2)].argsort()[-50:][::-1]
#tell_I = tell_I / np.median(tell_I[maxes])

#wl limit
wlmask = (tell_wl > dracs_wl[0]/1.0001) & (tell_wl < dracs_wl[-1]*1.0001)
tell_wl = tell_wl[wlmask]
tell_I = tell_I[wlmask] 


# In[ ]:

# Corrected values
#dracs_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/{0}/Combined_Nods/".format(target)
#dracs_path = "../HD30501_data/{0}".format(obs_num)
dracs_path = "C:/Users/Jason/Documents/PhD/Phd-codes/Notebooks/HD30501_data/{0}/".format(obs_num)

tellcorr_name = get_filenames(dracs_path, "CRIRE.*","*{}.nod.ms.norm.sum.wavecal.tellcorr.fits".format(chip_num))
h20tellcorr_name = get_filenames(dracs_path, "CRIRE.*","*{}.nod.ms.norm.sum.wavecal.h2otellcorr.fits".format(chip_num))
print(tellcorr_name)
tellcorr_name = dracs_path + tellcorr_name[0]
h20tellcorr_name = dracs_path + h20tellcorr_name[0]

tellcorr_data = fits.getdata(tellcorr_name)
#print(tellcorr_data.columns)
tellcorr_hdr = fits.getheader(tellcorr_name) 
tellcorr_wl = tellcorr_data["Wavelength"]
tellcorr_I = tellcorr_data["Corrected_DRACS"]
tellcorr_tell = tellcorr_data["Interpolated_Tapas"]   # for masking

h20tellcorr_data = fits.getdata(h20tellcorr_name)
#print(h20tellcorr_data.columns)
h20tellcorr_hdr = fits.getheader(h20tellcorr_name) 
h20tellcorr_wl = h20tellcorr_data["Wavelength"]
h20tellcorr_I = h20tellcorr_data["Corrected_DRACS"]


# ### Load Reference Target
# Also Berv corrected

# In[ ]:

### Reference data 
# Same as above just a different target
#reference_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/{0}/Combined_Nods/".format(reference_target)
reference_path = "C:/Users/Jason/Documents/PhD/Phd-codes/Notebooks/HD30501_data/{0}/".format(ref_num)
reftellcorr_name = get_filenames(reference_path, "CRIRE.*","*{}.nod.ms.norm.sum.wavecal.tellcorr.fits".format(chip_num))
refh20tellcorr_name = get_filenames(reference_path, "CRIRE.*","*{}.nod.ms.norm.sum.wavecal.h2otellcorr.fits".format(chip_num))

######################################3 TESTING only
#reference_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/{0}/Combined_Nods/Bervcorrected_tapas/".format(reference_target)
#reftellcorr_name = get_filenames(reference_path, "CRIRE.*","*{}.nod.ms.norm.sum.wavecal.tellcorr.test.fits".format(chip_num))
#refh20tellcorr_name = get_filenames(reference_path, "CRIRE.*","*{}.nod.ms.norm.sum.wavecal.tellcorr.test.fits".format(chip_num))

###########################################
print(reftellcorr_name)

reftellcorr_name = reference_path + reftellcorr_name[0]
refh20tellcorr_name = reference_path + refh20tellcorr_name[0]

reftellcorr_data = fits.getdata(reftellcorr_name)
reftellcorr_hdr = fits.getheader(reftellcorr_name) 
reftellcorr_wl = reftellcorr_data["Wavelength"]
reftellcorr_I = reftellcorr_data["Corrected_DRACS"]
reftellcorr_tell = reftellcorr_data["Interpolated_Tapas"]   # for masking

refh20tellcorr_data = fits.getdata(refh20tellcorr_name)
refh20tellcorr_hdr = fits.getheader(refh20tellcorr_name) 
refh20tellcorr_wl = h20tellcorr_data["Wavelength"]
refh20tellcorr_I = h20tellcorr_data["Corrected_DRACS"]
refh20tellcorr_tell = h20tellcorr_data["Interpolated_Tapas"]  # for masking


# In[ ]:




# In[ ]:

# Make barycorr fucntion
import time 
import datetime
from PyAstronomy import pyasl

def barycorr_CRIRES(wavelength, flux, header):
   """
   Calculate Heliocenteric correction values and apply to spectrum.
   
   SHOULD test again with bary and see what the  difference is.
   """"

    longitude = float(header["HIERARCH ESO TEL GEOLON"])
    latitude = float(header["HIERARCH ESO TEL GEOLAT"])
    altitude = float(header["HIERARCH ESO TEL GEOELEV"])

    ra = header["RA"]    # CRIRES RA already in degrees 
    dec = header["DEC"]  # CRIRES hdr DEC already in degrees

# Pyastronomy helcorr needs the time of observation in julian Days
##########################################################################################
    Time =  header["DATE-OBS"]    # Observing date  '2012-08-02T08:47:30.8425'
# Get Average time **** from all raw files!!!  #################################################################

    wholetime, fractionaltime = Time.split(".")
    Time_time = time.strptime(wholetime, "%Y-%m-%dT%H:%M:%S")
    dt = datetime.datetime(*Time_time[:6])   # Turn into datetime object
    # Account for fractions of a second
    seconds_fractionalpart = float("0." + fractionaltime) / (24*60*60)   # Divide by seconds in a day

    # Including the fractional part of seconds changes pyasl.helcorr RV by the order of 1cm/s
    jd  = pyasl.asl.astroTimeLegacy.jdcnv(dt) + seconds_fractionalpart

    # Calculate helocentric velocity
    helcorr = pyasl.helcorr(longitude, latitude, altitude, ra, dec, jd, debug=False)

    # Apply doopler shift to the target spectra with helcorr correction velocity 
    nflux, wlprime = pyasl.dopplerShift(wavelength, flux, helcorr[0], edgeHandling=None, fillValue=None)

    print(" RV size of heliocenter correction for spectra", helcorr[0])
    return nflux, wlprime


# In[ ]:

target_nflux_tell, __ = barycorr_CRIRES(tellcorr_wl, tellcorr_I, tellcorr_hdr)

ref_nfluxtell, __ = barycorr_CRIRES(reftellcorr_wl, reftellcorr_I, reftellcorr_hdr)


# # Before and After Heliocentric Correction

# In[ ]:

plt.plot(reftellcorr_wl, reftellcorr_I, label="Reference" )
plt.plot(tellcorr_wl, tellcorr_I, label="Target")

plt.title("Not BERV Corrected")
plt.xlabel("Wavelength(nm)")

bokeh.plotting.show(bokeh.mpl.to_bokeh())


# In[ ]:

plt.plot(reftellcorr_wl, ref_nflux, label="Reference" )
plt.plot(tellcorr_wl, target_nflux, label="Target")

plt.title("BERV Corrected")
plt.xlabel("Wavelength(nm)")

bokeh.plotting.show(bokeh.mpl.to_bokeh())


# In[ ]:


# wavelength reference untill spectral tools is fixed 
from scipy.interpolate import interp1d
def match_wl(wl, spec, ref_wl, method="scipy", kind="linear", bounds_error=False):
    """Interpolate Wavelengths of spectra to common WL
    Most likely convert telluric to observed spectra wl after wl mapping performed"""
    starttime = time.time()
    if method == "scipy":
        print(kind + " scipy interpolation")
        linear_interp = interp1d(wl, spec, kind=kind, bounds_error=False)
        new_spec = linear_interp(ref_wl)
    elif method == "numpy":
        if kind.lower() is not "linear":
            print("Warning: Cannot do " + kind + " interpolation with numpy, switching to linear" )
        print("Linear numpy interpolation")
        new_spec = np.interp(ref_wl, wl, spec)  # 1-d peicewise linear interpolat
    else:
        print("Method was given as " + method)
        raise("Not correct interpolation method specified")
    print("Interpolation Time = " + str(time.time() - starttime) + " seconds")

    return new_spec  # test inperpolations 


# # Subtraction !

# In[ ]:

# Shift to the reference wavelength scale for subtraction
matched_tellcorr_I = match_wl(tellcorr_wl, target_nflux, reftellcorr_wl)

subtracted_I = reftellcorr_I - matched_tellcorr_I    # O/C I think


plt.plot(reftellcorr_wl, subtracted_I)
plt.hlines(0, np.min(reftellcorr_wl), np.max(reftellcorr_wl), colors='k', linestyles='dashed', label='')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Recovered Difference")
plt.title("Difference between {0} and {1}".format(target, reference_target))
bokeh.plotting.show(bokeh.mpl.to_bokeh())


# In[ ]:




# In[ ]:

# Combine all 3 together
from bokeh.models import Range1d
# Following example from http://bokeh.pydata.org/en/latest/docs/user_guide/quickstart.html

s1 = figure(width=750, height=300, title=None)
s1.line([2116, 2122], [1,1], color="grey", line_dash="dashed", line_width=1)
s1.line(tell_wl, tell_I, legend="TAPAS", color="black", line_width=2)
s1.line(gas_wl, gas_I, legend="ESO", color="blue", line_dash="dotdash", line_width=2)
s1.line(dracs_wl, dracs_I, legend="DRACS", color="red", line_dash="dashed",line_width=2)

#plt.plot(gas_wl, gas_I, label="Gasgano")
#plt.plot(dracs_wl, dracs_I, label="Dracs")
#plt.plot(tell_wl, tell_I, label="Tapas")
s1.title = "HD30501 Spectra"
s1.xaxis.axis_label = 'Wavelength (nm)'
s1.yaxis.axis_label = 'Nomalized Flux'
s1.legend.location = "bottom_right"
s1.title_text_font_size = "14pt"
s1.xaxis.axis_label_text_font_size = "12pt"
s1.yaxis.axis_label_text_font_size = "12pt"
s1.set(x_range=Range1d(2116, 2122), y_range=Range1d(0.68, 1.04))  #Edit wl range

# NEW: Tapas normal and H20 Scaling
s2 = figure(width=750, height=300, x_range=s1.x_range, y_range=s1.y_range, title=None)
s2.line([2116, 2122], [1,1], color="grey", line_dash="dashed", line_width=1)
s2.line(tellcorr_wl, tellcorr_I, legend="Airmass Scaling", color="blue", line_width=2)
s2.line(h20tellcorr_wl, h20tellcorr_I, legend="H20 Scaling", color="red", line_dash="dashed", line_width=2)

#plt.plot(tellcorr_wl,tellcorr_I, label= "Airmas Scaling")
#plt.plot(h20tellcorr_wl,h20tellcorr_I, label="H20 Scaling")

s2.title = "Telluric Correction"
s2.title_text_font_size = "14pt"
s2.xaxis.axis_label = 'Wavelength (nm)'
s2.xaxis.axis_label_text_font_size = "12pt"
s2.yaxis.axis_label = 'Nomalized Flux'
s2.yaxis.axis_label_text_font_size = "12pt"
s2.legend.location = "bottom_right"
#plt.xlabel("Wavelength(nm)")

# NEW: create a new plot and share only one range
s3 = figure(width=750, height=300, x_range=s1.x_range, title=None)
s3.line([2116, 2122], [0,0], color="grey", line_dash="dashed", line_width=1)
s3.line(reftellcorr_wl, subtracted_I, color="black", line_width=2)
bokeh_telluric_mask(s3, reftellcorr_wl, ref_nfluxtell, mask_limit=0.95, fill_alpha=0.4, fill_color='green')
bokeh_telluric_mask(s3, reftellcorr_wl, reftellcorr_I, mask_limit=0.95, fill_alpha=0.4, fill_color='yellow')

s3.title = "Subtraction of {} from {}".format(reference_target, target)
s3.title_text_font_size = "14pt"
s3.xaxis.axis_label = 'Wavelength (nm)'
s3.xaxis.axis_label_text_font_size = "12pt"
s3.yaxis.axis_label = 'Flux Difference'
s3.yaxis.axis_label_text_font_size = "12pt"
s3.legend.location = "bottom_right"
#p = gridplot([[s1],[s2],[s3]], toolbar_location=None)

# show the results
#show(p)

show(vplot(s1, s2, s3))



# coding: utf-8

# # Azores poster plots
# 

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import Obtain_Telluric as obt
from Get_filenames import get_filenames
get_ipython().magic(u'matplotlib inline')


# In[ ]:

# Import Bokeh modules for interactive plotting
import bokeh.io
import bokeh.mpl
import bokeh.plotting
get_ipython().magic(u"config InlineBackend.figure_formats = {'svg',}")
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
dracs_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/{0}/Combined_Nods/".format(target)
#dracs_path = "C:/Users/Jason/Documents/PhD/Phd-codes/Notebooks/HD30501_data/{0}/".format(obs_num)
#dracs_path = "../HD30501_data/{0}".format(obs_num)
dracs_name = get_filenames(dracs_path, "CRIRE.*","*{0}.nod.ms.norm.sum.wavecal.fits".format(chip_num))

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
#dracs_path = "C:/Users/Jason/Documents/PhD/Phd-codes/Notebooks/HD30501_data/{0}/".format(obs_num)

tellcorr_name = get_filenames(dracs_path, "CRIRE.*","*{0}.nod.ms.norm.sum.wavecal.tellcorr.fits".format(chip_num))
h20tellcorr_name = get_filenames(dracs_path, "CRIRE.*","*{0}.nod.ms.norm.sum.wavecal.h2otellcorr.fits".format(chip_num))
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
reference_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/{0}/Combined_Nods/".format(reference_target)
#reference_path = "C:/Users/Jason/Documents/PhD/Phd-codes/Notebooks/HD30501_data/{0}/".format(ref_num)
reftellcorr_name = get_filenames(reference_path, "CRIRE.*","*{0}.nod.ms.norm.sum.wavecal.tellcorr.fits".format(chip_num))
refh20tellcorr_name = get_filenames(reference_path, "CRIRE.*","*{0}.nod.ms.norm.sum.wavecal.h2otellcorr.fits".format(chip_num))

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

def barycorr_CRIRES(wavelength, flux, header, extra_offset=None):
    #"""
    #Calculate Heliocenteric correction values and apply to spectrum.
   
    #SHOULD test again with bary and see what the  difference is.
    #"""

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
    
    if extra_offset:
        print("Warning!!!! have included a manual offset for testing")
        helcorr_val = helcorr[0] + extra_offset
    else:
        helcorr_val = helcorr[0]
    # Apply doopler shift to the target spectra with helcorr correction velocity 
    nflux, wlprime = pyasl.dopplerShift(wavelength, flux, helcorr_val, edgeHandling=None, fillValue=None)

    print(" RV s}ize of heliocenter correction for spectra", helcorr_val)
    return nflux, wlprime


# In[ ]:

manual_ofset_for_testing = 0

target_nflux, target_wlprime = barycorr_CRIRES(tellcorr_wl, tellcorr_I, tellcorr_hdr, extra_offset=manual_ofset_for_testing)

ref_nflux, ref_wlprime = barycorr_CRIRES(reftellcorr_wl, reftellcorr_I, reftellcorr_hdr)

# telluric line shift for masking
target_nflux_tell, __ = barycorr_CRIRES(tellcorr_wl, tellcorr_tell, tellcorr_hdr)
ref_nfluxtell, __ = barycorr_CRIRES(reftellcorr_wl, reftellcorr_tell, reftellcorr_hdr)


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
### Old values
matched_tellcorr_I = match_wl(tellcorr_wl, target_nflux, reftellcorr_wl)

#subtracted_I = reftellcorr_I - matched_tellcorr_I    # O/C I think     ##### THIS was a BUG!!!

## BARY Corrected values

#target_nflux, target_wlprime = barycorr_CRIRES(tellcorr_wl, tellcorr_I, tellcorr_hdr, extra_offset=manual_ofset_for_testing)
#ref_nflux, ref_wlprime = barycorr_CRIRES(reftellcorr_wl, reftellcorr_I, reftellcorr_hdr)
#correct_match_I = match_wl(tellcorr_wl, target_nflux, reftellcorr_wl)

subtracted_I = ref_nflux - matched_tellcorr_I    ##### This fixed the bug and removed stellar lines very well!!!!


plt.plot(reftellcorr_wl, subtracted_I)
plt.hlines(0, np.min(reftellcorr_wl), np.max(reftellcorr_wl), colors='k', linestyles='dashed', label='')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Recovered Difference")
plt.title("Difference between {0} and {1}".format(target, reference_target))
bokeh.plotting.show(bokeh.mpl.to_bokeh())


# In[ ]:

# Include masking
from bokeh.plotting import figure, show, output_file, gridplot, vplot
from bokeh.models import BoxAnnotation

def bokeh_telluric_mask(fig, wl, I, mask_limit=0.9, fill_alpha=0.2, fill_color='red'):
    """ For use with bokeh"""
    wl_mask = I < mask_limit
    mean_step = np.mean([wl[1]-wl[0], wl[-1]-wl[-2]])   # Average nominal step size
    starts, ends = mask_edges(wl[wl_mask], mean_step)
    Boxes = [BoxAnnotation(plot=fig, left=start, right= end, fill_alpha=fill_alpha, fill_color=fill_color) for start, end in zip(starts, ends)]
    fig.renderers.extend(Boxes)
    
def matplotlib_telluric_mask(wl, I, mask_limit=0.9):
    """For use with matplotlib"""
    wl_mask = I < mask_limit
    mean_step = np.mean([wl[1]-wl[0], wl[-1]-wl[-2]])   # Average nominal step size
    starts, ends = mask_edges(wl[wl_mask], mean_step)
    [plt.axvspan(start, end, facecolor='g', alpha=0.5) for start, end in zip(starts, ends)] 
    
def mask_edges(wl, mean_step):
    beginings = [wav2 for wav1, wav2 in zip(wl[:-1], wl[1:]) if wav2-wav1 > 3*np.abs(mean_step)]
    ends = [wav1 for wav1, wav2 in zip(wl[:-1], wl[1:]) if wav2-wav1 > 3*np.abs(mean_step)]
    
    # prepend start of first line, and append end of last line
    beginings = [wl[0]] + beginings   # prepend starting value
    ends = ends + [wl[-1]] # append end value
    
    return beginings, ends

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p = figure(tools=TOOLS)

p.line(reftellcorr_wl, subtracted_I)
#plt.hlines(0, np.min(reftellcorr_wl), np.max(reftellcorr_wl), colors='k', linestyles='dashed', label='')

bokeh_telluric_mask(p, reftellcorr_wl, ref_nfluxtell, mask_limit=0.95, fill_alpha=0.4, fill_color='green')
bokeh_telluric_mask(p, reftellcorr_wl, reftellcorr_I, mask_limit=0.95, fill_alpha=0.4, fill_color='yellow')
p.title = "Comparison with Masks"
p.xaxis.axis_label = 'Wavelength'
p.yaxis.axis_label = 'Signal'
show(p)


# In[ ]:

# Combine all 3 together
from bokeh.models import Range1d
# Following example from http://bokeh.pydata.org/en/latest/docs/user_guide/quickstart.html
fig_height = 250
fig_width = 800

s1 = figure(width=fig_width, height=fig_height, title="HD30501 Spectrum with telluric line model")
s1.line([np.min(tellcorr_wl), np.max(tellcorr_wl)], [1,1], color="black", line_dash="dashed", line_width=1)
s1.line(tell_wl, tell_I, legend="TAPAS", color="blue", line_width=2)
bokeh_telluric_mask(s1, tellcorr_wl, tellcorr_tell, mask_limit=0.95, fill_alpha=0.4, fill_color='green')
bokeh_telluric_mask(s1, tellcorr_wl, tellcorr_I, mask_limit=0.95, fill_alpha=0.4, fill_color='yellow')
#s1.line(gas_wl, gas_I, legend="ESO", color="blue", line_dash="dotdash", line_width=2)
s1.line(dracs_wl, dracs_I, legend="HD30501", color="red", line_dash="dashed",line_width=2)

#plt.plot(gas_wl, gas_I, label="Gasgano")
#plt.plot(dracs_wl, dracs_I, label="Dracs")
#plt.plot(tell_wl, tell_I, label="Tapas")
#s1.title = "HD30501 Spectrum"
s1.xaxis.axis_label = 'Wavelength (nm)'
s1.yaxis.axis_label = 'Nomalized Intensity'
s1.legend.location = "bottom_right"
s1.title_text_font_size = "12pt"
s1.xaxis.axis_label_text_font_size = "12pt"
s1.yaxis.axis_label_text_font_size = "12pt"
s1.legend.border_line_color = None
s1.set(x_range=Range1d(2111.8, 2123.6), y_range=Range1d(0.8, 1.03))  #Edit wl range

# NEW: Tapas normal and H20 Scaling
s2 = figure(width=fig_width, height=fig_height, x_range=s1.x_range, y_range=s1.y_range, 
            title="Telluric correction through division of the telluric line model")
s2.line([np.min(tellcorr_wl), np.max(tellcorr_wl)], [1,1], color="black", line_dash="dashed", line_width=1)
#s2.line(tellcorr_wl, tellcorr_I, legend="Airmass Scaling", color="blue", line_width=2)
#s2.line(h20tellcorr_wl, h20tellcorr_I, legend="H20 Scaling", color="red", line_dash="dashed", line_width=2)
s2.line(h20tellcorr_wl, h20tellcorr_I, legend="H20 Scaling", color="blue", line_dash="solid", line_width=2)
 
bokeh_telluric_mask(s2, tellcorr_wl, tellcorr_tell, mask_limit=0.95, fill_alpha=0.4, fill_color='green')
bokeh_telluric_mask(s2, tellcorr_wl, tellcorr_I, mask_limit=0.95, fill_alpha=0.4, fill_color='yellow')

#s2.title = "Telluric correction by division of telluric line model"
s2.title_text_font_size = "12pt"
s2.xaxis.axis_label = 'Wavelength (nm)'
s2.xaxis.axis_label_text_font_size = "12pt"
s2.yaxis.axis_label = 'Normalized Intensity'
s2.yaxis.axis_label_text_font_size = "12pt"
s2.legend.location = None
#s2.legend.location = "bottom_right"
#s2.legend.border_line_color = None
#plt.xlabel("Wavelength(nm)")

# NEW: create a new plot and share only one range
s3 = figure(width=fig_width, height=fig_height, x_range=s1.x_range, title=None)
s3.line([np.min(reftellcorr_wl), np.max(reftellcorr_wl)], [0,0], color="black", line_dash="dashed", line_width=1)
s3.line(reftellcorr_wl, subtracted_I, color="blue", line_width=2)
bokeh_telluric_mask(s3, reftellcorr_wl, ref_nfluxtell, mask_limit=0.95, fill_alpha=0.4, fill_color='green')
bokeh_telluric_mask(s3, reftellcorr_wl, reftellcorr_I, mask_limit=0.95, fill_alpha=0.4, fill_color='yellow')

s3.title = "Subtraction of two barycentic RV corrected observations"
s3.title_text_font_size = "12pt"
s3.xaxis.axis_label = 'Wavelength (nm)'
s3.xaxis.axis_label_text_font_size = "12pt"
s3.yaxis.axis_label = 'Difference'
s3.yaxis.axis_label_text_font_size = "12pt"
s3.legend.location = "bottom_right"
s3.legend.border_line_color = None
#p = gridplot([[s1],[s2],[s3]], toolbar_location=None)

# show the results
#show(p)

show(vplot(s1, s2, s3))


# In[ ]:




# In[ ]:




# # Minimize Subtraction Residual to remove stellar line
# 
# ## This is unneed at present as I found a bug in my code so I was not doing the subtration with the berv corrected reference I. It is fixed now!!! 11/7/16

# In[ ]:

from lmfit import minimize, Parameters
import lmfit
manual_ofset_for_testing = -8.5


### Fit using lmfit
def wav_selector(wav, flux, wav_min, wav_max, verbose=False):
    """ Faster Wavelenght selector
    
    If passed lists it will return lists.
    If passed np arrays it will return arrays
    
    Fastest is using np.ndarrays
    fast_wav_selector ~1000-2000 * quicker than wav_selector
    """
    if isinstance(wav, list): # if passed lists
          wav_sel = [wav_val for wav_val in wav if (wav_min < wav_val < wav_max)]
          flux_sel = [flux_val for wav_val, flux_val in zip(wav,flux) if (wav_min < wav_val < wav_max)]
    elif isinstance(wav, np.ndarray):
        # Super Fast masking with numpy
        mask = (wav > wav_min) & (wav < wav_max)
        if verbose:
            print("mask=", mask)
            print("len(mask)", len(mask))
            print("wav", wav)
            print("flux", flux)
        wav_sel = wav[mask]
        flux_sel = flux[mask]
    else:
          raise TypeError("Unsupported input wav type")
    return [wav_sel, flux_sel]

#from SpectralTools import wav_selector

def stellar_line_residuals(params, target_data, reference_data):
    # Parameters 
    rv_offset = params["rv_offset"].value
    wl_min = params["wl_min"].value
    wl_max = params["wl_max"].value
    
    # Data
    target_wl = target_data[0]
    target_I = target_data[1]
    
    reference_wl = reference_data[0]
    reference_I = reference_data[1]
    
    # dopler shift target spectrum
    nflux, wlprime = pyasl.dopplerShift(target_wl, target_I, rv_offset, edgeHandling=None, fillValue=None)
    
  
    
    matched_wl_reference_I = match_wl(reference_wl, reference_I, target_wl)
    
    subtracted_I = nflux - matched_wl_reference_I
    
    selected_section = wav_selector(target_wl, subtracted_I, wl_min, wl_max)
 
    # calulate aproximate area of region 
    area = np.sum(np.abs(subtracted_I[:-1] * (target_wl[1:] - target_wl[:-1])))
    
    return area
print("Done")


#tell_data4 = fast_wav_selector(tapas_h20_data[0], tapas_h20_data[1], 0.9995*np.min(wl4), 1.0005*np.max(wl4))


# In[ ]:

# Set up parameters 
params = Parameters()
params.add("rv_offset", value=-0)   # add min and max values ?
params.add('wl_min', value=2116.6, vary=False)   #  hack valuses for first run. get from mask later
params.add('wl_max', value=2117.4, vary=False)


# In[ ]:

out = minimize(stellar_line_residuals, params, args=([tellcorr_wl, target_nflux], [reftellcorr_wl, ref_nflux]))
outreport = lmfit.fit_report(out)
print(outreport)


# In[ ]:




# In[ ]:





#

""" Codes for Telluric contamination removal 
    Interpolates telluric spectra to the observed spectra.
    Divides spectra telluric spectra
    can plot result

"""
import numpy as np
import scipy as sp

def divide_spectra(spec_a, spec_b):
""" Assumes that the spectra have been interpolated to same wavelength step"""
""" Divide two spectra"""
    assert(len(spec_a) == len(spec_b)), "Not the same length"
    divide = spec_a / spec_b
    return divide

def match_wl(wl, spec, ref_wl):
    """Interpolate Wavelengths of spectra to common WL
    Most likely convert telluric to observed spectra wl after wl mapping performed"""
    newspec1 = np.interp(ref_wl, wl, spec)  # 1-d peicewise linear interpolat
    # cubic spline with scipy   
    newspec2 = sp.interpolate.interp1d(wl, spec, kind='cubic')(ref_wl)
    return newspec1, newspec2  # test inperpolations 

def plot_spectra(wl, spec, colspec="k.-", label=None, title="Spectrum"):
    """ Do I need to replicate plotting code?
     Same axis
    """
    plt.plot(wl, spec, colspec, label=label)
    plt.title(title)
    plt.legend()
    plt.show(break=False)
    
def test_plot_interpolation(x1, y1, x2, y2, methodname=None):
   """ Plotting code """
    plt.plot(x1, y1, label="original values")
    plt.plot(x2, y2, label="new points")
    plt.title("testing Interpolation: ", methodname)
    plt.label()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Norm Intensity")
    plt.show(block=False)


def TelluricCorrect(wl_obs, spec_obs,wl_tell, spec_tell):
    """Code to contain other functions in this file

 1. Interpolate spectra to same wavelengths with match_WLs()
 2. Divide by Telluric
3.   ...
 """
   
    interp1, interp2 = match_wls(wl_tell, spec_tell, wl_obs)
    # test outputs
    test_plot_interpolation(wl_tell, spec_tell, wl_obs, interp1)
    test_plot_interpolation(wl_tell, spec_tell, wl_obs, interp2)

    # division
    corrected_spec = divide_spectra(spec_obs, interp2)
     # 
    # other corrections?
    
    
    return WL, corrected_spec

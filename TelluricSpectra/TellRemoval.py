
""" Codes for Telluric contamination removal 
    Interpolates telluric spectra to the observed spectra.
    Divides spectra telluric spectra
    can plot result

"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import interpolate

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
    newspec2 = interpolate.interp1d(wl, spec, kind='cubic')(ref_wl)   
    #newspec2 = sp.interpolate.interp1d(wl, spec, kind='cubic')(ref_wl)
    return newspec1, newspec2  # test inperpolations 

def plot_spectra(wl, spec, colspec="k.-", label=None, title="Spectrum"):
    """ Do I need to replicate plotting code?
     Same axis
    """
    plt.plot(wl, spec, colspec, label=label)
    plt.title(title)
    plt.legend()
    plt.show(block=False)
    
def test_plot_interpolation(x1, y1, x2, y2, methodname=None):
    """ Plotting code """
    plt.plot(x1, y1, label="original values")
    plt.plot(x2, y2, label="new points")
    plt.title("testing Interpolation: ", methodname)
    plt.label()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Norm Intensity")
    plt.show(block=False)


def telluric_correct(wl_obs, spec_obs, wl_tell, spec_tell):
    """Code to contain other functions in this file

     1. Interpolate spectra to same wavelengths with match_WLs()
     2. Divide by Telluric
     3.   ...
    """
   
    interp1, interp2 = match_wl(wl_tell, spec_tell, wl_obs)
    # could just do interp here without  match_wl function 
    # test outputs

#    test_plot_interpolation(wl_tell, spec_tell, wl_obs, interp1)
 #   test_plot_interpolation(wl_tell, spec_tell, wl_obs, interp2)

    # division
    corrected_spec = divide_spectra(spec_obs, interp2)
    # 
    # other corrections?
    
    
    return corrected_spec



if __name__ == "__main__":
    """ Some test code for testing functions """
    sze = 20
    x2 = range(sze)
    y2 = np.random.randn(len(x2)) + np.ones_like(x2)
    y2 = 0.5 * np.ones_like(x2)
    x1 = np.linspace(1, sze-1.5, 9)
    y1 = np.random.randn(len(x1)) + np.ones_like(x1)
    y1 = np.ones_like(x1)
    print(x1)
    print(x2)
    #print(y1)
    #print(y2)
    y1_cor = telluric_correct(x1, y1, x2, y2)
    print(x1)
    print(y1)
    print(y1_cor)  
    
    
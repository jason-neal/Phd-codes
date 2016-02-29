#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""Fitting the Centroid positions of telluric lines with gausians 
to obtain a wavelenght calibration from pixel to wavelength.

"""

## GaussainFitting.py
from __future__ import division
#from astropy.io import fits
#import Obtain_Telluric

import copy
import math
import numpy as np
#import scipy as sp 
#from Get_filenames import get_filenames
import matplotlib.pyplot as plt
import scipy.optimize as opt

import IOmodule
from Gaussian_fit_testing import Get_DRACS

## Gaussian Fitting Module
## Develop the advanced fitting routine that fits slices of spectra 
## with potentailly multiple gausians.

#######################################################################
#                                                                     #
#                        # Plot interaction                           #
#                                                                     #
#                                                                     #
#######################################################################
# def onclick(event):    # dont need know as can use fig.ginput()
#     global ix, iy, coords, fig, cid
#     # Disconnect after right click
#     if event.button == 3:   # Right mouse click
#         fig.canvas.mpl_disconnect(cid)
#         plt.close(1)
#         return
#     ix, iy = event.xdata, event.ydata
#     coords.append((ix, iy))
    # print("Click position", [ix, iy])
#     return

def get_rough_peaks(wl_a, spec_a, wl_b, spec_b):
    """ Get rough coordinate values to use advanced fitting on
    First run through of peaks in spectra

    """
    textloc_a = (np.median(wl_a), max([min(spec_a), 0.7]))
    text_a = "Select Spectral regions/lines for finer calibration fitting"
    #textloc_b = (np.median(wl_b), max([min(spec_b), 0.5]))
    text_b = "Select matching Telluric regions/lines for finer calibration fitting"
    a_coords = get_coords(wl_a, spec_a, wl_b, spec_b, title="Observed Spectra",
                                textloc=textloc_a, text=text_a) 
    b_coords = get_coords(wl_b, spec_b, wl_a, spec_a, title="Telluric Lines",
                               points_b=a_coords, textloc=textloc_a, text=text_b)
    return a_coords, b_coords, 

def get_coords(wl_a, spec_a, wl_b, spec_b, title="Mark Lines on Spectra", 
                    points_a=None, points_b=None, textloc=False, text=False, 
                    model=False):
    """ Obtains Coordinates of clicked points on the plot of spectra.
     The blue plot is the plot to click on to get peak coordinates
     the black plot is the other spectra to compare against.
     points show the peaks that were choosen to fit.

    """    
    while True:
        #global coords, fig  #, cid
        coords = []
        #fig, ax1, ax2 = plot_both_fits(wl_a, spec_a, wl_b, spec_b,title=title)
        #ax1.set_xlabel("Wavelength/pixels")
        #ax2.legend()
        #pfig.show()
        #testcoords = fig.ginput(n=0, timeout=0, show_clicks=True, 
        #    mouse_add=1, mouse_pop=2, mouse_stop=3) # better way to do it
        #print("Test coords with function", testcoords)
        fig = plt.figure(figsize=(15, 15))
        fig.suptitle(title)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) # remove edge space
        #fig.set_size_inches(25, 15, forward=False)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.plot(wl_b, spec_b, "k-", lw=2, label="Ref Spec")
        ax2.plot(wl_b, spec_b, "k-", lw=2, label="Ref Spec")
        ax1.set_xlabel("spec_b")
        ax1.set_xlim(np.min(wl_b), np.max(wl_b))
        ax2.plot(wl_a, spec_a, "b", lw=2, label="Spectra to Click")
        ax2.plot(wl_a, np.ones_like(spec_a), "b-.")
        ax2.set_ylabel("Normalized Flux/Intensity")
        # Stopping scientific notation offset in wavelength
        ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        ax2.get_xaxis().get_major_formatter().set_useOffset(False)
      
        # Setting ylimits
        amin = np.min(spec_a)
        bmin = np.min(spec_b)
        amax = np.max(spec_a)
        bmax = np.max(spec_b)
        if amin < 0 or bmin < 0:  #set limits only when large values occur 
            if amax > 1.2 or bmax > 1.2:
                ax1.set_ylim(0, 1.2)
                #ax2.set_ylim(0, 1.2)
            else:
                ax1.set_ylim(0, np.max([amax, bmax]) + 0.02)

        if model:
            ax1.plot(model[0],model[1], 'r', label="model spectrum")
            ax2.plot(model[0],model[1], 'r', label="model spectrum")
        
        #ymax = np.min([amax, bmax, 1.05]) + 0.02 * np.min([amax-amin,bmax-bmin])
        #ymin = np.max([amin, bmin, 0.5]) - 0.05 
        #ax1.set_ylim(ymin, ymax)
        #ax2.set_ylim(ymin, ymax)
        #ax2.set_xlabel("Wavelength/pixels")
        ax2.set_xlim(np.min(wl_a), np.max(wl_a))
        ax2.legend()        ### ISSUES with legend
        #print("get_coords points_A input ", points_a)
        #print("get_coords points_B input ", points_b)
        if points_a is not None:
            #print("points_a", points_a)
            #xpoints = points_a[0::3]
            #ypoints = np.ones_like(points_a[1::3]) - points_a[1::3]
            xpoints = []
            ypoints = []
            for coord in points_a:
                xpoints.append(coord[0])
                ypoints.append(coord[1])
            ax2.plot(xpoints, ypoints, "g<", label="Selected A points", ms=13)    
        if points_b is not None:
            #xpoints = points_b[0::3]
            #ypoints = np.ones_like(points_b[1::3]) - points_b[1::3]
            xpoints = []
            ypoints = []
            for coord in points_b:
                xpoints.append(coord[0])
                ypoints.append(coord[1])
            ax1.plot(xpoints, ypoints, "md", label="Selected B points", ms=13)    
        
        if textloc and text:
            """ Display text on the plot"""
            print("text location", textloc)
            ax1.text(textloc[0], textloc[1]-0.005, text, fontsize=20, color='red',
                fontweight='bold', horizontalalignment='center')
            ax2.text(textloc[0], textloc[1]-0.005, text, fontsize=20, color='red',
                fontweight='bold', horizontalalignment='center')

        coords = fig.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1,
                            mouse_pop=2, mouse_stop=3) # better way to do it
        plt.close()
        #cid = fig.canvas.mpl_connect('button_press_event', onclick)      
        #plt.show()
        #ans = input("Are these the corrdinate values you want to use to fit these peaks? y/N?")
        #if ans.lower() == "y":   # Good coordinates to use
        return coords

#######################################################################
#                                                                     #
#                        # Fitting Code                               #
#                                                                     #
#                                                                     #
#######################################################################
def do_fit(wl, spec, init_params, stel=None, tell=None):
    """ Perfrom fit using opt.curvefit and return the found parameters. 
    if there is stellar lines then stel will not be None.

    """
    #print("Params before fit", init_params, type(init_params))
    if stel is not None:
        #use lambda instead here
        #__ is junk parameter to take the covar returned by curve_fit
        print("init params", init_params, "stellar params", stel)
        assert len(stel)%3 is 0, "stel parameters not multiple of 3"
        stelnum = int(len(stel)/3)  # number of stellar lines
        print("number of stellar lines ", stelnum)
        init_params = np.concatenate((init_params, stel), axis=0)
        print("appended array with stellar lines", init_params)
        params, __ = opt.curve_fit(lambda x, *params: func_with_stellar(x,
                                    stelnum, params), wl, spec, init_params)
    elif tell is not None:
        """ Don't need all this as telluric lines are just added"""
        print("init params", init_params, "telluric params", tell)
        assert len(tell)%3 is 0, "Telluric parameters not multiple of 3"
        tellnum = int(len(tell)/3)  # number of stellar lines
        print("number of telluric lines ", tellnum)
        init_params = np.concatenate((init_params, tell), axis=0)
        print("appended array with telluric lines", init_params)
        #params, __ = opt.curve_fit(lambda x, *params: func_with_telluric(x, 
        #                            tellnum, params), wl, spec, init_params)
        params, __ = opt.curve_fit(func, wl, spec, init_params)
        print("returned tell params", params)
    else:    
        params, __ = opt.curve_fit(func, wl, spec, init_params)
        
    #print("Params after fit", params, "type output", type(params))
    assert len(params) is len(init_params), "len(Params) do not match"     
    return params

def adv_wavelength_fitting(wl_a, spec_a, AxCoords, wl_b, spec_b, BxCoords, model=False):
    """ Returns the positions of matching peaks for calibration map

    """
    best_a_coords = []
    best_a_peaks = []
    best_b_coords = []
    best_b_peaks = []

    wl_a = np.array(wl_a)      # make sure all are numpy arrays
    spec_a = np.array(spec_a)  # make sure all are numpy arrays
    wl_b = np.array(wl_b)      # make sure all are numpy arrays
    spec_b = np.array(spec_b)  # make sure all are numpy arrays
    delta_a = np.abs(np.mean(wl_a[1:] - wl_a[:-1]))   # average wl step
    delta_b = np.abs(np.mean(wl_b[1:] - wl_b[:-1]))

    assert len(AxCoords) is len(BxCoords), "Lenght of Coords do not match"
    for i in range(len(AxCoords)):
        print("A coords Axcoords", AxCoords)
        print(i," ith value in Axcoords", AxCoords[i])
        wl_a_sec, sect_a = slice_percentage(wl_a, spec_a, AxCoords[i])
        wl_b_sec, sect_b = slice_percentage(wl_b, spec_b, BxCoords[i])
        
       #renormalize by upperquartile to give beter fit chance 
       # print("upper quatrile A", upper_quartile(sect_a))
        a_copy = copy.copy(sect_a)
        b_copy = copy.copy(sect_b)
        auq = upper_quartile(a_copy)
        auq = upper_quartile(b_copy)
        #print ("upper quartile A", auq, type(auq))
        #print ("upper quartile B", auq, type(auq))
        sect_a = sect_a/auq
        sect_b = sect_b/auq
        #sect_a = sect_a/np.median(sect_a)
        #sect_b = sect_b/np.median(sect_b)
        
        while True:   # Was this a good fit
            try:  # RuntimeError of fitting
            # Get more accurate coordinates with zoomed in sections
                a_coords = get_coords(wl_a_sec, sect_a, wl_b_sec, sect_b,
                                           title="Select Spectra Lines",
                                           textloc= (np.median(wl_a_sec), max(min(sect_a), 0.5)),
                                           text="Choose lines for calibration", model=model)  #(x,y)'s
                b_coords = get_coords(wl_b_sec, sect_b, wl_a_sec, sect_a,
                                           title="Select Telluric Lines", 
                                           points_b=a_coords,
                                           textloc= (np.median(wl_b_sec), max(min(sect_b), 0.5)),
                                           text="Choose lines to calibrate with", model=model)
                print("Returned a_coords = ", a_coords)
                print("Returned b_coords = ", b_coords)
            
                # Turn Coords of peaks into init params for fit
                init_params_a = coords2gaussian_params(a_coords, delta_a)
                init_params_b = coords2gaussian_params(b_coords, delta_b)

            # Plot guessed coords
                #plot_fit(wl_a_sec, sect_a, a_coords, title="Line Guesses")
                #plot_fit(wl_b_sec, sect_b, b_coords, title="Telluric Guesses")

            # If spectral lines do a fit with spectral lines multiplied to telluric lines
                #print("Fitting a_coords", "!") 
                # show figure
                #fig, __, __ = plot_both_fits(wl_a_sec, sect_a, wl_b_sec, sect_b, show_plot=True, 
                #                             title="Any Stellar lines", hor=1) 
                #stel = inputer("Are there any Stellar lines to include in the fit y/N") 
                stel = "y"
                #plt.close(fig)
                if stel in ["y", "Yes", "YES" "yes"]:
                    # Select the stellar lines for the spectral fit
                    stellar_lines = get_coords(wl_a_sec, sect_a, wl_b_sec, sect_b, 
                                                    title="Select Spectral Lines", 
                                                    points_a=a_coords, 
                                                    points_b=b_coords, 
                                                    textloc=(np.median(wl_a_sec), np.max(np.min(sect_a), 0.5)), 
                                                    text="Select Stellar lines to multiply")
                    num_stellar = len(stellar_lines)
                    stellar_params = coords2gaussian_params(stellar_lines, delta_a)
                # perform the stellar line fitting version
                    fit_params_a = do_fit(wl_a_sec, sect_a, init_params_a, 
                                          stel=stellar_params)
                else: # Perform the normal fit
                    num_stellar = 0
                    fit_params_a = do_fit(wl_a_sec, sect_a, init_params_a)
                
                tell = "y"
                if tell in ["y", "Yes", "YES" "yes"]:
                    telluric_lines = get_coords(wl_b_sec, sect_b, wl_a_sec, sect_a, 
                                                    title="Select Spectral Lines", 
                                                    points_a=b_coords, 
                                                    points_b=a_coords, 
                                                    textloc=(np.median(wl_b_sec), np.max(np.min(sect_b), 0.5)), 
                                                    text="Select extra Telluric lines to add.")
                    num_telluric = len(telluric_lines)
                    telluric_params = coords2gaussian_params(telluric_lines, delta_b)
                    fit_params_b = do_fit(wl_b_sec, sect_b, init_params_b, 
                                          tell=telluric_params)
                else:  
                    num_telluric = 0
                    fit_params_b = do_fit(wl_b_sec, sect_b, init_params_b)
                
                fit_worked = True    
            except RuntimeError:
                print("Runtime Error: Fit could not find good parameters" )
                cont_try = inputer('Would you like to keep trying to fit to this section Y/n?')
                if cont_try in ['n', 'no','N','No','NO']:
                    fit_worked = False
                    break
                else:    # Else try fit points again.
                    fit_worked = True
                    continue # start next iteration of loop selecting points.
            
            
            print ("Using plot_both_fits ", " to plot fit results")
            fig, __, __ = overlay_fitting(wl_a_sec,  sect_a, wl_b_sec, sect_b, show_plot=True, paramsA=fit_params_a,
            paramsB=fit_params_b, hor=1)
  
            #fig, __, __ = plot_both_fits(wl_a_sec, sect_a, wl_b_sec, sect_b, paramsA=fit_params_a,
            #                             paramsB=fit_params_b, init_params_a=None, init_params_b=None, 
            #                             title="Displaying Fits with Originals", show_plot=True, hor=1) # need to show without stoping  
            
            goodfit = inputer(" Was this a good fit? y/N/s (skip)?")
            if goodfit in ["yes", "y", "Y", "Yes", "YES"]:
                plt.close(fig)
                break   # exit while true loop
            elif goodfit in ["skip", "s"]:
                plt.close(fig)
                pass   # skip this fit and try another 
            else:
                print("Registered this as not a good fit. Trying again.") 
                plt.close(fig)
                continue

        if fit_worked:
            # Seperate back out the stellar/telluric lines
            if num_stellar is not 0:
                fit_line_params_a, fit_stell_params = split_telluric_stellar(fit_params_a, num_stellar)
            else:
                fit_line_params_a = fit_params_a
                fit_stell_params = []

            if num_telluric is not 0:
                fit_line_params_b, fit_tell_params = split_telluric_stellar(fit_params_b, num_telluric)
            else:
                fit_line_params_b = fit_params_b
                fit_stell_params = []

            fitted_coords_a = params2coords(fit_line_params_a)      # Spectra
            fitted_coords_b = params2coords(fit_line_params_b)      # Tellruic spectrum
            #Add a large Marker to each peak and then label by number underneath

            fig, __, __ = plot_both_fits(wl_a_sec, sect_a, wl_b_sec, sect_b, show_plot=True,
                title="Pick Results to use", fitcoords_a=fitted_coords_a,
                best_a=best_a_coords, fitcoords_b=fitted_coords_b,
                best_b=best_b_coords, hor=1)
                   
            for i in range(0,len(fitted_coords_a)):
                coord_a = fitted_coords_a[i]
                coord_b = fitted_coords_b[i]
                # include = input("Use Peak #" + str(i+1) +" corresponding to" + 
                #                    str([coord_a[0],'pxls', coord_b[0],'nm']) + " y/N?")
                include = input("Use Peak # {} ".format(i+1) + "corresponding to" +
                             " [a-{0:.2f}, b-{1:.2f}]? y/N?".format(coord_a[0], coord_b[0]))
                if include.lower() == "y" or include.lower() == "yes":
                    best_a_coords.append(coord_a[0])
                    best_b_coords.append(coord_b[0])
                    best_a_peaks.append(coord_a[1])
                    best_b_peaks.append(coord_b[1])

            # ask do you want to include all lines? if yes BestCoordsA.append(), BestCoordsB.append()
            # no - individually ask if want want each line included and append if yes
            # probably plot each individually to identify 
            #include = input(" Use Coordinate point" + str([CoordsA[i], CoordsB[i]]) + " y/N?")
            #if include.lower() == "y" or include.lower() == "yes":
             #   best_a_coords.append(CoordsA[i]) # tempry filler for return
            #    best_b_coords.append(CoordsB[i]) # tempry filler to return same as inputs
            print("best_a_coords", best_a_coords)
            print("best_b_coords", best_b_coords)
            plt.close(fig)
    return best_a_coords, best_a_peaks, best_b_coords, best_b_peaks


#######################################################################
#                                                                     #
#                        # Fitting Models                             #
#                                                                     #
#                                                                     #
#######################################################################

def func(x, *params):
    """ Function to generate multiple gaussian profiles. 
    Adapted from 
    http://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python. 
    Params are now list (or numpy array) of values in order of xpos, 
    ypos, sigma of each gausian peak

    """
    y = np.ones_like(x)
    #print("*params inside function", type(params),len(params), params)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = np.abs(params[i+1]) # always positive so peaks always down
        wid = params[i+2]
        y = y - amp * np.exp(-0.5 * ((x - ctr)/wid)**2)
    y[y < 0] = 0
    return y

def func_with_stellar(x, num_stell, *params):
    """ Function to generate the multiple gaussian profiles with 
    stellar gausian. Adapted from 
    http://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python 
    For the stellar line case the frist param contains then number of
    stellar lines are present. The stellar lines are at the end of the
     list of parameters so need to seperate them out. 
    [[number of telluric lines], telluric lines, stellar lines] 
    # not any more use lambda fucntion as a fixed parameter

    """
    y_line = np.ones_like(x)
    y_stel = np.ones_like(x)
    #num_stel = params[0]                     # number of stellar lines given
    par = params[0]
    line_params, stellar_params = split_telluric_stellar(par, num_stell)

    for i in range(0, len(line_params), 3):
        ctr = line_params[i]
        amp = abs(line_params[i+1])   # always positive so peaks are downward
        wid = line_params[i+2]
        y_line = y_line - amp*np.exp(-0.5*((x-ctr)/wid)**2)  #Add teluric lines
    for i in range(0, len(stellar_params), 3):
        stel_ctr = stellar_params[i]
        stel_amp = abs(stellar_params[i+1])  #always positive so peaks are down
        stel_wid = stellar_params[i+2]
        # Addition of stellar lines
        y_stel = y_stel - stel_amp*np.exp(-0.5*((x-stel_ctr)/stel_wid)**2)

    y_combined = y_line*y_stel   # multiplication of stellar and telluric lines
    y_combined[y_combined < 0] = 0   # limit minimum value to zero
    return y_combined

def func_for_plotting(x, params):
    """ Function to generate multiple gaussian profiles. 
    Adapted from 
    http://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python. 
    Params are now a numpy array of values in order of xpos, ypos, 
    sigma of each gausian peak

    """
    y = np.ones_like(x)
    #print("*params inside plotting func function", type(params), 
    #    len(params), params)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = np.abs(params[i+1])  # always positive so peaks are always down
        wid = params[i+2]
        y = y - amp * np.exp(-0.5 * ((x - ctr)/wid)**2)
    y[y < 0] = 0
    return y

#######################################################################
#                                                                     #
#                        # Data manipulationions                      #
#                        # i.e. type conversions, slicing             #
#                                                                     #
#######################################################################
def split_telluric_stellar(params, num_stellar):
    """ Split up the array of lines into the telluric and stellar parts 
    input np.array(teluric lines, stellar lines)
    num_stellar is the number of stellar lines at the end
    output telluric lines, stellar lines

    """
    first_stel = -3*num_stellar  # index of first stellar line (from end)
    line_params = params[:first_stel]
    stellar_params = params[first_stel:]
    #print("line_params in func", line_params)
    #print("stellar_params in func", stellar_params)
    #print("len linelist", len(line_params), "len stellar line params", 
    #    len(stellar_params), "params length", len(params))
    assert len(line_params)%3 is 0, "Line list is not correct length"
    assert len(stellar_params)%3 is 0, "Stellar line list not correct length"
    return line_params, stellar_params,


def coords2gaussian_params(coords, delta):
    """ Convert list of coordinate tuples (x,y) into a numpy gaussian 
    paramater array for the fitting routines. Delta is the mean step 
    in the x(wl) coordinate of the data
    input form [(x1,y1),(x2,y2),...]
    output form np.array([x1,y1,sig1,x2,y2,sig2....])

    """
    if delta > 0.5:  # large steps like pixel number
        sigma = 2 * delta    # Guess standard deviation  (2 * mean wl step)
    else:
        sigma = 6 * delta    # change based on observation of spectral lines
    newcoords = []
    for coord in coords:   # Create list
        newcoords.append(coord[0])
        newcoords.append(1 - coord[1])
        newcoords.append(sigma)
    numpycoords = np.array(newcoords)
    return numpycoords    # numpy array 


def params2coords(params):
    """ Turn numpy array of gausian fit parameters into 
    (x,y) tuples of peak coordinates

    """
    coords = []
    for i in range(0, len(params), 3):
        xpos = params[i]
        ypos = params[i+1]
        coords.append((xpos, 1-ypos))
    return coords

def upper_quartile(nums):
    """Upper quartile range for normalizing"""
    nums.sort() #< Sort the list in ascending order   
    try:
        high_mid = (len(nums) - 1) * 0.75
        upq = nums[high_mid]
    except TypeError:   #<  There were an even amount of values
        # Make sure to type results of math.floor/ceil to int for use in list indices
        ceil = int(math.ceil(high_mid))
        floor = int(math.floor(high_mid))
        upq = (nums[ceil] + nums[floor]) / 2
        #print("upper quartile value", uq)
    return upq

def slice_percentage(wl, spectrum, pos, prcnt=0.15):
    """ Extract a section of a spectrum around a given wavelenght position. 
        percnt is the percentage lenght of the spectra to use.
        Returns both the sections of wavelength and spectra extracted.
        """
    span = np.abs(wl[-1] - wl[0])
    print("Span Size", span)
    print("pos", pos, type(pos))
    print("percent", prcnt, type(prcnt))
    map1 = wl > (pos - (prcnt/2)*span)
    map2 = wl < (pos + (prcnt/2)*span)
    wl_sec = wl[map1*map2]
    spectrum_sec = spectrum[map1*map2]   
    return wl_sec, spectrum_sec 

def slice_spectra(wl, spectrum, low, high):
    """ Extract a section of a spectrum between wavelength bounds. 
        percnt is the percentage lenght of the spectra to use.
        Returns both the sections of wavelength and spectra extracted.
        """
    #span = np.abs(wl[-1] - wl[0])
    #print("Span Size", span)
    print("lower bound", low)
    print("upper bound", high)
    map1 = wl > low
    map2 = wl < high
    wl_sec = wl[map1*map2]
    spectrum_sec = spectrum[map1*map2]   
    return wl_sec, spectrum_sec 

def inputer(question):
    """ Print input question above.
    To enable questions to appear when using Gooey
    """
    print(question)
    ans = input("")
    return ans
#######################################################################
#                                                                     #
#                        # Plotting Functions                         #
#                                                                     #
#                                                                     #
#######################################################################
def plot_fit(wl, Spec, params, init_params=None, title=None):
    fig = plt.figure(figsize=(15, 10))
    #http://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) # remove edge space ?
    plt.plot(wl, Spec, label="Spectrum")
    if init_params is not None:
        guessfit = func_for_plotting(wl, init_params)
        plt.plot(wl, guessfit, label="Clicked lines")  
    returnfit = func_for_plotting(wl, params)
    plt.plot(wl, returnfit, label="Fitted Lines")
    plt.title(title)
    plt.legend(loc=0)
    # Stopping scientific notation offset in wavelength
    plt.get_xaxis().get_major_formatter().set_useOffset(False)
      
    plt.show()
    return fig

def plot_both_fits(wl_a, spec_a, wl_b, spec_b, show_plot=False, paramsA=None,
    init_params_a=None, paramsB=None, init_params_b=None, title=None, 
    fitcoords_a=None, fitcoords_b=None, best_a=None, best_b=None, hor=None, 
    textloc=False, text=False):
    """ Plotting both together, many kwargs for different parts of code"""
    """ hor for add horizontal line"""
    fig2 = plt.figure(figsize=(12, 12))
    #fig.set_size_inches(25, 15, forward=False)
    ax1 = fig2.add_subplot(111)
    ax2 = ax1.twiny()
    
    # Stopping scientific notation offset in wavelength
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
      
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) # remove edge space
    # Add horizontal line at given height
    if hor is not None:
        ax1.plot(wl_b, hor*np.ones_like(wl_b), "k-.")
    ax1.plot(wl_b, spec_b, "k", label="Spectra B", lw=2)
    ax1.set_xlim(np.min(wl_b), np.max(wl_b))
    #if init_params_b is not None:
    #    guessfit_b = func_for_plotting(wl_b, init_params_b)
    #    ax1.plot(wl_b, guessfit_b, "c.", label="Guess Fit", lw=2)
    if paramsB is not None:      
        returnfit_b = func_for_plotting(wl_b, paramsB)
        ax1.plot(wl_b, returnfit_b, "r-", label="Fit", lw=2)
    ax1.set_xlabel("1st axis")
    
    bb1 = best_b is not None
    bb2 = best_b is not []
    if bb1 and bb2:
        """ Mark peaks that have already been added to cood fitted peaks 
        list to prevent doubleing up """
        for xpos in best_b:
            print("Xpos", xpos)
            ax1.plot(xpos, 1, "kx", ms=20, label="already picked", lw=4)
    
    ax2.plot(wl_a, spec_a, "g-", label="Spectra A", lw=2)
    # plot also on ax1 for legend
    ax1.plot(wl_a, spec_a, "g-", label="Spectra A", lw=2) # for label
    ax2.set_xlim(np.min(wl_a), np.max(wl_a))
    #if init_params_a is not None:
    #    guessfit_a = func_for_plotting(wl_a, init_params_a)
    #   ax2.plot(wl_a, guessfit_a, "g.", label="Guess Fit", lw=2) 
    if paramsA is not None:
        returnfit_a = func_for_plotting(wl_a, paramsA)
        ax2.plot(wl_a, returnfit_a, "m-", label="Fit A", lw=2)
        # plot also on ax1 for legend
        ax1.plot(wl_a, returnfit_a, "m-", label="Fit A", lw=2)
    ax2.set_xlabel("2nd axis")

    fita = fitcoords_a is not None
    fitb = fitcoords_b is not None
    print("fit coords a", fitcoords_a, "fit coords b", fitcoords_b)
    print("fit a", fita, "fit b", fitb)
    if fita and fitb:
        assert len(fitcoords_a) is len(fitcoords_b), " Coords not same length"
        for i in range(0, len(fitcoords_a)):
            coord_a = fitcoords_a[i]
            coord_b = fitcoords_b[i]
            ax2.plot(coord_a[0], coord_a[1], "bo", ms=15, label="Fitted A peak")
            ax1.plot(coord_a[0], coord_a[1], "bo", ms=15, label="Fitted A peak")
            ax1.plot(coord_b[0], coord_b[1], "ro", ms=15, label="Fitted B peak")
            ax2.text(coord_a[0], coord_a[1]-0.005, " "+str(i+1), fontsize=20, color='blue', 
                fontweight='bold')
            ax1.text(coord_b[0], coord_b[1]-0.005, " "+str(i+1), fontsize=20, color='red',
                fontweight='bold')
    elif fita or fitb:
        print("Only one of the fitting coords was provided")
    ba1 = best_a is not None
    ba2 = best_a is not []
    if ba1 and ba2:
        """ Mark peaks that have already been added to cood fitted peaks 
        list to prevent doubling up """
        for xpos in best_a:
            print("Xpos", xpos)
            ax2.plot(xpos, 1, "kx", ms=20, lw=5, label="Already picked line")
    ax1.legend(loc="best")
    
    if textloc and text:
        """ display text on the plot"""
        print("text location",textloc)
        ax1.text(textloc[0], textloc[1]-0.005, text, fontsize=20, color='red',
                fontweight='bold', horizontalalignment='center')
        ax2.text(textloc[0], textloc[1]-0.005, text, fontsize=20, color='red',
                fontweight='bold', horizontalalignment='center')

    if show_plot:
        plt.show(block=False)
        
    return fig2, ax1, ax2,   


def overlay_fitting(wl_a,  spec_a, wl_b, spec_b, show_plot=False, paramsA=None,
    init_params_a=None, paramsB=None, init_params_b=None, hor=None):
    """Function to plot both spectra and their fitted peaks on separate
    subplots to better see if the fits are good
     """
    fitted_a = func_for_plotting(wl_a, paramsA)
    fitted_b = func_for_plotting(wl_b, paramsB)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(12, 8))
    
    if hor is not None:
        ax1.plot(wl_a, hor*np.ones_like(wl_a), "k-.")
        ax2.plot(wl_b, hor*np.ones_like(wl_b), "k-.")
    
    ax1.plot(wl_a, spec_a, label="Spectra", lw=2)
    ax1.plot(wl_a, fitted_a, label="Fit", lw=2) 
    ax1.set_title("CRIRES NIR Spectra")
    ax1.set_xlabel("Pixel Number")
    ax1.legend(loc="best")

    ax2.plot(wl_b, spec_b, label="Spectra", lw=2)
    ax2.plot(wl_b, fitted_b, label="Fit", lw=2)
    ax2.set_title("TAPAS Telluric Spectra")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    #ax2.get_xaxis().get_major_formatter().set_scientific(False)
    ax2.legend(loc="best")
    
    if show_plot:
        plt.show(block=False)

    return fig, ax1, ax2



def print_fit_instructions():
    """ Print to screen the fitting instructions
    """
    print("\n\n Instructions: \n\n ")

    return None

#######################################################################
#                                                                     #
#                        # Mapping                                  #
#                                                                     #
#                                                                     #
#######################################################################

def wavelength_mapping(pixels, wavelengths, order=2):
    """ Generate the wavelenght map equation
      fit polynomial 
      default order of polynomial is 2

    """
    #linfit = np.polyfit(pixels, wavelengths,1)
    lin_map = np.polyfit(pixels, wavelengths, 1)
    quad_map = np.polyfit(pixels, wavelengths, 2)
    cube_map = np.polyfit(pixels, wavelengths, 3)
    quartic_map = np.polyfit(pixels, wavelengths, 4)
    
    print("linear fit x, c", lin_map)
    print("wl_quad_map equation x**2, x, c", quad_map)
    print("wl_cube_map equation x**3, x**2, x, c", cube_map)
    print("wl_quartic_map equation x**4, x**3, x**2, x, c", quartic_map)
    
    linvals = np.polyval(linfit, range(1,1025))
    quadvals = np.polyval(quad_map, range(1,1025))
    cubevals = np.polyval(cube_map, range(1,1025))
    quarticvals = np.polyval(quartic_map, range(1,1025))

    plt.plot(pixels, wavelengths , 'ko',lw=4, ms=7, label="Points")
    plt.plot(range(1,1025), quadvals, "-.r", lw=3, label="Quadfit")
    plt.plot(range(1,1025), cubevals, "-.g", lw=3, label="Cubefit")
    plt.plot(range(1,1025), quarticvals, "-.b", lw=3, label="quarticfit")
    plt.title("Plot fitted points and different fits")
    plt.legend()
    plt.show(block=False)
    print("quad fit vals ", quadvals)
    print("cube fit vals ", cubevals)
    print("quartic fit vals ", quarticvals)
    
    #lin_pointvals = np.polyval(linfit, pixels)
    #generate mapped wavelength values for the pixel positions
    quad_pointvals = np.polyval(wl_map, pixels)
    cube_pointvals = np.polyval(cube_map, pixels)
    quartic_pointvals = np.polyval(quartic_map, pixels)

    quad_diff = quad_pointvals - wavelengths
    cube_diff = cube_pointvals - wavelengths
    quartic_diff = quartic_pointvals - wavelengths

    std_diff = np.std(quad_diff)
    std_cube_diff = np.std(cube_diff)
    std_quartic_diff = np.std(quartic_diff)
    print("Differences in wavelength from mapped points to choosen points")
    print("quad_diff", quad_diff, "\ncube_diff", cube_diff, "\nquartic_diff", quartic_diff)
    print("Standard deviation of the differences (wavelength mapping error value?)")
    print("quad std", std_diff, "\ncube_std", std_cube_diff, "\nquartic_std", std_quartic_diff)
    
    return lin_map, quad_map, cube_map, quartic_map



#######################################################################
#                                                                     #
#                        # Test Case                                  #
#                                                                     #
#                                                                     #
#######################################################################

if __name__ == "__main__":

    # Do the test case HD30501-1 Chip-1
    path = "/home/jneal/Phd/Codes/Phd-codes/WavelengthCalibration/testfiles/"  # Updated for git repo
    #global param_nums
    #param_nums = 3 
    Dracspath = "/home/jneal/Phd/data/Crires/BDs-DRACS/"
    obj = "HD30501-1"
    objpath = Dracspath + obj + "/"
    chip = 0   # for testing
    hdr, DracsUncalibdata = Get_DRACS(objpath,0)
    UnCalibdata_comb = DracsUncalibdata["Combined"]
    #UnCalibdata_noda = DracsUncalibdata["Nod A"]
    #UnCalibdata_nodb = DracsUncalibdata["Nod B"]
    UnCalibdata = [range(1,1025), UnCalibdata_comb]
    Calibdata = IOmodule.read_2col(path + "Telluric_spectra_CRIRES_Chip-" + 
                                    str( 1) + ".txt")
    #print(Calibdata)
    #get_rough_peaks()
    Test_pxl_pos = [58.4375, 189.0625, 583.0, 688.1875, 715.6875, 741.8125, 
                    971.4375]
    Test_wl_pos = [2112.4897175403221, 2114.0333233870965, 2118.5533179435483,
                   2119.748622983871, 2120.0573441532256, 2120.358149395161, 
                   2122.9624895161287]

    CoordsA = Test_pxl_pos  # first one for now
    CoordsB = Test_wl_pos

    print_fit_instructions()
    #Comment Below line to skip this and work on next part
    good_coords_a, good_coords_b = adv_wavelength_fitting(UnCalibdata[0], UnCalibdata[1], 
                                       CoordsA, Calibdata[0], Calibdata[1],
                                       CoordsB)
    
    
     # to continue on with the wavelength maping 
    #('Good coords val = ', ([58.751104497854982, 81.658541491501651, 189.34108796650241, 583.07310836733564, 674.44574875440139, 688.75467383238379, 715.71056015872659, 741.04649082758874, 755.65385861534787, 971.61400877925826], [2112.5013784537928, 2112.7671666575789, 2114.0161400469569, 2118.5337956108197, 2119.5747945058301, 2119.7372298600226, 2120.0462545073347, 2120.3424115686403, 2120.505718389647, 2122.9485968267259]))
    good_coords_a = [58.751104497854982, 81.658541491501651, 189.34108796650241, 583.07310836733564, 674.44574875440139, 688.75467383238379, 715.71056015872659, 741.04649082758874, 755.65385861534787, 971.61400877925826]
    good_coords_b = [2112.5013784537928, 2112.7671666575789, 2114.0161400469569, 2118.5337956108197, 2119.5747945058301, 2119.7372298600226, 2120.0462545073347, 2120.3424115686403, 2120.505718389647, 2122.9485968267259]
    print("Skipping ahead to wavelength mapping part")
    print("Good coords vals A= ", good_coords_a)
    print("Good coords vals B = ", good_coords_b)
    wl_map, __, __ = wavelength_mapping(good_coords_a, good_coords_b)
    #""" Generate the wavelenght map
    #  fit polynomial (use pedros code)

    #"""
    print("Returned wl_map parameters", wl_map)
    
    calibrated_wl = np.polyval(wl_map, UnCalibdata[0])
    
    plt.plot(calibrated_wl, UnCalibdata[1], label="Calibrated spectra")
    plt.plot(Calibdata[0], Calibdata[1], label="Telluric spectra")
    plt.title("Calibration Output")
    plt.show()









####### 

  # calculate Sigma of gassian based on delta of wl
    #deltawl = np.mean(wl[1:]-wl[:-1])
    #print("deltawl", deltawl)
    #sig = 2 * deltawl
    #print("Sig", sig)
    #for i in range(len(Coords)):
    #for i in range(0, len(Coords), 2):
        # init_params += [Coords[i][0], Coords[i][1] ,sig]


    #prcnt = 0.05
    #Span_A = np.abs(wl_a[-1]-wl_a[0])
    #Span_B = np.abs(wl_b[-1]-wl_b[0])
    ##print("SpanA", Span_A,type(Span_A), "SpanB", Span_B,type(Span_B))


      # # print("wl_a ", type(wl_a))
      # # print(" coordsA", CoordsA,"type",type(CoordsA))
      # # print("wl_b ", type(wl_b))
      # # print("CoordsB",CoordsB, "type",type(CoordsB))
       #  wl_a_up = CoordsA[i] + (prcnt/2)*Span_A
       #  wl_a_low = CoordsA[i] - (prcnt/2)*Span_A
       #  wl_b_up = CoordsB[i] + (prcnt/2)*Span_B
       #  wl_b_low = CoordsB[i] - (prcnt/2)*Span_B
       #  mapA1 = wl_a > wl_a_low
       #  mapA2 = wl_a < wl_a_up
       #  mapB1 = wl_b > wl_b_low
       #  mapB2 = wl_b < wl_b_up
       #  compa = [wl for wl in wl_a if ( wl>wl_a_low and wl<wl_a_up)]
       #  compb = [wl for wl in wl_b if ( wl>wl_b_low and wl<wl_b_up)]
       #  compA = [ix for (ix, wl) in zip(range(len(wl_a)), wl_a) if ( wl>wl_a_low and wl<wl_a_up)]
       #  compB = [ix for (ix, wl) in zip(range(len(wl_b)), wl_b) if ( wl>wl_b_low and wl<wl_b_up)]
       # # print("Compa",len(compa),"wl vals",compa)
       # # print("CompA",len(compA),"ix vals",compA)
       #  #print("CompB",len(compB),"ix vals",compB)
       #  #print("Compb",len(compb),"wl vals",compb)
       # # print("wl_alow",type(wl_a_low),"wl_aup",type(wl_a_up))
       # # print("wl_blow",type(wl_b_low),"wl_bup",type(wl_b_up))
       # # print("wl_a_low" , wl_a_low,"wl_a_up" , wl_a_up)
       # # print("wl_b_low" , wl_b_low,"wl_b_up" , wl_b_up)
       # # print("mapA1 =" , np.sum(mapA1), "mapA2 =" , np.sum(mapA2))
       # # print("mapB1 =" ,np.sum( mapB1), "mapB2 =" , np.sum(mapB2))
       # # print("type A", type(mapA1*mapA2), "multiplyA", mapA1*mapA2)
       # # print("type B", type(mapB1*mapB2), "MultiplyB", mapB1*mapB2)
       #  multiA=mapA1*mapA2
       #  multiB=mapB1*mapB2
       #  #print("len multiA", len(multiA),type(multiA))
       #  #print("len multiB", len(multiB),type(multiB))

       #  #multiA = multiA.nonzero()
       #  #multiB = multiB.nonzero()
       #  #print("multiA nonzero", multiA)
       #  #print("multiB nonzero", multiB.nonzero())
       #  #print("type wl_a",type(wl_a),"type spec_a",type(spec_a))
       #  # wl_a_sec = wl_a[multiA]
      #   wl_a_sec = wl_a[compA]
      #   #sect_a = spec_a[multiA]
      #   sect_a = spec_a[compA]
      #   print("len wl_a_sec",len(wl_a_sec),"len sect_a",len(sect_a))
      #   #wl_b_sec = wl_b[multiB]
      #   wl_b_sec = wl_b[compB]
    
      #   #print("wl_b", type(wl_b),"wl_b_sec",type(wl_b_sec),"len",len(wl_b_sec))
      #   print("----SpectB ---",type(spec_b),"len",len(spec_b))  
      # # sect_b = spec_b[multiB]
      #   sect_b = spec_b[compB]
      #   print("sect_b type",type(sect_b))
      #   #print("sect_a",sect_a,"sect_b",sect_b)


# Old code from extract chunks

    #wl_up = wlpos + (prcnt/2)*Span
    #wl_low = wlpos - (prcnt/2)*Span
    #map1 = wl > wl_low
    
    #map2 = wl < wl_up
    
    #multimap = map1*map2
    #comp = [ix for (ix, wl) in zip(range(len(wl)), wl) if (wl>wl_low and wl<wl_up)]
    #wl_sec = wl[comp]
    #spectra_sec = spectra[comp]
    #print("len wl_sec comp", len(wl_sec), "len spectra_sec", len(spectra_sec))
    
    #wl_sec2 = wl[multimap]
    #spectra_sec2 = spectra[multimap]
    #print("Len wl_sec multimap", len(wl_sec2), "len Sect", len(spectra_sec2), "Type", type(spectra_sec2))

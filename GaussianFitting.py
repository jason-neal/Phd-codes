#!/usr/bin/env python
""" Module docstring
"""
# -*- coding: utf8 -*-

## GaussainFitting.py
from __future__ import division
#from astropy.io import fits
#import Obtain_Telluric
import IOmodule
import copy
import math
import numpy as np
#import scipy as sp 
#from Get_filenames import get_filenames
import matplotlib.pyplot as plt
import scipy.optimize as opt
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
    a_coords = get_coordinates(wl_a, spec_a, wl_b, spec_b, title="Observed Spectra") 
    b_coords = get_coordinates(wl_b, spec_b, wl_a, spec_a, title="Telluric Lines",
        points_a=a_coords)
    return a_coords, b_coords, 

def get_coordinates(wl_a, spec_a, wl_b, spec_b, title="Mark Lines on Spectra", 
        points_a=None, points_b=None):
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
        fig = plt.figure(figsize=(25, 15))
        fig.suptitle(title)
        fig.set_size_inches(25, 15, forward=False)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.plot(wl_b, spec_b, "k--", linewidth=3, label="Ref Spec")
        ax1.set_xlabel("spec_b")
        ax1.set_xlim(np.min(wl_b), np.max(wl_b))
        ax2.plot(wl_a, spec_a, "m", linewidth=5, label="Spectra to Click")
        ax2.plot(wl_a, np.ones_like(spec_a), "b-.")
        ax2.set_ylabel("Normalized Flux/Intensity")
        ax2.set_xlabel("Wavelength/pixels")
        ax2.set_xlim(np.min(wl_a), np.max(wl_a))
        ax2.legend()        ### ISSUES with legend
        print("get_coordinates points_A input ", points_a)
        print("get_coordinates points_B input ", points_b)
        if points_a is not None:
            #print("points_a", points_a)
            #xpoints = points_a[0::3]
            #ypoints = np.ones_like(points_a[1::3]) - points_a[1::3]
            xpoints = []
            ypoints = []
            for coord in points_a:
                xpoints.append(coord[0])
                ypoints.append(coord[1])
            ax2.plot(xpoints, ypoints, "g<", label="Selected A points")    
        if points_b is not None:
            #xpoints = points_b[0::3]
            #ypoints = np.ones_like(points_b[1::3]) - points_b[1::3]
            xpoints = []
            ypoints = []
            for coord in points_b:
                xpoints.append(coord[0])
                ypoints.append(coord[1])
            ax1.plot(xpoints, ypoints, "cd", label="Selected B points")    
        
        coords = fig.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1,
           mouse_pop=2, mouse_stop=3) # better way to do it
        plt.close()
        #cid = fig.canvas.mpl_connect('button_press_event', onclick)      
        #plt.show()
        #ans = raw_input("Are these the corrdinate values you want to use to fit these peaks? y/N?")
        #if ans.lower() == "y":   # Good coordinates to use
        return coords

#######################################################################
#                                                                     #
#                        # Fitting Code                               #
#                                                                     #
#                                                                     #
#######################################################################
def do_fit(wl, spec, init_params, stel=None):
    """ Perfrom fit using opt.curvefit and return the found parameters. 
    if there is stellar lines then stel will not be None.
    """
    print("Params before fit", init_params, type(init_params))
    if stel is not None:
        #use lambda instead here
        #__ is junk parameter to take the covar returned by curve_fit
        print("init params", init_params, "stellar params", stel)
        assert len(stel)%3 is 0, "stel parameters not multiple of 3"
        stelnum = int(len(stel)/3)  # number of stellar lines
        print("number of stellar lines ", stelnum)
        #def func(x,a,b):
        #    return a*x*x + b
        #for b in xrange(10):
        #   popt,pcov = curve_fit(lambda x, a: func(x, a, b), x1, x2)
        #params, __ = opt.curve_fit(func_with_stellar, wl, spec, 
        #    init_params, stel)
        #init_params.append(stel)
        init_params = np.concatenate((init_params, stel), axis=0)
        print("appended array with stellar lines", init_params)
        params, __ = opt.curve_fit(lambda x, *params: func_with_stellar(x, stelnum, params), wl, spec, 
            init_params)
    else:    
        params, __ = opt.curve_fit(func, wl, spec, init_params)
        #__ is junk parameter to take the covar returned by curve_fit
    print("Params after fit", params, "type output", type(params))
    assert len(params) is len(init_params), "len(Params) do not match"     
    #print("Test plotting the fit with output params")
    #adjusting output params back into 
    #plot_fit(wl, Spec, params, init_params=init_params)  
    return params

def wavelength_mapping(pixels, wavelengths):
    """ Generate the wavelenght map
      fit polynomial (use pedros code)

    """
    wl_map = None
    return wl_map


def adv_wavelength_fitting(wl_a, spec_a, AxCoords, wl_b, spec_b, BxCoords):
    """ Returns the 
    """
    best_a_coords = []
    best_b_coords = []
   
    wl_a = np.array(wl_a)      # make sure all are numpy arrays
    spec_a = np.array(spec_a)  # make sure all are numpy arrays
    wl_b = np.array(wl_b)      # make sure all are numpy arrays
    spec_b = np.array(spec_b)  # make sure all are numpy arrays
    delta_a = np.abs(np.mean(wl_a[1:] - wl_a[:-1]))   # average wl step
    delta_b = np.abs(np.mean(wl_b[1:] - wl_b[:-1]))
    assert len(AxCoords) is len(BxCoords), "Lenght of Coords do not match"
    for i in range(len(AxCoords)):
        wl_a_sec, sect_a = slice_spectra(wl_a, spec_a, AxCoords[i])
        wl_b_sec, sect_b = slice_spectra(wl_b, spec_b, BxCoords[i])
        
       #renormalize by upperquartile to give beter fit chance 
       # print("upper quatrile A", upper_quartile(sect_a))
        a_copy = copy.copy(sect_a)
        b_copy = copy.copy(sect_b)
        auq = upper_quartile(a_copy)
        auq = upper_quartile(b_copy)
        print ("upper quartile A", auq, type(auq))
        print ("upper quartile B", auq, type(auq))
        #sect_a = sect_a/auq
        sect_b = sect_b/auq
        #sect_a = sect_a/np.median(sect_a)
        #sect_b = sect_b/np.median(sect_b)
        
        while True:
        # Get more accurate coordinates with zoomed in sections
            a_coords = get_coordinates(wl_a_sec, sect_a, wl_b_sec, sect_b,
                title="Select Spectra Lines")  # Returns list of (x,y)
            b_coords = get_coordinates(wl_b_sec, sect_b, wl_a_sec, sect_a,
                title="Select Telluric Lines", points_a=a_coords)
            print("Returned a_coords", a_coords)
            print("Returned b_coords", b_coords)
        
            # Turn Coords of peaks into init params for fit
            init_params_a = coords2gaussian_params(a_coords, delta_a)
            init_params_b = coords2gaussian_params(b_coords, delta_b)

        # Plot guessed coords
            #plot_fit(wl_a_sec, sect_a, a_coords, title="Line Guesses")
            #plot_fit(wl_b_sec, sect_b, b_coords, title="Telluric Guesses")

        # If spectral lines do a fit with spectral lines multiplied to telluric lines
            print("Fitting a_coords", "!") 
            stel = raw_input("Are there any Stellar lines to include in the fit y/N") 
            print("Output from stel", stel)
            #stel = None   # for now
            #if stel is not None:
            if stel in ["y", "Yes", "YES" "yes"]: # Are there 
                print(" Doing Stellar fit now")
            #any spectral lines you want to add?
                # Select the stellar lines for the spectral fit
                stellarlines = get_coordinates(wl_a_sec, sect_a, wl_b_sec, sect_b, 
                    title="Select Spectral Lines", points_a=a_coords, 
                    points_b=b_coords)
                stellar_params = coords2gaussian_params(stellarlines , delta_a)
            # perform the stellar line fitting version
                fit_params_a = do_fit(wl_a_sec, sect_a, init_params_a, stel=stellar_params)
            else:
            #perform the normal fit 
                print("Init params for A to fit gausians", init_params_a)
                fit_params_a = do_fit(wl_a_sec, sect_a, init_params_a)
            
            print("Fitting B coords", "!")
            print("Init params for B to fit gausians", init_params_b)
            fit_params_b = do_fit(wl_b_sec, sect_b, init_params_b)
            # Plot Fitting values 
            print ("FitParamsA", fit_params_a)
            print ("FitParamsB", fit_params_b)
            print ("Using plot_both_fits ", " to plot fit results")
            plot_both_fits(wl_a_sec, sect_a, wl_b_sec, sect_b, paramsA=fit_params_a,
                paramsB=fit_params_b, init_params_a=None, init_params_b=None, 
                title="Displaying Fits with Originals", show_plot=True) # need to show without stoping
            
            goodfit = raw_input(" Was this a good fit? y/N?")
            if goodfit in ["yes", "y", "Y", "Yes", "YES"]:
                break   # exit while true loop
            elif goodfit in ["skip", "s"]:
                pass   # skip this fit and try another  

        # ask do you want to include all lines? if yes BestCoordsA.append(), BestCoordsB.append()
        # no - individually ask if want want each line included and append if yes
        # probably plot each individually to identify 
        include = raw_input(" Use Coordinate point" + str([CoordsA[i], CoordsB[i]]) + " y/N?")
        if include.lower() == "y" or include.lower() == "yes":
            best_a_coords.append(CoordsA[i]) # tempry filler for return
            best_b_coords.append(CoordsB[i]) # tempry filler to return same as inputs

    return best_a_coords, best_b_coords,


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
    for i in range(0, len(params), param_nums):
        ctr = params[i]
        amp = np.abs(params[i+1]) # always positive so peaks always down
        wid = params[i+2]
        y = y - amp * np.exp(-0.5 * ((x - ctr)/wid)**2)
    return y

def func_with_stellar(x, num_stellar, *params):
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
    params = params[0]
    first_stel = -3*num_stellar  # index of first stellar line (from end)
    print("num stellar", num_stellar, "first stel", -3*num_stellar)
    print("params length", len(params))
    print("params legnth minus stellar line values" , len(params)-3*num_stellar )
    print("first_stel in func", first_stel)
    line_params = params[:first_stel]
    print("line_params in func", line_params)
    stellar_params = params[first_stel:]
    print("stellar_params in func", stellar_params)
    # need to assert that lenght is multiple of 3
    print("len linelist", len(line_params), "len stellar line params", 
        len(stellar_params), "params length", len(params))
    assert len(line_params)%3 is 0, "Line list is not correct length"
    assert len(stellar_params)%3 is 0, "Stellar line list not correct length"

    for i in range(0, len(line_params), param_nums):
        ctr = line_params[i]
        amp = abs(line_params[i+1])   # always positive so peaks are downward
        wid = line_params[i+2]
        y_line = y_line - amp*np.exp(-0.5*((x-ctr)/wid)**2)  #Add teluric lines
    for i in range(0, len(stellar_params), param_nums):
        stel_ctr = stellar_params[i]
        stel_amp = abs(stellar_params[i+1])  #always positive so peaks are down
        stel_wid = stellar_params[i+2]
        # Addition of stellar lines
        y_stel = y_stel - stel_amp*np.exp(-0.5*((x-stel_ctr)/stel_wid)**2)

    y_combined = y_line*y_stel   # multiplication of stellar and telluric lines
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
    for i in range(0, len(params), param_nums):
        ctr = params[i]
        amp = np.abs(params[i+1])  # always positive so peaks are always down
        wid = params[i+2]
        y = y - amp * np.exp(-0.5 * ((x - ctr)/wid)**2)
    return y




#######################################################################
#                                                                     #
#                        # Data manipulationions                      #
#                        # i.e. type conversions, slicing             #
#                                                                     #
#######################################################################
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
    """ Turn numpy array of gausian fit parameters into (x,y) tuples of peak coordinates"""
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

def slice_spectra(wl, spectrum, pos, prcnt=0.10):
    """ Extract a section of a spectrum around a given wavelenght position. 
        percnt is the percentage lenght of the spectra to use.
        Returns both the sections of wavelength and spectra extracted.
        """
    span = np.abs(wl[-1] - wl[0])
    map1 = wl > (pos - (prcnt/2)*span)
    map2 = wl < (pos + (prcnt/2)*span)
    wl_sec = wl[map1*map2]
    spectrum_sec = spectrum[map1*map2]   
    return wl_sec, spectrum_sec 


#######################################################################
#                                                                     #
#                        # Plotting Functions                         #
#                                                                     #
#                                                                     #
#######################################################################
def plot_fit(wl, Spec, params, init_params=None, title=None):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(wl, Spec, label="Spectrum")
    if init_params is not None:
        guessfit = func_for_plotting(wl, init_params)
        plt.plot(wl, guessfit, label="Clicked lines")  
    returnfit = func_for_plotting(wl, params)
    plt.plot(wl, returnfit, label="Fitted Lines")
    plt.title(title)
    plt.legend(loc=0)
    plt.show()
    return None

def plot_both_fits(wl_a, spec_a, wl_b, spec_b, show_plot=False, paramsA=None,
    init_params_a=None, paramsB=None, init_params_b=None, title=None):
    fig = plt.figure(figsize=(10, 10))
    #fig.set_size_inches(25, 15, forward=False)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.plot(wl_b, spec_b, "k--", label="Spectra")
    ax1.set_xlim(np.min(wl_a), np.max(wl_a))
    ax2.plot(wl_a, spec_a, "g*-", label="Spectrum 2")
    ax2.set_xlim(np.min(wl_a), np.max(wl_a))
    if init_params_b is not None:
        guessfit_b = func_for_plotting(wl_b, init_params_b)
        ax1.plot(wl_b, guessfit_b, "c.", label="Guess Fit")
    if init_params_a is not None:
        guessfit_a = func_for_plotting(wl_a, init_params_a)
        ax2.plot(wl_a, guessfit_a, "g.", label="Guess Fit") 
    if paramsB is not None:      
        returnfit_b = func_for_plotting(wl_b, paramsB)
        ax1.plot(wl_b, returnfit_b, "r-.", label="Fit")
    if paramsA is not None:
        returnfit_a = func_for_plotting(wl_a, paramsA)
        ax2.plot(wl_a, returnfit_a, "m-.", label="Fit")
    ax1.set_xlabel("Cordinate 1")
    ax2.set_xlabel("Cordinate 2")
    fig.suptitle(title)
    plt.legend(loc=0)
    
    if show_plot:
        #plt.show()
        plt.display()
    return fig, ax1, ax2,   








#######################################################################
#                                                                     #
#                        # Test Case                                  #
#                                                                     #
#                                                                     #
#######################################################################

if __name__ == "__main__":

    # Do the test case HD30501-1 Chip-1

    path = "/home/jneal/Phd/Codes/Phd-codes/WavelengthCalibration/testfiles/"  # Updated for git repo
    global param_nums
    param_nums = 3 
    Dracspath = "/home/jneal/Phd/data/Crires/BDs-DRACS/"
    obj = "HD30501-1"
    objpath = Dracspath + obj + "/"
    chip = 0   # for testing
    hdr, DracsUncalibdata = Get_DRACS(objpath,0)
    UnCalibdata_comb = DracsUncalibdata["Combined"]
    #UnCalibdata_noda = DracsUncalibdata["Nod A"]
    #UnCalibdata_nodb = DracsUncalibdata["Nod B"]
    UnCalibdata = [range(1,1025), UnCalibdata_comb]
    Calibdata = IOmodule.read_2col(path + "Telluric_spectra_CRIRES_Chip-" + str( 1) + ".txt")
    #print(Calibdata)
    #get_rough_peaks()
    Test_pxl_pos = [58.4375, 189.0625, 583.0, 688.1875, 715.6875, 741.8125, 971.4375]
    Test_wl_pos = [2112.4897175403221, 2114.0333233870965, 2118.5533179435483, 2119.748622983871, 2120.0573441532256, 2120.358149395161, 2122.9624895161287]

    CoordsA = Test_pxl_pos  # first one for now
    CoordsB = Test_wl_pos
    Goodcoords = adv_wavelength_fitting(UnCalibdata[0], UnCalibdata[1], CoordsA, Calibdata[0], Calibdata[1], CoordsB)
    print("Good coords val = ", Goodcoords)
















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
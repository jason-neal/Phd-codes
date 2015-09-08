#!/usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import division
from astropy.io import fits

import IOmodule
import numpy as np
import scipy as sp 
from Get_filenames import get_filenames
import matplotlib.pyplot as plt

#from astropy.modeling import models, fitting
#from astropy.modeling import SummedCompositeModel
#from astropy.modeling.models import Gaussian1D
#from astropy.modeling.models import custom_model

import scipy.optimize as opt

def onclick(event):
    global ix, iy
    # Disconnect after right click
    if event.button == 3:
        #print("that was a right mouse click")
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
        return
    ix, iy = event.xdata, event.ydata
    #print("event button", event.button)
    # print 'x = %d, y = %d'%(
    #     ix, iy)
    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))
    return

#@custom_model
#def sum_of_gaussians(x, *params):
    #''' Could also use this to generate the data.
  #  Has been adjusted from http://docs.astropy.org/en/stable/modeling/new.html

   # x is the x-axis (wavelength or pixel number)  - should be a numpy array
    #ampltiudes, means asnd sigmas are lists of the values for each gausian we want to fit'''
    #amplitudes = params[0]
    #means = params[1]
    #sigmas = params[2]

    #modeloutput = np.ones_like(x)
    #for amp, mean, sig in zip(amplitudes, means, sigmas):
    #    modeloutput -= amp * np.exp(-0.5 * ((x - mean) / sig)**2) # Subtract each gausian

    #return modeloutput


def make_mix(numg): 
    def mix(x, *p):
        #print("inside mix")
        ng = numg
        amplitudes = p[:ng]
        means = p[ng:2*ng]
        sigmas = p[2*ng:]
        modeloutput = np.ones_like(x)
        print("type model ouput", type(modeloutput))
        print("type model ouput value", type(modeloutput[0]))
        print("amplitudes, means, sigmas", amplitudes, means, sigmas)
        for amp, mean, sig in zip(amplitudes, means, sigmas):
            #print("inside the mix for loop")
            #print("this amp", amp, "this mean", mean, "this_sigma", sig)
            subtraction = amp * np.exp(-0.5 * ((x - mean) / sig)**2)
            #print("type subtarction",type(subtraction))
            #print("type subtarction value",type(subtraction[20]))
            #print(" subtraction ", subtraction)
            #plt.plot(subtraction, label="subtracted value")
            #plt.plot(modeloutput, "k.-",label="before subtraction")
            modeloutput = modeloutput - subtraction # Subtract each gausian
            #plt.plot(modeloutput, "r.-",label="modeloutput-subtraction")
            #plt.legend()
            #plt.show()
        #print ("modeloutput after for loop", modeloutput)
        #a = sumarray(gaussian(x,p1),lorentzian(x,p2))
        #plt.plot(modeloutput)
        #plt.show()
        return modeloutput
    return mix

def mix2(x, numg, *p):
        print("inside mix 2")
        ng = numg       # number of gausians
        amplitudes = p[:ng]
        means = p[ng:2*ng]
        sigmas = p[2*ng:]
        print("number of gausians", ng,"amps", amplitudes, "means", means, "sigmas", sigmas)
        modeloutput = np.ones_like(x)
        print("type model ouput", type(modeloutput))
        for amp, mean, sig in zip(amplitudes, means, sigmas):
            print("inside the for loop")
            this_gausian = amp * np.exp(-0.5 * ((x - mean) / sig)**2)
            print("this_gausian", this_gausian)
            plt.plot(this_gausian)
            plt.title("this gausian")
            plt.show()
            print("subtraction value", amp * np.exp(-0.5 * ((x - mean) / sig)**2))
            modeloutput -= amp * np.exp(-0.5 * ((x - mean) / sig)**2) # Subtract each gausian
            print("subtractin a gausian")
            #a = sumarray(gaussian(x,p1),lorentzian(x,p2))
        print("modeloutput",modeloutput)
        plt.plot(modeloutput)
        plt.show()
        return modeloutput

def func(x, *params):
    y = np.ones_like(x)
    #print("func params", params)
    #print("func params", len(params))
    for i in range(0, len(params), 3):
        #print("params", params, "length", len(params), "range",range(0, len(params), 3)," i", i)
        ctr = params[i]
        #print("ctr",ctr)
        amp = params[i+1]
        #print("amp",amp)
        wid = params[i+2]
        #vert = params[i+3]
        #print("wid",wid)
        y = y - amp * np.exp( -0.5 * ((x - ctr)/wid)**2)
    return y

""" http://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python
#from scipy.optimize import curve_fit
#import numpy as np
#import matplotlib.pyplot as plt
#
#data = np.loadtxt('data.txt', delimiter=',')
#x, y = data
#
#plt.plot(x,y)
#plt.show()
#
#def func(x, *params):
#    y = np.zeros_like(x)
#    for i in range(0, len(params), 3):
#        ctr = params[i]
#        amp = params[i+1]
#        wid = params[i+2]
#        y = y + amp * np.exp( -((x - ctr)/wid)**2)
#    return y
#
#guess = [0, 60000, 80, 1000, 60000, 80]
#for i in range(12):
#    guess += [60+80*i, 46000, 25]   
#
#popt, pcov = curve_fit(func, x, y, p0=guess)
#print popt
#fit = func(x, *popt)
#
#plt.plot(x, y)
#plt.plot(x, fit , 'r-')
#plt.show()
"""

#HD30501-1_DRACS_Blaze_Corrected_spectra_chip-1.txt
#Telluric_spectra_CRIRES_Chip-1.txt

path = "/home/jneal/Documents/Programming/UsableScripts/WavelengthCalibration/testfiles/"


for chip in range(4):
    coords = []
    UnCalibdata = IOmodule.read_2col(path + "HD30501-1_DRACS_Blaze_Corrected_spectra_chip-" + str(chip + 1) + ".txt")
    Calibdata = IOmodule.read_2col(path + "Telluric_spectra_CRIRES_Chip-" + str(chip + 1) + ".txt")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.plot(Calibdata[0], Calibdata[1])
    ax1.set_ylabel('Transmittance')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_xlim(np.min(Calibdata[0]), np.max(Calibdata[0]))
    ax2.plot(UnCalibdata[0],UnCalibdata[1],'r')
    ax2.set_xlabel('Pixel vals')
    ax2.set_ylabel('Normalized ADU')
    ax2.set_xlim(np.min(UnCalibdata[0]), np.max(UnCalibdata[0]))
# automatically update ylim of ax2 when ylim of ax1 changes.
    #ax_f.callbacks.connect("ylim_changed", convert_ax_c_to_celsius)
    #ax_f.plot(np.linspace(-40, 120, 100))
    #ax_f.set_xlim(0, 100)

    #ax_f.set_title('Two scales: Fahrenheit and Celsius')
    #ax_f.set_ylabel('Fahrenheit')
    #ax_c.set_ylabel('Celsius')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    print(coords)


    # Calculate the ratios to determine where to try fit gausians

    xpos =[]
    ypos = []
    for tup in coords:
        xpos.append(tup[0])
        ypos.append(1-tup[1])

    #coords_x = coords[0]
    #coords_y = coords[1]
    ratio = (xpos - np.min(UnCalibdata)) / (np.max(UnCalibdata) - np.min(UnCalibdata))
    print("xpositions", xpos)
    print("y positions", ypos)
    print("x ratio", ratio)

    cal_xpos = ratio * (np.max(Calibdata[0])-np.min(Calibdata[0])) + np.min(Calibdata[0])
    print("calibration xpos cal_xpos", cal_xpos)
    """ # cal_xpos and xpos are the xpostions to try fit
    # ypos are the amplitudes
    # sig = 5?
    """
    #params = []
    #params.append(ypos)
    #print(params) 
    #params.append(list(cal_xpos))
    #print(params) 
    #params.append(list(np.ones_like(ypos)))
    #print(params) 
    params_uncalib = []
    params_calib = []
    for i in range(len(ypos)):
        params_uncalib += [xpos[i], ypos[i], 5] # center , amplitude, std
        params_calib += [cal_xpos[i], ypos[i], 5]    # center , amplitude, std

    # print("params_uncalib1", params_uncalib1)
    # print("params_calib1", params_calib1)


    #params_uncalib = ypos + list(xpos) + list(5*np.ones_like(ypos)) 
    #params_calib = ypos + list(cal_xpos) + list(5*np.ones_like(ypos))
    
    #print("params_uncalib", params_uncalib)
   
    #Spec_init = sum_of_gaussians()
    #Cal_init = sum_of_gaussians()


    #make_mix(len(ypos))
    #scipy.optimize.curve_fit(f, xdata, ydata, p0=None)
    #leastsq_uncalib, covar = opt.curve_fit(make_mix(len(ypos)),UnCalibdata[0],UnCalibdata[1],params_uncalib)
    #leastsq_calib, covar = opt.curve_fit(make_mix(len(ypos)),Calibdata[0],Calibdata[1],params_calib)
    
    
    leastsq_uncalib, covar = opt.curve_fit(func,UnCalibdata[0],UnCalibdata[1], params_uncalib)
    leastsq_calib, covar_cal = opt.curve_fit(func,Calibdata[0],Calibdata[1], params_calib)


    print("fit params uncalibrated", leastsq_uncalib) #, "covar", covar)
    print("fit params calibrated", leastsq_calib)#, "covar", covar_cal)
    print( "array into list", list(leastsq_uncalib))
    print("input params", params_uncalib)
    print("type input params", type(params_uncalib))
    print(" fit params", leastsq_uncalib)
    print("type fit params", type(leastsq_uncalib))


    print("first value ", leastsq_uncalib[0])

    print("np array length",len(np.array([1,2,3,4])))
    Fitted_uncalib = func(UnCalibdata[0], *leastsq_uncalib)
    Fitted_calib = func(Calibdata[0], *leastsq_calib)
    
    fig2 = plt.figure()
    plt.plot(UnCalibdata[0],UnCalibdata[1],'r', label="uncalib")
    plt.plot(UnCalibdata[0],Fitted_uncalib,'k.-', label="fitted")

    fig3 = plt.figure()
    plt.plot(Calibdata[0],Calibdata[1],'b', label="Calib")
    plt.plot(Calibdata[0],Fitted_calib,'k.-', label="fitted")


    plt.legend()
    plt.show()


    #Fitoutput = mix2(UnCalibdata[0],len(ypos),leastsq)

    #Fit_uncalib = mix2(UnCalibdata[0],len(ypos),leastsq_uncalib)
    #Fit_calib = mix2(Calibdata[0],len(ypos),leastsq_calib)

    #fitter = fitting.LevMarLSQFitter()
        #sol_spec = fitter(Spec_init(amplitudes=ypos, means=xpos, sigmas=np.ones_like(xpos)), UnCalibdata[0], UnCalibdata[1])
    #sol_Cal = fitter(Cal_init(amplitudes=ypos, means=cal_xpos, sigmas=np.ones_like(cal_xpos)), Calibdata[0], Calibdata[1])
    #g_int = sum_of_gaussians( amplitudes=ypos, means=xpos, sigmas=np.ones_like(xpos))


            #plt.plot(Calibdata[0], Fit_calib, label='fit calib')
    
    #plt.plot(Calibdata[0], mix2(Calibdata[0],len(ypos),params_calib), label='guess')
    
    #ax2.plot(x, y, label='data')
    #plt.plot(UnCalibdata[0], Fit_uncalib, label='fit uncalib')
    #plt.legend()
   # plt.show()



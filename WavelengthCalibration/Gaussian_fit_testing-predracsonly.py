#!/usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import division
from astropy.io import fits
import Obtain_Telluric
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
    global ix, iy, coords
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
    #global coords
    coords.append((ix, iy))
    print("Click position", [ix, iy])
    return

def func(x, *params):
    y = np.ones_like(x)
    #print("func params", params)
    #print("func params", len(params))
    for i in range(0, len(params), param_nums):
        #print("params", params, "length", len(params), "range",range(0, len(params), 3)," i", i)
        ctr = params[i]
        #print("ctr",ctr)
        amp = abs(params[i+1]) #always positive so peaks are always downward
        #print("amp",amp)
        wid = params[i+2]
        #vert = params[i+3]
        #print("wid",wid)
       # print("ctr", ctr, " type ", type(ctr))
        #print("amp", amp, " type ", type(amp))
        #print("wid", wid, " type ", type(wid))
       # print(" type y", type(y))
       # print(" type amp*np.exp", type(amp * np.exp( -0.5 * ((x - ctr)/wid)**2)))
       # print(" type np.exp", type(np.exp( -0.5 * ((x - ctr)/wid)**2)))
        y = y - amp * np.exp( -0.5 * ((x - ctr)/wid)**2)
    return y

def func4(x, *params):
    # includes vertical shift
    y = np.ones_like(x)
    #print("func params", params)
    #print("func params", len(params))
    global param_nums
    for i in range(0, len(params), param_nums):
        #print("params", params, "length", len(params), "range",range(0, len(params), 3)," i", i)
        ctr = params[i]
        amp = abs(params[i+1]) #always positive so peaks are always downward
        wid = params[i+2]
        if param_nums == 4: # doesn't work well 
            vert = params[i+3]
            mask = (x > (ctr - 1.5*wid)) * (x < (ctr + 1.5*wid))
            y = y - amp * np.exp(-0.5 * ((x - ctr)/wid)**2) + vert * mask
        else:
            y = y - amp * np.exp(-0.5 * ((x - ctr)/wid)**2) 
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

#path = "/home/jneal/Documents/Programming/UsableScripts/WavelengthCalibration/testfiles/"
path = "/home/jneal/Phd/Codes/Phd-codes/WavelengthCalibration/testfiles/"  # Updated for git repo
#path = "C:/Users/Jason/Documents/Phd/Phd-codes/WavelengthCalibration/testfiles/"  # Updated for git repo
global param_nums
param_nums = 3  # 4 does not work as well

for chip in range(4):
   
    #coordsa = []
    #coordsb = []
    UnCalibdata = IOmodule.read_2col(path + "HD30501-1_DRACS_Blaze_Corrected_spectra_chip-" + str(chip + 1) + ".txt")
    Calibdata = IOmodule.read_2col(path + "Telluric_spectra_CRIRES_Chip-" + str(chip + 1) + ".txt")

    Goodfit = False # for good line fits
    while True:
        coords = []
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.plot(Calibdata[0], Calibdata[1])
        ax1.set_ylabel('Transmittance')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_xlim(np.min(Calibdata[0]), np.max(Calibdata[0]))
        ax2.plot(UnCalibdata[0],UnCalibdata[1],'r')   #-0.03*np.ones_like(UnCalibdata[1])
        ax2.set_xlabel('Pixel vals')
        ax2.set_ylabel('Normalized ADU')
        ax2.set_xlim(np.min(UnCalibdata[0]), np.max(UnCalibdata[0]))
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        print("Left click on the maximum of each Spectra line peak (Red) that you want to fit from left to right. \nThen right click to close and perform fit")
        plt.show()
        print("coords found for first plot", coords)
        coords_pxl = coords
        xpos = []
        ypos = []
        for tup in coords_pxl:
            xpos.append(tup[0])
            ypos.append(1-tup[1])

        while True:
            coords =[]
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            ax2.plot(Calibdata[0], Calibdata[1])
            ax2.set_ylabel('Transmittance')
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_xlim(np.min(Calibdata[0]), np.max(Calibdata[0]))
            ax1.plot(UnCalibdata[0],UnCalibdata[1],'r')
            ax1.plot(xpos,np.ones_like(ypos)-ypos,"*k", linewidth=5,markersize=10)
            ax1.set_xlabel('Pixel vals')
            ax1.set_ylabel('Normalized ADU')
            ax1.set_xlim(np.min(UnCalibdata[0]), np.max(UnCalibdata[0]))
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            print("Left click on the maximum of each spectral line peak (Red) that you want to select that match the already sellected lines in order from left to right. \nThen right click to close and perform fit")
            plt.show()
            print("coords found for second plot", coords)
            coords_wl = coords
            print("coords lengths","wl",len(coords_wl), "pxl", len(coords_pxl))
            #assert len(coords_wl) == len(coords_pxl), " Choosen points were not the same so retry" 
            if len(coords_wl) == len(coords_pxl):
                break  # continue on outside while loop
            
        # Calculate the ratios to determine where to try fit gausians
        cal_xpos = []
        cal_ypos = []
        for tup in coords_wl:
            cal_xpos.append(tup[0])
            cal_ypos.append(1-tup[1])
     
        #ratio = (xpos - np.min(UnCalibdata)) / (np.max(UnCalibdata) - np.min(UnCalibdata))
        print("xpositions", xpos)
        print("y positions", ypos)
        print("cal_x wl positions", cal_xpos)
        print("cal y pos", cal_ypos)

        #cal_xpos = ratio * (np.max(Calibdata[0])-np.min(Calibdata[0])) + np.min(Calibdata[0])
        #print("calibration xpos cal_xpos", cal_xpos)
        """ # cal_xpos and xpos are the xpostions to try fit
        # ypos are the amplitudes
        # sig = 5?
        """
    
        init_params_uncalib = []
        init_params_calib = []
        for i in range(len(ypos)):
            if param_nums == 3:
                init_params_uncalib += [xpos[i], ypos[i], 1.2]        # center , amplitude, std 
            elif param_nums == 4:
                init_params_uncalib += [xpos[i], ypos[i], 1.2, 0.01]        # center , amplitude, std (vertshift)
        for i in range(len(cal_ypos)):
            if param_nums == 3:
                init_params_calib += [cal_xpos[i], cal_ypos[i], 0.04]    # center , amplitude, std  
            elif param_nums == 4:
                init_params_calib += [cal_xpos[i], cal_ypos[i], 0.04, 0.004]    # center , amplitude, std (vertshift)
       
        print("init_params_calib", init_params_calib)
        print("init_params_uncalib", init_params_uncalib)

        #leastsq_uncalib, covar = opt.curve_fit(make_mix(len(ypos)),UnCalibdata[0],UnCalibdata[1],params_uncalib)
        #leastsq_calib, covar = opt.curve_fit(make_mix(len(ypos)),Calibdata[0],Calibdata[1],params_calib)
        
        fit_params_uncalib = []
        fit_params_calib = []

        for jj in range(0,len(init_params_uncalib),param_nums):
            print("jj", jj)
            print("type jj",type(jj))
            
            print(type([jj, jj + 1, jj + 2]))
            print("[jj, jj + 1, jj + 2]",[jj,jj+param_nums])
            this_params_uncalib = init_params_uncalib[jj:jj+param_nums]
            print("this_params_uncalib", this_params_uncalib)
            this_params_calib = init_params_calib[jj:jj+param_nums]
            print("this_params_calib", this_params_calib)
            this_fit_uncalib, covar = opt.curve_fit(func, UnCalibdata[0], UnCalibdata[1], this_params_uncalib)
            this_fit_calib, covar_cal = opt.curve_fit(func, Calibdata[0], Calibdata[1], this_params_calib)
            # save parameters
            for par in range(param_nums):
                fit_params_uncalib.append(this_fit_uncalib[par])
                fit_params_calib.append(this_fit_calib[par])
            #leastsq_uncalib, covar = opt.curve_fit(func,UnCalibdata[0],UnCalibdata[1], params_uncalib)
            #leastsq_calib, covar_cal = opt.curve_fit(func,Calibdata[0],Calibdata[1], params_calib)

        print("fit params individual", fit_params_uncalib, fit_params_calib) #, "covar", covar)
        print("init_params_uncalib", init_params_uncalib)
      
        Fitted_uncalib = func(UnCalibdata[0], *fit_params_uncalib)
        Fitted_calib = func(Calibdata[0], *fit_params_calib)
        # Guess models used for fitting
        Guess_uncalib = func(UnCalibdata[0], *init_params_uncalib)
        Guess_calib = func(Calibdata[0], *init_params_calib)


        plt.figure()
        plt.subplot(211)
        plt.plot(UnCalibdata[0],UnCalibdata[1], 'r', label="uncalib")
        plt.plot(UnCalibdata[0],Guess_uncalib, 'go-', label="guess uncalib")
        plt.plot(UnCalibdata[0],Fitted_uncalib, 'k.-', label="fitted uncalib")
        plt.title("Spectral line fits")
        plt.legend()

        plt.subplot(212)
        plt.plot(Calibdata[0],Calibdata[1], 'b', label="Calib")
        plt.plot(Calibdata[0],Guess_calib, 'go-', label="guess calib")
        plt.plot(Calibdata[0],Fitted_calib, 'k.-', label="fitted calib")
        plt.title("Telluric line fits")
        plt.legend(loc="best")
        print("init params_uncalib", init_params_uncalib)
        print("fit params uncalib", fit_params_uncalib)
        print("init params_calib", init_params_calib)
        print("fit params calib", fit_params_calib)
        plt.show()

        try:
            Reply = raw_input(" Is this a good fit, y/n?")
        except:
            pass
        try:
            Reply = input(" Is this a good fit, y/n?")  #python 3.4
        except:
            pass
        if Reply == "y":
            print("Good fit found")
            break
        #Goodfit = input(" Is this a good fit")  # python 3
    # after good fit

    #### pixel map creation

    # plot positions verse wavelength
    fig4 = plt.figure()

    pixel_pos = fit_params_uncalib[0:-1:param_nums]
    wl_pos = fit_params_calib[0:-1:param_nums]
    plt.plot(pixel_pos,wl_pos,"g*", markersize=10, linewidth=7)
    plt.ylabel("Wavelength")
    plt.xlabel("Pixel position")

    #plt.plot([min(pixel_pos), max(pixel_pos)],[min(wl_pos), max(wl_pos)], "k")
    # need to fit a linear fit to this from star to end values
   
    # create wavelenght map

    # fit linear
    linfit = np.polyfit(pixel_pos, wl_pos,1)
    print("linear fit", linfit)
    quadfit = np.polyfit(pixel_pos, wl_pos,2)
    print("quad fit", quadfit)
    # fit quadratic

    linvals = np.polyval(linfit, range(1,1025))
    quadvals = np.polyval(quadfit, range(1,1025))

    plt.plot(range(1,1025), linvals , label="linearfit")
    plt.plot(range(1,1025), quadvals, "-.r",label="quadfit")
    plt.legend(loc="best")
    print("quad fit vals" , quadvals)
    plt.show()

    lin_pointvals = np.polyval(linfit, pixel_pos)
    quad_pointvals = np.polyval(quadfit, pixel_pos) 

    #plot differences in points from the fits
    diff_lin = lin_pointvals-wl_pos
    diff_quad = quad_pointvals-wl_pos
    std_diff_lin = np.std(diff_lin)
    std_diff_quad = np.std(diff_quad)
    fit_diffs = linvals-quadvals

    plt.plot(pixel_pos, diff_lin, "or", label="linfit")
    plt.plot(pixel_pos, diff_quad, "sk", label="quad fit")
    plt.plot([pixel_pos[0], 1024], [0,0], 'b--')
    plt.plot([1,1024],fit_diffs[[0,-1]],  "*g",label="End Fitting Values")
    plt.title("Differences between points and the fits")
    plt.text(400,0, "Std diff linear fit = "+str(std_diff_lin))
    plt.text(400,-.01, "Std diff quad fit = "+str(std_diff_quad))
    plt.xlabel("Pixel Position")
    plt.ylabel("Wavelenght Diff (nm)")
    plt.legend(loc="best")
    plt.show()

   
    # Perform calibration on the spectrum
    Calibrated_lin = np.polyval(linfit, UnCalibdata[0])
    Calibrated_quad = np.polyval(quadfit, UnCalibdata[0])
    # plot calibrated wavelength with telluric spectrum to see how they align now

    fig = plt.figure()
    fig.add_subplot(111)
    #ax2 = ax1.twiny()
    plt.plot(Calibdata[0], Calibdata[1], label="Telluric")
    #plt.set_ylabel('Transmittance')
    plt.xlabel('Wavelength (nm)')
    plt.xlim(np.min(Calibdata[0]), np.max(Calibdata[0]))
    plt.plot(Calibrated_lin,UnCalibdata[1], 'r', label="Lin Caibrated Spectrum")
    plt.plot(Calibrated_quad,UnCalibdata[1], 'g', label="Quad Caibrated Spectrum")
    plt.ylabel('Normalized ADU')
    plt.title("Testing Calibrated spectrum")
    plt.legend(loc="best")
    plt.show()

    CalibratedSpectra = [Calibrated_lin, UnCalibdata[1]] ## Justa test for now
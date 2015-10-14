#!/usr/bin/env python
# -*- coding: utf8 -*-

##GaussainFitting.py
from __future__ import division
from astropy.io import fits
import Obtain_Telluric
import IOmodule
import numpy as np
import scipy as sp 
from Get_filenames import get_filenames
import matplotlib.pyplot as plt
import scipy.optimize as opt

from Gaussian_fit_testing import func, Get_DRACS
## Gaussian Fitting Module
  ## develop the advanced fitting routine that fits individual windows with potentailly multiple gausians.

def onclick(event):
    global ix, iy, coords, fig, cid
    # Disconnect after right click
    if event.button == 3:   # Right mouse click
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
        return
    ix, iy = event.xdata, event.ydata
    coords.append((ix, iy))
    print("Click position", [ix, iy])
    return


def GetCoords(WLA, SpecA,WLB, SpecB, title="Comparing Spectra", PointsA=False, PointsB=False):
    """ Plotting and getting coordinates from clicks on plot """
    """Black plot = plot to get coordinates from  
     Red plot = comparison spectra """
     #Bcoords = GetCoords(WLB_sec,SectB,WLA_sec,SectA,title="TelluricLines"$    
    while True:
        global coords, fig, cid
        coords = []
        fig = plt.figure()
        fig.set_size_inches(25, 15, forward=False)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.plot(WLB,SpecB,"k",linewidth=3,label="Ref Spec" )
        ax1.set_xlabel("SpecB")
        ax1.set_xlim(np.min(WLB),np.max(WLB))
        ax1.set_title(title)
        ax1.legend()
        ax2.plot(WLA, SpecA, "r",linewidth=5,label="Coordinate spec")
        ax2.plot(WLA, np.ones_like(SpecA),"b-.")
        ax2.set_ylabel("Normalized Flux/Intensity")
        ax2.set_xlabel("Wavelength/pixels")
        ax2.set_xlim(np.min(WLA),np.max(WLA))
        ax2.legend()  ### ISSUES with legend
        
        if PointsA:
            print("PointsA",PointsA)
            xpoints = []
            ypoints = []
            for point in PointsA:
                xpoints.append(point[0])
                ypoints.append(point[1])
            ax2.plot(xpoints, ypoints, "g<", label="Selected A points")
        if PointsB:
            xpoints = []
            ypoints = [] #ax2.plot(PointsB,"cd",label="Selected B points")
            for point in PointsB:
                xpoints.append(point[0])
                ypoints.append(point[1])
            ax1.plot(xpoints, ypoints, "cd", label="Selected B points")
            
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        print("Coordinate selected =", coords)
        #ans = raw_input("Are these the corrdinate values you want to use to fit these peaks? y/N?")
        #if ans.lower() == "y":   # Good coordinates to use
        return coords

def plotFit(WL,Spec,params, init_params=False):
    plt.plot(WL,Spec,label="spectra")
    if init_params:
        GuessFit = func(WL,init_params)
        plt.plot(WL,GuessFit,label="Clicked lines")
    ReturnFit = func(WL,params)
    plt.plot(WL,ReturnFit,label="Fitted Lines")
    plt.legend(loc=0)
    plt.show()
    return None

def GetRoughCoords(WLA, SpecA,WLB,SpecB):
    """ Get rough coordinate values to use for """
    Acoords = GetCoords(WLA, SectA, WLB, SectB, title="Observations") 
    Bcoords = GetCoords(WLB, SectB,WLA, SectA, title="TelluricLines", Points=Acoords)
    return Acoords, Bcoords,  

def ExtractChunks(WL, Spectra, WLpos , prcnt=0.05):
    """ Extract a section around a given wavelenght
    Take out a section around the peak for plotting 
    --- Maybe need a better name 
percent is the percentage of spectra to use """
    Span = np.abs(WL[-1]-WL[0])
    WL_up = WLpos + (prcnt/2)*Span
    WL_low = WLpos - (prcnt/2)*Span
    map1 = WL > WL_low
    map2 = WL < WL_up
    multimap = map1*map2
    comp = [ix for (ix, wl) in zip(range(len(WL)), WL) if ( wl>WL_low and wl<WL_up)]
    WL_sec = WL[comp]
    Spectra_sec = Spectra[comp]
    print("len WL_sec comp",len(WL_sec),"len Spectra_sec",len(Spectra_sec))
    WL_sec2 = WL[multimap]
    Spectra_sec2 = Spectra[multimap]
    print("Len WL_sec multimap", len(WL_sec2), "len Sect", len(Spectra_sec2), "Type", type(Spectra_sec2))
    
    return WL_sec, Spectra_sec 

def func(x, *params):
#""" Function to generate the multiple gaussian profiles. 
#    Adapted from http://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python """
    y = np.ones_like(x)
    for i in range(0, len(params), param_nums):
        #print("params", params, "length", len(params), "range",range(0, len(params), 3)," i", i)
        ctr = params[i]
        amp = abs(params[i+1]) #always positive so peaks are always downward
        wid = params[i+2]
        y = y - amp * np.exp( -0.5 * ((x - ctr)/wid)**2)
    return y

def func_withstellar(x, *params):
#""" Function to generate the multiple gaussian profiles with stellar gausian. 
#    Adapted from http://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python """
    y = np.ones_like(x)
    for i in range(0, len(params), param_nums):
        #print("params", params, "length", len(params), "range",range(0, len(params), 3)," i", i)
        ctr = params[i]
        amp = abs(params[i+1]) #always positive so peaks are always downward
        wid = params[i+2]
        y = y - amp * np.exp( -0.5 * ((x - ctr)/wid)**2)
    return y

def FitGausians(WL, Spec, Coords, stel=False):
    """ Gausian fitting 

    if there is stellar lines then stel will not be false """
    print("Coords", Coords)
    init_params = []
    # calculate Sigma of gasuian based on delta of WL
    deltaWL = np.mean(WL[1:]-WL[:-1])
    print("deltaWL", deltaWL)
    sig = 2 * deltaWL
    print("Sig", sig)
    for i in range(len(Coords)):
    #for i in range(0, len(Coords), 2):
         init_params += [Coords[i][0], Coords[i][1] ,sig]
    print("Params before fit", init_params)
    params, covar = opt.curve_fit(func, WL, Spec, init_params)
    #this_fit_uncalib, covar = opt.curve_fit(func, UnCalibdata[0], UnCalibdata[1], this_params_uncalib)
    print("Params after fit",params)      
    plotFit(WL, Spec, params, init_params=init_params)  
    return params

def FittingLines(WLA,SpecA,AxCoords,WLB,SpecB,BxCoords):
   # plot 5% of spectra either side of each spectra
    BestCoordsA, BestCoordsB = [],[]
    assert len(AxCoords) == len(BxCoords)
    WLA = np.array(WLA)      # make sure all are numpy arrays
    SpecA = np.array(SpecA)  # make sure all are numpy arrays
    WLB = np.array(WLB)      # make sure all are numpy arrays
    SpecB = np.array(SpecB)  # make sure all are numpy arrays

    #prcnt = 0.05
    #Span_A = np.abs(WLA[-1]-WLA[0])
    #Span_B = np.abs(WLB[-1]-WLB[0])
    ##print("SpanA", Span_A,type(Span_A), "SpanB", Span_B,type(Span_B))
    
    for i in range(len(AxCoords)):
        WLA_sec, SectA = ExtractChunks(WLA, SpecA, AxCoords[i]) # split spectra
        WLB_sec, SectB = ExtractChunks(WLB, SpecB, BxCoords[i])

      # # print("WLA ", type(WLA))
      # # print(" coordsA", CoordsA,"type",type(CoordsA))
      # # print("WLB ", type(WLB))
      # # print("CoordsB",CoordsB, "type",type(CoordsB))
       #  WLA_up = CoordsA[i] + (prcnt/2)*Span_A
       #  WLA_low = CoordsA[i] - (prcnt/2)*Span_A
       #  WLB_up = CoordsB[i] + (prcnt/2)*Span_B
       #  WLB_low = CoordsB[i] - (prcnt/2)*Span_B
       #  mapA1 = WLA > WLA_low
       #  mapA2 = WLA < WLA_up
       #  mapB1 = WLB > WLB_low
       #  mapB2 = WLB < WLB_up
       #  compa = [wl for wl in WLA if ( wl>WLA_low and wl<WLA_up)]
       #  compb = [wl for wl in WLB if ( wl>WLB_low and wl<WLB_up)]
       #  compA = [ix for (ix, wl) in zip(range(len(WLA)), WLA) if ( wl>WLA_low and wl<WLA_up)]
       #  compB = [ix for (ix, wl) in zip(range(len(WLB)), WLB) if ( wl>WLB_low and wl<WLB_up)]
       # # print("Compa",len(compa),"wl vals",compa)
       # # print("CompA",len(compA),"ix vals",compA)
       #  #print("CompB",len(compB),"ix vals",compB)
       #  #print("Compb",len(compb),"wl vals",compb)
       # # print("WLAlow",type(WLA_low),"WLAup",type(WLA_up))
       # # print("wlblow",type(WLB_low),"wlbup",type(WLB_up))
       # # print("WLA_low" , WLA_low,"WLA_up" , WLA_up)
       # # print("WLB_low" , WLB_low,"WLB_up" , WLB_up)
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
       #  #print("type WLA",type(WLA),"type SpecA",type(SpecA))
       #  # WLA_sec = WLA[multiA]
      #   WLA_sec = WLA[compA]
      #   #SectA = SpecA[multiA]
      #   SectA = SpecA[compA]
      #   print("len WLA_sec",len(WLA_sec),"len SectA",len(SectA))
      #   #WLB_sec = WLB[multiB]
      #   WLB_sec = WLB[compB]
    
      #   #print("WLB", type(WLB),"WLB_sec",type(WLB_sec),"len",len(WLB_sec))
      #   print("----SpectB ---",type(SpecB),"len",len(SpecB))  
      # # SectB = SpecB[multiB]
      #   SectB = SpecB[compB]
      #   print("SectB type",type(SectB))
      #   #print("sectA",SectA,"SectB",SectB)

    # Get more accurate coordinates with zoomed in sections
        Acoords = GetCoords(WLA_sec, SectA, WLB_sec, SectB, title="Observations")  # new fucntion for plotting and getting coords
        Bcoords = GetCoords(WLB_sec, SectB, WLA_sec, SectA, title="TelluricLines", PointsA=Acoords)
        #plt.plot(WLA_sec,SectA,label="sectA")
        #plt.plot(WLB_sec,SectB,label="sectB")


    # Seek new coords of line center. Would usually be just one but now could be more


    # If spectral lines do a fit with spectral lines multiplied to telluric lines
    
        #stel= raw_input("Are there any Stellar lines to include in the fit y/N")   
        stel = False   # for now
        if stel:
        #if stel.lower() == "y" or stel.lower()== "yes" : # Are there any spectral lines you want to add?
            # Select the stellar lines for the spectral fit
            Stellarlines = GetCoords(WLA, SpecA, WLB, SpecB, title="Select Spectral Lines", PointsA=Acoords, PointsB=Bcoords)
        # perform the stellar line fitting version
            ParamsA = FitGausians(WLA,SpecA,CoordsA, stel=Stellarlines)
        else:
        #perform the normal fit 
            ParamsA = FitGausians(WLA, SpecA, Acoords)
        
        ParamsB = FitGausians(WLB, SpecB, Bcoords)
        # Plot Fitting values 
        print ("FitParamsA", ParamsA)
        print ("FitParamsB", ParamsB)
        # ask if line fits were good.
        goodfit = raw_input(" Was this a good fit? y/N?")
        
        # ask do you want to include all lines? if yes BestCoordsA.append(), BestCoordsB.append()
        # no - individually ask if want want each line included and append if yes
        # probably plot each individually to identify 
        include = raw_input(" Use Coordinate point" + str([CoordsA[i],CoordsB[i]]) + " y/N?")
        if include.lower() == "y" or include.lower() == "yes":
            BestCoordsA.append(CoordsA[i]) # tempry filler for return
            BestCoordsB.append(CoordsB[i]) # tempry filler to return same as inputs

    return BestCoordsA, BestCoordsB







if __name__=="__main__":

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
    #GetRoughCoords()
    Test_pxl_pos = [58.4375, 189.0625, 583.0, 688.1875, 715.6875, 741.8125, 971.4375]
    Test_wl_pos = [2112.4897175403221, 2114.0333233870965, 2118.5533179435483, 2119.748622983871, 2120.0573441532256, 2120.358149395161, 2122.9624895161287]

    CoordsA = Test_pxl_pos  # first one for now
    CoordsB = Test_wl_pos
    Goodcoords = FittingLines(UnCalibdata[0], UnCalibdata[1], CoordsA, Calibdata[0], Calibdata[1], CoordsB)
    print(" Good coords val = ", Goodcoords)








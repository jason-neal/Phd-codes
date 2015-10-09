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
    global ix, iy, coords
    # Disconnect after right click
    if event.button == 3:   # Right mouse click
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
        return
    ix, iy = event.xdata, event.ydata
    coords.append((ix, iy))
    print("Click position", [ix, iy])
    return


def FittingLines(WLA,SpecA,CoordsA,WLB,SpecB,CoordsB):
   # plot 5% of spectra either side of each spectra
    BestCoordsA, BestCoordsB = [],[]
    prcnt = 0.05
    Span_A = np.abs(WLA[-1]-WLA[0])
    Span_B = np.abs(WLB[-1]-WLB[0])
    print("SpanA", Span_A,type(Span_A), "SpanB", Span_B,type(Span_B))
    assert len(CoordsA) == len(CoordsB)
    WLA=np.array(WLA)
    SpacA = np.array(SpecA)
    WLB=np.array(WLB)
    SpecB = np.array(SpecB)
    for i in range(len(CoordsA)):
        print("WLA ", type(WLA))
        print(" coordsA", CoordsA,"type",type(CoordsA))
        print("WLB ", type(WLB))
        print("CoordsB",CoordsB, "type",type(CoordsB))
        WLA_up = CoordsA[i] + (prcnt/2)*Span_A
        WLA_low = CoordsA[i] - (prcnt/2)*Span_A
        WLB_up = CoordsB[i] + (prcnt/2)*Span_B
        WLB_low = CoordsB[i] - (prcnt/2)*Span_B
        mapA1 = WLA > WLA_low
        mapA2 = WLA < WLA_up
        mapB1 = WLB > WLB_low
        mapB2 = WLB < WLB_up
        compa = [wl for wl in WLA if ( wl>WLA_low and wl<WLA_up)]
        compb = [wl for wl in WLB if ( wl>WLB_low and wl<WLB_up)]
        compA = [ix for (ix, wl) in zip(range(len(WLA)), WLA) if ( wl>WLA_low and wl<WLA_up)]
        compB = [ix for (ix, wl) in zip(range(len(WLB)), WLB) if ( wl>WLB_low and wl<WLB_up)]
        print("Compa",len(compa),"wl vals",compa)
        print("CompA",len(compA),"ix vals",compA)
        #print("CompB",len(compB),"ix vals",compB)
        #print("Compb",len(compb),"wl vals",compb)
       # print("WLAlow",type(WLA_low),"WLAup",type(WLA_up))
       # print("wlblow",type(WLB_low),"wlbup",type(WLB_up))
       # print("WLA_low" , WLA_low,"WLA_up" , WLA_up)
       # print("WLB_low" , WLB_low,"WLB_up" , WLB_up)
        print("mapA1 =" , np.sum(mapA1), "mapA2 =" , np.sum(mapA2))
        print("mapB1 =" ,np.sum( mapB1), "mapB2 =" , np.sum(mapB2))
       # print("type A", type(mapA1*mapA2), "multiplyA", mapA1*mapA2)
       # print("type B", type(mapB1*mapB2), "MultiplyB", mapB1*mapB2)
        multiA=mapA1*mapA2
        multiB=mapB1*mapB2
        print("len multiA", len(multiA),type(multiA))
        print("len multiB", len(multiB),type(multiB))

        #multiA = multiA.nonzero()
        #multiB = multiB.nonzero()
        #print("multiA nonzero", multiA)
        #print("multiB nonzero", multiB.nonzero())
        print("type WLA",type(WLA),"type SpecA",type(SpecA))
       # WLA_sec = WLA[multiA]
        WLA_sec = WLA[compA]
        #SectA = SpecA[multiA]
        SectA = SpecA[compA]
        print("len WLA_sec",len(WLA_sec),"len SectA",len(SectA))
        #WLB_sec = WLB[multiB]
        WLB_sec = WLB[compB]
    
        print("WLB", type(WLB),"WLB_sec",type(WLB_sec),"len",len(WLB_sec))
        print("----SpectB ---",type(SpecB),"len",len(SpecB))  
      # SectB = SpecB[multiB]
        SectB = SpecB[compB]
        print("SectB type",type(SectB))
        #print("sectA",SectA,"SectB",SectB)
    # make double axes to overlay
        #plt.plot(WLA_sec,SectA,label="sectA")
        #plt.plot(WLB_sec,SectB,label="sectB")
    # seek new coords of line center. would usually be just one but

    # if spectral lines do a fit with spectral lines multiplied to telluric lines
    
        stel= raw_input("Are there any Stellar lines to include in the fit y/N")   
        if stel.lower() == "y" or stel.lower()== "yes" : # Are there any spectral lines you want to add?
            # Select the stellar lines for the spectral fit
            # 
        # perform the stellar line fitting version
            pass
        else:
            
        #perform the normal fit 
            pass
        # ask if line fits were good.
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
    Test_pxl_pos = [58.4375, 189.0625, 583.0, 688.1875, 715.6875, 741.8125, 971.4375]
    Test_wl_pos = [2112.4897175403221, 2114.0333233870965, 2118.5533179435483, 2119.748622983871, 2120.0573441532256, 2120.358149395161, 2122.9624895161287]

    #plt.plot(Calibdata[0])
    #plt.show()
    CoordsA = Test_pxl_pos  # first one for now
    CoordsB = Test_wl_pos
    Goodcoords = FittingLines(UnCalibdata[0],UnCalibdata[1],CoordsA,Calibdata[0],Calibdata[1],CoordsB)
    print(" Good coords val = ", Goodcoords)








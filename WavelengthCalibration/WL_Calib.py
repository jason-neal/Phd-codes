#!/usr/bin/env python
# -*- coding: utf8 -*-



"""Module for Wavelength Correction"""



from astropy.io import fits
import IOmodule
import numpy as np
import scipy as sp 
from Get_filenames import get_filenames
import matplotlib.pyplot as plt
coords = []
# Simple mouse click function to store coordinates
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    # print 'x = %d, y = %d'%(
    #     ix, iy)

    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))

    # Disconnect after 2 clicks
    #if len(coords) == 2:
     #   fig.canvas.mpl_disconnect(cid)
     #   plt.close(1)
    if event.button == 3:
        print("that was a right mouse click")
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return

def Cut_Spectra(x,y, xmin, xmax):
    """ Remove items outside of the range xmin, xmax"""
    x0 = x > xmin
    x1 = x < xmax
    newx = x[x0 * x1]
    newy = y[x0 * x1]
    return newx, newy 

def WlCalib(x_uncal,y_uncal,x_calib,y_calib, Wlmin, Wlmax):
    """Four inputs for now
    Pixel/incorrect wl values - x_uncal
    Uncalib FLux values
    Telluric WL values
    Telluric transmittence

    Returns the calibration MAP """

    # limit telluric spectra to Wlmin and Max
    x_calib, y_calib = Cut_Spectra(x_calib, y_calib, Wlmin, Wlmax)

    # Fit gausians 
    plt.figure()
    plt.plot(x_calib, y_calib)
    plt.xlabel("WL")
    plt.ylabel("Transmittance")
    plt.title("Calib Spectra")
    plt.show(2)
    fig = plt.figure()
    plt.plot(x_uncal, y_uncal)
    plt.xlabel("WL")
    plt.ylabel("Transmittance")
    plt.title("Calib Spectra")         
          
    

# Call click func
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show(1)
    print(coords)
    CalibMap = "Not finished"

    return CalibMap


def sum_of_gaussians(x, amplitudes, means, sigmas):
    ''' Could also use this to generate the data.
    Has been adjusted from http://docs.astropy.org/en/stable/modeling/new.html

    x is the x-axis (wavelength or pixel number)  - should be a numpy array
    ampltiudes, means asnd sigmas are lists of the values for each gausian we want to fit'''
    modeloutput = np.ones_like(x)
    for amp, mean, sig in zip(amplitudes, means, sigmas):
        modeloutput -= amplitude1 * np.exp(-0.5 * ((x - mean1) / sigma1)**2) # Subtract each gausian

    return modeloutput


def main():
    coords = []
    """ Code to load in some test data so that can test calibration code"""
    # GET some obs to CAlibrate
    #observes = ["HD30501-1","HD30501-2","HD30501-3"]
    #for obs in observes:
    obs = "HD30501-1"
    path = '/home/jneal/data/BrownDwarfs-PedrosCode/' + obs + '/'

	#for chip in range(4):
    chip = 0 

    I_Ped = []
    # get_filenames(path, regexp, regexp2=False):
    namelist = get_filenames(path, "*.ms.fits", "*_" + str(chip + 1) +".*")
    #print(namelist)
    for name in namelist: 	  
				#print("name", name)
        ThisFile = path + name
        I_Ped.append(fits.getdata(ThisFile,0))
        I_org = I_Ped   # Keep Original values
        for i in range(len(I_Ped)):
				#print("i", i)
            I_Ped[i] /= np.median(I_Ped[i]) # normalized values
				#print(I_Ped)
        #plt.figure()
        #plt.plot(np.transpose(I_Ped))
        #plt.xlabel("Pixel")
        #plt.ylabel("ADU")
        #plt.title(obs + " Pedro Obs ccd " + str(chip + 1))			
    Pixels = range(1, 1025) 
    Combined_DRACS = np.zeros_like(I_Ped[0])
    
   
    for i in range(8):
        Combined_DRACS += np.array(I_Ped[i])
    Combined_DRACS /= 8
    plt.figure()
    print((Pixels))
    print((Combined_DRACS))
    plt.plot(Pixels, Combined_DRACS)
    plt.xlabel("Pixel")
    plt.ylabel("ADU")
    plt.title(obs + " Pedro Obs ccd " + str(chip + 1))         
    plt.show()

    DracsHdr = fits.getheader(namelist[0])
    wl0 = DracsHdr["HIERARCH ESO INS WLEN STRT"]
    wl1 = DracsHdr["HIERARCH ESO INS WLEN END"]
    print("starting wl", wl0)
    print("end wl", wl1)
    #plt.show()

    # get the calibration file

    Tapas_path = "/home/jneal/data/Tapas/"

    filelist = get_filenames(Tapas_path, "*.ipac")

    file = filelist[0]
    import IOmodule
#   #loaddata = open(Tapas_path + file)
    #linesymbol = ["-r","--k"]
    with open(Tapas_path + file) as f:
            col1 = []
            col2 = []
            for line in f:
                firstchar = line[0]
            #print("first char =", firstchar)
                if line[0] == "\\" or line[0] == "|":
                    pass #print("Match a header line")
                else:
                    line.strip()
                    val1, val2 = line.split()
                    col1.append(val1)
                    col2.append(val2)
    Calibdata = np.array([col1,col2], dtype="float64")

    plt.figure()
    plt.plot(Calibdata[0], Calibdata[1])
    plt.xlabel("WL")
    plt.ylabel("Transmittance")
    plt.title("Telluric Spectra")         
    plt.show()


    Cal_Map = WlCalib(Pixels, Combined_DRACS, Calibdata[0], Calibdata[1], wl0, wl1)
	#Cal_Map = WlCalib(x_uncal,y_uncal,x_calib,y_calib)
    print("CalibMap ", Cal_Map)



    #Saving sections of teleuric spectra
    #for chip in range(4):
 
        # get_filenames(path, regexp, regexp2=False):
      #  namelist = get_filenames(path, "*.ms.fits", "*_" + str(chip + 1) +".*")
      #  DracsHdr = fits.getheader(namelist[0])
      #  wl0 = DracsHdr["HIERARCH ESO INS WLEN STRT"]
      #  wl1 = DracsHdr["HIERARCH ESO INS WLEN END"]
       # wl_cut, tel_cut = Cut_Spectra(Calibdata[0], Calibdata[1], wl0, wl1)
      #  Name = path + "Telluric_spectra_CRIRES_Chip-" + str(chip + 1)+ ".txt"
      #  IOmodule.write_2col(Name,wl_cut, tel_cut) 














if __name__ == "__main__":
    main()

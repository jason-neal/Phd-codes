#!/usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from astropy.io import fits
from octotribble import IOmodule
from octotribble.Get_filenames import get_filenames

# Use raw_input if running on python 2.x
if hasattr(__builtins__, 'raw_input'):
    input = raw_input


def onclick(event):
    global ix, iy, coords
    # Disconnect after right click
    if event.button == 3:  # Right mouse click
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
        return
    ix, iy = event.xdata, event.ydata
    coords.append((ix, iy))
    # print("Click position", [ix, iy])
    return


def func(x, *params):
    # """ Function to generate the multiple gaussian profiles.
    #    Adapted from http://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python """
    y = np.ones_like(x)
    for i in range(0, len(params), param_nums):
        # print("params", params, "length", len(params), "range", range(0, len(params), 3), " i", i)
        ctr = params[i]
        amp = abs(params[i + 1])  # always positive so peaks are always downward
        wid = params[i + 2]
        y = y - amp * np.exp(-0.5 * ((x - ctr) / wid) ** 2)
    return y


def func4(x, *params):
    # includes vertical shift
    y = np.ones_like(x)
    global param_nums
    for i in range(0, len(params), param_nums):
        # print("params", params, "length", len(params), "range", range(0, len(params), 3), " i", i)
        ctr = params[i]
        amp = abs(params[i + 1])  # always positive so peaks are always downward
        wid = params[i + 2]
        if param_nums == 4:  # doesn't work well
            vert = params[i + 3]
            mask = (x > (ctr - 1.5 * wid)) * (x < (ctr + 1.5 * wid))
            y = y - amp * np.exp(-0.5 * ((x - ctr) / wid) ** 2) + vert * mask
        else:
            y = y - amp * np.exp(-0.5 * ((x - ctr) / wid) ** 2)
    return y


def Get_DRACS(filepath, chip):
    """Filepath needs to point object"""
    filename = get_filenames(filepath, "CRIRE*.norm.comb.fits", "*_" + str(chip + 1) + ".*")
    # print("Filename =", filepath + filename[0])
    # print("length filename", len(filename))
    assert len(filename) is 1  # Check only one filename found
    hdr = fits.getheader(filepath + filename[0])
    data = fits.getdata(filepath + filename[0])
    return hdr, data


def RV_Calc(Lambda, deltalambda):
    """ Calcualte the Radial velocity associated to an error in wavelength calibrations"""
    c = 299792458  # Speed of Light in m/s
    assert len(Lambda) == len(deltalambda)
    Verror = [err / wl * c for err, wl in zip(deltalambda, Lambda)]
    return Verror


# def FittingLines(WLA, SpecA, CoordsA, WLB, SpecB, CoordsB):
#    # plot 5% of spectra either side of each spectra
#     BestCoordsA, BestCoordsB = [], []
#     Len_A = len(SpecA)
#     Len_B = len(SpecB)
#     assert len(CoordsA) == len(CoordsB)
#     # for i in range(len(CoordsA)):
#     mapA = [WLA < CoordsA[i] + prcnt / 2 * Len_A and WLA > CoordsA[i] - prcnt / 2 * Len_A ]
#     mapB = [WLB < CoordsB[i] + prcnt / 2 * Len_B and WLB > CoordsB[i] - prcnt / 2 * Len_B ]

#     SectA = SpecA[mapA]
#     SectB = SpecB[mapB]

#     # make double axes to overlay
#     plt.plot(WLA[mapA], SectA, label="sectA")
#     plt.plot(WLB[mapB], SectB, label="sectB")
#     # seek new coords of line center. would usually be just one but

#     # if spectral lines do a fit with spectral lines multiplied to telluric lines

#     stel= input("Are there any Stellar lines to include in the fit y/N") == y
#     if stel.lower() == "y" or stel.lower()== "yes" : # Are there any spectral lines you want to add?
#         # Select the stellar lines for the spectral fit
#
#   # perform the stellar line fitting version
#         pass
#     else:

#     # perform the normal fit
#         pass
#     # ask if line fits were good.
#     # ask do you want to include all lines? if yes BestCoordsA.append(), BestCoordsB.append()
#     # no - individually ask if want want each line included and append if yes
#     # probably plot each individually to identify


#     return BestCoordsA, BestCoordsB


if __name__ == "__main__":

    # path = "/home/jneal/Documents/Programming/UsableScripts/WavelengthCalibration/testfiles/"
    path = "/home/jneal/Phd/Codes/Phd-codes/WavelengthCalibration/testfiles/"  # Updated for git repo
    # path = "C:/Users/Jason/Documents/Phd/Phd-codes/WavelengthCalibration/testfiles/"  # Updated for git repo
    global param_nums
    param_nums = 3  # 4 does not work as well

    Dracspath = "/home/jneal/Phd/data/Crires/BDs-DRACS/"
    obj = "HD30501-1"
    objpath = Dracspath + obj + "/"

    for chip in range(4):

        hdr, DracsUncalibdata = Get_DRACS(objpath, chip)
        # print("Dracs hdr", hdr)
        # print("Dracs data ", DracsUncalibdata)
        UnCalibdata_comb = DracsUncalibdata["Combined"]
        UnCalibdata_noda = DracsUncalibdata["Nod A"]
        UnCalibdata_nodb = DracsUncalibdata["Nod B"]

        UnCalibdata = [range(1024), UnCalibdata_comb]

        # Need to get proper telluric lines from the folders for each observation

        # Orignal way I tested this
        # UnCalibdata = IOmodule.read_2col(path + "HD30501-1_DRACS_Blaze_Corrected_spectra_chip-" + str(chip + 1) + ".txt")
        Calibdata = IOmodule.read_2col(path + "Telluric_spectra_CRIRES_Chip-" + str(chip + 1) + ".txt")

        # plt.plot(UnCalibdata)
        # plt.plot(Calibdata[1], "g")
        # plt.title("Test of new combined nods fits")
        # plt.show()

        Goodfit = False  # for good line fits
        while True:
            coords = []
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            ax1.plot(Calibdata[0], Calibdata[1], label="Calib")
            ax1.plot(Calibdata[0], np.ones_like(Calibdata[1]))  # horizontal line
            ax1.set_ylabel('Transmittance')
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_xlim(np.min(Calibdata[0]), np.max(Calibdata[0]))

            ax2.plot(UnCalibdata[0], UnCalibdata[1], 'r', label="UnCalib")  # -0.03 * np.ones_like(UnCalibdata[1])
            ax2.set_xlabel('Pixel vals')
            ax2.set_ylabel('Normalized ADU')
            ax2.set_xlim(np.min(UnCalibdata[0]), np.max(UnCalibdata[0]))
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            print(
                "Left click on the maximum of each Spectra line peak (Red) that you want to fit from left to right. \nThen right click to close and perform fit")
            plt.show()
            # print("coords found for first plot", coords)
            coords_pxl = coords
            xpos = []
            ypos = []
            for tup in coords_pxl:
                xpos.append(tup[0])
                ypos.append(1 - tup[1])

            while True:
                coords = []
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax2 = ax1.twiny()
                ax2.plot(Calibdata[0], Calibdata[1])
                ax2.set_ylabel('Transmittance')
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_xlim(np.min(Calibdata[0]), np.max(Calibdata[0]))
                ax1.plot(UnCalibdata[0], UnCalibdata[1], 'r')
                ax1.plot(xpos, np.ones_like(ypos) - ypos, "*k", linewidth=5, markersize=10)
                ax1.set_xlabel('Pixel vals')
                ax1.set_ylabel('Normalized ADU')
                ax1.set_xlim(np.min(UnCalibdata[0]), np.max(UnCalibdata[0]))
                cid = fig.canvas.mpl_connect('button_press_event', onclick)
                print(
                    "Left click on the maximum of each Telluric line peak (Blue) that you want to select that match the already sellected lines in order from left to right. \nThen right click to close and perform fit")
                plt.show()
                # print("coords found for second plot", coords)
                coords_wl = coords
                # print("coords lengths", "wl", len(coords_wl), "pxl", len(coords_pxl))
                # assert len(coords_wl) == len(coords_pxl), " Choosen points were not the same so retry"
                if len(coords_wl) == len(coords_pxl):
                    break  # continue on outside while loop

            # Calculate the ratios to determine where to try fit gausians
            cal_xpos = []
            cal_ypos = []
            for tup in coords_wl:
                cal_xpos.append(tup[0])
                cal_ypos.append(1 - tup[1])

            # ratio = (xpos - np.min(UnCalibdata)) / (np.max(UnCalibdata) - np.min(UnCalibdata))
            # print("xpositions", xpos)
            # print("y positions", ypos)
            # print("cal_x wl positions", cal_xpos)
            # print("cal y pos", cal_ypos)

            # cal_xpos = ratio * (np.max(Calibdata[0]) - np.min(Calibdata[0])) + np.min(Calibdata[0])
            # print("calibration xpos cal_xpos", cal_xpos)
            """ # cal_xpos and xpos are the xpostions to try fit
            # ypos are the amplitudes
            # sig = 5?
            """
            # Test new fitting function
            # CoordsA = xpos[0]  # first one for now
            # CoordsB = cal_xpos[0]
            # Goodcoords = FittingLines(UnCalibdata[0], UnCalibdata[1], CoordsA, Calibdata[0], Calibdata[1], CoordsB)
            # print(" Good coords val = ", Goodcoords)

            init_params_uncalib = []
            init_params_calib = []
            for i in range(len(ypos)):
                if param_nums == 3:
                    init_params_uncalib += [xpos[i], ypos[i], 1.2]  # center , amplitude, std
                elif param_nums == 4:
                    init_params_uncalib += [xpos[i], ypos[i], 1.2, 0.01]  # center , amplitude, std (vertshift)
            for i in range(len(cal_ypos)):
                if param_nums == 3:
                    init_params_calib += [cal_xpos[i], cal_ypos[i], 0.04]  # center , amplitude, std
                elif param_nums == 4:
                    init_params_calib += [cal_xpos[i], cal_ypos[i], 0.04, 0.004]  # center , amplitude, std (vertshift)

            print("init_params_calib", init_params_calib)
            print("init_params_uncalib", init_params_uncalib)

            # leastsq_uncalib, covar = opt.curve_fit(make_mix(len(ypos)), UnCalibdata[0], UnCalibdata[1], params_uncalib)
            # leastsq_calib, covar = opt.curve_fit(make_mix(len(ypos)), Calibdata[0], Calibdata[1], params_calib)

            fit_params_uncalib = []
            fit_params_calib = []

            for jj in range(0, len(init_params_uncalib), param_nums):
                # print("jj", jj)
                # print("type jj", type(jj))

                # print(type([jj, jj + 1, jj + 2]))
                # print("[jj, jj + 1, jj + 2]", [jj, jj + param_nums])
                this_params_uncalib = init_params_uncalib[jj:jj + param_nums]
                # print("this_params_uncalib", this_params_uncalib)
                this_params_calib = init_params_calib[jj:jj + param_nums]
                # print("this_params_calib", this_params_calib)
                this_fit_uncalib, covar = opt.curve_fit(func, UnCalibdata[0], UnCalibdata[1], this_params_uncalib)
                this_fit_calib, covar_cal = opt.curve_fit(func, Calibdata[0], Calibdata[1], this_params_calib)
                # save parameters
                for par in range(param_nums):
                    fit_params_uncalib.append(this_fit_uncalib[par])
                    fit_params_calib.append(this_fit_calib[par])
                    # leastsq_uncalib, covar = opt.curve_fit(func, UnCalibdata[0], UnCalibdata[1], params_uncalib)
                    # leastsq_calib, covar_cal = opt.curve_fit(func, Calibdata[0], Calibdata[1], params_calib)

            # print("fit params individual", fit_params_uncalib, fit_params_calib) # , "covar", covar)
            # print("init_params_uncalib", init_params_uncalib)

            Fitted_uncalib = func(UnCalibdata[0], *fit_params_uncalib)
            Fitted_calib = func(Calibdata[0], *fit_params_calib)
            # Guess models used for fitting
            Guess_uncalib = func(UnCalibdata[0], *init_params_uncalib)
            Guess_calib = func(Calibdata[0], *init_params_calib)

            plt.figure()
            plt.subplot(211)
            plt.plot(UnCalibdata[0], UnCalibdata[1], 'r', label="uncalib")
            plt.plot(UnCalibdata[0], Guess_uncalib, 'go-', label="guess uncalib")
            plt.plot(UnCalibdata[0], Fitted_uncalib, 'k.-', label="fitted uncalib")
            plt.title("Spectral line fits")
            plt.legend()

            plt.subplot(212)
            plt.plot(Calibdata[0], Calibdata[1], 'b', label="Calib")
            plt.plot(Calibdata[0], Guess_calib, 'go-', label="guess calib")
            plt.plot(Calibdata[0], Fitted_calib, 'k.-', label="fitted calib")
            plt.title("Telluric line fits")
            plt.legend(loc="best")
            print("init params_uncalib", init_params_uncalib)
            print("fit params uncalib", fit_params_uncalib)
            print("init params_calib", init_params_calib)
            print("fit params calib", fit_params_calib)
            plt.show()

            try:
                Reply = input(" Is this a good fit, y/n?")
            except:
                pass
            # try:
            #    Reply = input(" Is this a good fit, y/n?")  # python 3.4
            # except:
            #    pass
            if Reply == "y":
                print("Good fit found")
                break
                # Goodfit = input(" Is this a good fit")  # python 3
        # after good fit

        # ### pixel map creation

        # plot positions verse wavelength
        fig4 = plt.figure()

        pixel_pos = fit_params_uncalib[0:-1:param_nums]
        wl_pos = fit_params_calib[0:-1:param_nums]
        plt.plot(pixel_pos, wl_pos, "g*", markersize=10, linewidth=7)
        plt.ylabel("Wavelength")
        plt.xlabel("Pixel position")

        # plt.plot([min(pixel_pos), max(pixel_pos)], [min(wl_pos), max(wl_pos)], "k")
        # need to fit a linear fit to this from star to end values

        # create wavelength map

        # fit linear
        linfit = np.polyfit(pixel_pos, wl_pos, 1)
        print("linear fit", linfit)
        quadfit = np.polyfit(pixel_pos, wl_pos, 2)
        print("quad fit", quadfit)
        # fit quadratic

        linvals = np.polyval(linfit, range(1, 1025))
        quadvals = np.polyval(quadfit, range(1, 1025))

        plt.plot(range(1, 1025), linvals, label="linearfit")
        plt.plot(range(1, 1025), quadvals, "-.r", label="quadfit")
        plt.legend(loc="best")
        print("quad fit vals", quadvals)
        plt.show()

        lin_pointvals = np.polyval(linfit, pixel_pos)
        quad_pointvals = np.polyval(quadfit, pixel_pos)

        # plot differences in points from the fits
        diff_lin = lin_pointvals - wl_pos
        diff_quad = quad_pointvals - wl_pos
        std_diff_lin = np.std(diff_lin)
        std_diff_quad = np.std(diff_quad)
        fit_diffs = linvals - quadvals

        plt.plot(pixel_pos, diff_lin, "or", label="linfit")
        plt.plot(pixel_pos, diff_quad, "sk", label="quad fit")
        plt.plot([pixel_pos[0], 1024], [0, 0], 'b--')
        plt.plot([1, 1024], fit_diffs[[0, -1]], "*g", label="End Fitting Values")
        plt.title("Differences between points and the fits")
        plt.text(400, 0, "Std diff linear fit = " + str(std_diff_lin))
        plt.text(400, -.01, "Std diff quad fit = " + str(std_diff_quad))
        plt.xlabel("Pixel Position")
        plt.ylabel("Wavelength Diff (nm)")
        plt.legend(loc="best")
        plt.show()

        # # Velocity error of fits
        Verrors_lin = RV_Calc(wl_pos, diff_lin)
        Verrors_quad = RV_Calc(wl_pos, diff_quad)
        Verrors_ends = RV_Calc(linvals[[0, -1]], fit_diffs[[0, -1]])
        plt.plot(wl_pos, Verrors_lin, "*", label="linearfit")
        plt.plot(wl_pos, Verrors_quad, "s", label="quadfit")
        plt.plot([1, 1024], Verrors_ends, "<", label="Ends of fits")
        plt.ylabel("Velocity (m/s)")
        plt.xlabel("Wavelength (nm)")
        plt.title("Velocity errors due to fits")
        plt.legend()
        plt.show()

        # Perform calibration on the spectrum
        Calibrated_lin = np.polyval(linfit, UnCalibdata[0])
        Calibrated_quad = np.polyval(quadfit, UnCalibdata[0])
        # plot calibrated wavelength with telluric spectrum to see how they align now

        fig = plt.figure()
        fig.add_subplot(111)
        # ax2 = ax1.twiny()
        plt.plot(Calibdata[0], Calibdata[1], label="Telluric")
        # plt.set_ylabel('Transmittance')
        plt.xlabel('Wavelength (nm)')
        plt.xlim(np.min(Calibdata[0]), np.max(Calibdata[0]))
        plt.plot(Calibrated_lin, UnCalibdata[1], 'r', label="Lin Caibrated Spectrum")
        plt.plot(Calibrated_quad, UnCalibdata[1], 'g', label="Quad Caibrated Spectrum")
        plt.ylabel('Normalized ADU')
        plt.title("Testing Calibrated spectrum")
        plt.legend(loc="best")
        plt.show()

        CalibratedSpectra = [Calibrated_lin, UnCalibdata[1]]  # Just a test for now

#!/usr/bin/env python3
# -*- coding: utf8 -*-
""" This code shows that the tapas correction is equavalent
 to using Pyastronomy helcorr correction and doppler shift functions"""
import os
# import time
import ephem
# import datetime
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.io import fits
import argparse
import GaussianFitting as gf
import Obtain_Telluric as obt

from PyAstronomy import pyasl

def ra2deg(ra):
    split = ra.split(":")
    deg = float(split[0]) * 15.0 + float(split[1]) / 4.0 + float(split[2]) / 240.0
    return deg

def dec2deg(dec):
    #  degrees ( ° ), minutes ( ' ), and seconds ( " )
    # convert to degrees in decimal
    split = dec.split(":")
    print(split)
    if float(split[0]) < 0:
        deg = abs(float(split[0])) + (float(split[1]) + (float(split[2]) / 60) ) / 60
        deg *= -1
    else:
        deg = float(split[0]) + (float(split[1]) + (float(split[2]) / 60) ) / 60
    return deg

####### LOAD IN TELLURIC DATA ######
tapas1_path = "/home/jneal/Phd/data/Tapas/hd30501-50000-bervcorrected/"
Bervname = "tapas_HD30501_1_R50000_2012-04-07T00:20:00.ipac"

BervData, BervHdr = obt.load_telluric(tapas1_path, Bervname)

tapas2_path = "/home/jneal/Phd/data/Tapas/"
NoBervname = "tapas_HD30501_1_R50000_Vac_NoBERV_2012-04-07T00:20:00.ipac"

NoBervData, NoBervHdr = obt.load_telluric(tapas2_path, NoBervname)

Berv_wl = BervData[0]
Berv_trans = BervData[1]

NoBerv_wl = NoBervData[0]
NoBerv_trans = NoBervData[1]

### Corrdinates for BERV Correction
# From Tapas file
print(NoBervHdr)
obs_alt = float(NoBervHdr["SITEELEV"])
obs_long = float(NoBervHdr["SITELONG"])
obs_lat = float(NoBervHdr["SITELAT"])
print("obs_long ", obs_long)
ra = NoBervHdr["RA"]
ra_deg = ra2deg(ra)
print("ra decimal", ra_deg)

dec = NoBervHdr["DEC"]
dec_deg = dec2deg(dec)
print("dec decimal", dec_deg)

Time =  NoBervHdr["DATE-OBS"]

jd =  ephem.julian_date(Time)
print("jd tapas", jd)

# From My book
obs_alt_manual = 2635
obs_lat_manual = -24.6275
obs_long_manual = -70.4044

ra_manual = "04:45:38"
ra_deg_manual = ra2deg(ra_manual)

print("ra decimal", ra_deg_manual)

dec_manual = "-50:04:38"
dec_deg_manual = dec2deg(dec_manual)
print("dec decimal", dec_deg_manual)

Time_manual = "2012-04-07 00:20:00"
print("Time_manual", Time_manual)

jd_manual = ephem.julian_date(Time_manual)
print("JD manual", jd_manual)


# Apply corrections
tapas_barycorr = pyasl.baryCorr(jd, ra_deg, dec_deg, deq=0.0)

tapas_helcorr = pyasl.helcorr(obs_long, obs_lat, obs_alt, ra_deg, dec_deg, jd, debug=False)
my_barycorr = pyasl.baryCorr(jd_manual, ra_deg_manual, dec_deg_manual, deq=0.0)
my_helcorr = pyasl.helcorr(obs_long_manual, obs_lat_manual, obs_alt_manual, ra_deg_manual, dec_deg_manual, jd_manual, debug=False)
# helcorr calculates the motion of an observer in the direction of a star
print("Tapas barycorr", tapas_barycorr)
print("Tapas hellcorr", tapas_helcorr)
print("My barycorr", my_barycorr)
print("My hellcorr", my_helcorr)

nflux1, wlprime1= pyasl.dopplerShift(NoBerv_wl, NoBerv_trans, tapas_helcorr[0], edgeHandling=None, fillValue=None)
nflux2, wlprime2 = pyasl.dopplerShift(NoBerv_wl, NoBerv_trans, my_helcorr[0], edgeHandling=None, fillValue=None)

#### Plot berv correction stuff
plt.figure()
plt.plot(Berv_wl, Berv_trans, "k.-", label="Berv Corrected")
plt.plot(NoBerv_wl, NoBerv_trans, "g.-", label="No Berv")
plt.plot(wlprime1, NoBerv_trans, "rs-", label="Tapas Values correction")
plt.plot(wlprime2, NoBerv_trans, "c*-", label="Manual Values correction")
plt.xlim([2123, 2126])
plt.ylim([.96, 1])
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.legend(loc=0)
plt.show()

#### Plot berv correction stuff
plt.figure()
plt.plot(Berv_wl, Berv_trans, "k.-", label="Berv Corrected")
plt.plot(NoBerv_wl, NoBerv_trans, "g.-", label="No Berv")
plt.plot(NoBerv_wl, nflux1, "rs-", label="Tapas Values correction shifted")
plt.plot(NoBerv_wl, nflux2, "c*-", label="Manual Values correction shifted")
plt.xlim([2123, 2126])
plt.ylim([.96, 1])
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.legend(loc=0)
plt.show()

#### Plot berv correction stuff
plt.figure()
plt.plot(Berv_wl, Berv_trans, "k.-", label="Berv Corrected")
plt.plot(NoBerv_wl, NoBerv_trans, "g.-", label="No Berv")
plt.plot(NoBerv_wl, nflux1, "cs-", label="Tapas flux shift")
plt.plot(wlprime1, NoBerv_trans, "rs-", label="Tapas wl shift")
plt.xlim([2123, 2126])
plt.ylim([.96, 1])
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.legend(loc=0)

#### Plot berv correction stuff
plt.figure()
plt.plot(Berv_wl, Berv_trans, "k.-", label="Berv Corrected")
plt.plot(NoBerv_wl, NoBerv_trans, "g.-", label="No Berv")
plt.plot(wlprime2, NoBerv_trans, "r*-", label="Manual Wl shift")
plt.plot(NoBerv_wl, nflux2, "c*-", label="Manual Flux shift")
plt.xlim([2123, 2126])
plt.ylim([.96, 1])
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.legend(loc=0)
plt.show()

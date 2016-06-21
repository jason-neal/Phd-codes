## Berv correction from Tapas 

import time
import datetime
from PyAstronomy import pyasl

def ra2deg(ra):
    split = ra.split(":")
    deg = float(split[0])*15.0 + float(split[1])/4.0 + float(split[2])/240.0 
    return deg

def dec2deg(dec):
	#  degrees ( o ), minutes ( ' ), and seconds ( " )
	#convert to degrees in decimal
    split = dec.split(":")
    print(split)
    if float(split[0]) < 0:
        deg = abs(float(split[0])) + (float(split[1]) + (float(split[2])/60) )/60
        deg *= -1 
    else:
        deg = float(split[0]) + (float(split[1]) + (float(split[2])/60) )/60 
    return deg

def tapas_helcorr(hdr):
    """helcorr value from tapas header"""

    obs_long = float(hdr["SITELONG"])
    obs_lat = float(hdr["SITELAT"])
    obs_alt = float(hdr["SITEELEV"])

    ra = hdr["RA"]
    ra_deg = ra2deg(ra)

    dec = hdr["DEC"]
    dec_deg = dec2deg(dec)
    
    Time =  hdr["DATE-OBS"]
    Time_time = time.strptime(Time, "%Y/%m/%d %H:%M:%S")
    dt = datetime.datetime(*Time_time[:6])
    jd  = pyasl.asl.astroTimeLegacy.jdcnv(dt)
    
    tapas_helcorr = pyasl.helcorr(obs_long, obs_lat, obs_alt, ra_deg, dec_deg, jd, debug=False)

    return tapas_helcorr




#NoBerv_wl = NoBervData[0]
#NoBerv_trans = NoBervData[1]


## Apply corrections
#tapas_barycorr = pyasl.baryCorr(jd, ra_deg, dec_deg, deq=0.0)

#tapas_helcorr = pyasl.helcorr(obs_long, obs_lat, obs_alt, ra_deg, dec_deg, jd, debug=False)
#

#nflux, wlprime = pyasl.dopplerShift(NoBerv_wl, NoBerv_trans, tapas_helcorr[0], edgeHandling=None, fillValue=None)


#nflux : array
    #The shifted flux array at the old input locations.

#wlprime : array
    #The shifted wavelength axis.

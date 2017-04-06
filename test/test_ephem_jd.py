# Test ephem pacakge

# Jason Neal
# December 7, 2016

import ephem
import numpy as np
import datetime
import time
import PyAstronomy.pyasl as pyasl

# For tapas berv correction I was testing using this code
Time_manual = "2012-04-07 00:20:00"
print("Time_manual", Time_manual)

Time_manual_time = time.strptime(Time_manual, "%Y-%m-%d %H:%M:%S")
dt_manual = datetime.datetime(*Time_manual_time[:6])
jd_manual = pyasl.asl.astroTimeLegacy.jdcnv(dt_manual)
jd_manual_red = pyasl.asl.astroTimeLegacy.juldate(dt_manual)
print("jd manual", jd_manual)
print("jd manual", jd_manual_red)

# recently dicsocered ephem has a julian_date component.
jd_ephem = ephem.julian_date(Time_manual)
print("jd Time from ephem = {}".format(jd_ephem))
# This is a much nicer option!

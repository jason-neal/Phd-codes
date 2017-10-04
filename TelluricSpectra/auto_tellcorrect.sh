#!/usr/bin/env sh

# Script to run on the wavelength calibrated spectra.
# Does telluric and h20 on all .wavecal.fits files present

# Jason Neal July 2017
for i in $( ls *.wavecal.fits ); do
    TellRemoval.py -n -e -t ../Telluric_files/ -c $i   # separate constituients
    TellRemoval.py -n -e -t ../Telluric_files/ $i
    done

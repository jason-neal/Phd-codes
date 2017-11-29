#!/usr/bin/env sh

# Script to run on the telluric corrected spectra, on all .wavecal.tellcorr.fits files present
# Berv corrects spectra
# Creates a telluric mask file,
# Masks the observations


# Jason Neal November 2017
for i in $( ls *-mixavg-*tellcorr_*.fits ); do
   berv_and_mask.py -b -e $i
done

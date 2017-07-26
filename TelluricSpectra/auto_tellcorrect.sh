# Script to run on the wavelenght calibrated spectra.
# Does telluric and h20 on all .wavecal.fits files present

# Jason Neal July 2017
for i in $( ls *.wavecal.fits ); do
    TellRemoval.py  -e -c -n -t ../Telluric_files/ $i
    TellRemoval.py  -e -n -t ../Telluric_files/ $i
    done

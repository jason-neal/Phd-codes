## Spectral Differential Method

Method to recover the faint spectrum of a companion by mutual subtraction of the host star spectra.

This script is meant to take in two input fits files of the two spectra, and produce the differential spectrum.

It will use the information located in the fits header and a database of system orbital parameters to calculate the RV offset to apply to each spectrum.

It will then interpolate each spectrum to the same wavelength and make the subtraction.


The ability to create a simulated synthetic spectra from a phoenix aces spectra at the same RV shifts will also be possible to compare to observations.


##### Note:
The current systems analyzed contained too small of a RV separation to produce a significant signal in the differential.
 

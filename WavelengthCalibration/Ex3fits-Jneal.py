#!/usr/bin/python
## Python For Astronomers
## Exercises #3 - Adding to fits file

## imports:
from astropy.io import fits
import time


## My functions:
def change_hdr(keys, values, output='newfits.fits',item=0):
    ''' Open a fits file and change parameters, 
	can take list or tuple as input of keywords 
	and values to change in the header 
	Defaults at changing the header in the 0th item 
	unless the number the index is givien,
	If a key is not found it adds it to the header'''
	# open fits file
    hdulist = fits.open(output)
    hdr = hdulist[item].header
    #print repr(hdr[0:10])
    #assert type(keys) == type(values), 'keys and values do not match'
    if type(keys) == str:   		# To handle single value
        hdr[keys] = values
    else:
        assert len(keys) == len(values), 'Not the same number of keys as values' 
        for i in range(len(keys)):
            hdr[keys[i]] = values[i]
            print repr(hdr[0:10])
	# Save fits file with new parameters
    fits.writeto(output, hdulist[item].data, hdr, clobber=True)


# Main program:
def main():
    ''' Play with changing fits parameters'''
    keys = ['NAXIS', 'NAXIS2', 'FACT', 'Author']
    values = (3, 302, 'The moon is made of cheese', 'Jason')
    change_hdr(keys, values)   # Change with list/tupple
    change_hdr('Object', 'Sine wave at the beach')  # Change with single entry
    change_hdr('Comments', 'Some comment about adding comments to a fits file.')  
    change_hdr('Imprtant', 'This research has been faked and/or stolen. \
        Proceed cautiously.')
    T_Now = str(time.gmtime()[0:6])
   # Turn time into string
    change_hdr('Time', (T_Now, ' Time fits was last changed'))  
    # Print new header after changing it
    hdulist = fits.open('newfits.fits')
    print(repr(hdulist[0].header[0:15]))

if __name__ == "__main__":
    main()

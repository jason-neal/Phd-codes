from __future__ import print_function, division

import py.test

from TellRemoval import  divide_spectra, match_wl, airmass_scaling

import numpy as np



# Testing Telluric removal code


def test_divide_spectra():
    assert divide_spectra() 
    assert divide_spectra(np.array([1,2,1,2,3,1]), np.array([1,2,1,2,3,1])) == np.array(np.array([1,1,1,1,1,1]))



def test_match_wl():
    a = [1, 3, 5, 7, 9]
    b = [2, 4, 6, 8, 10]
    c = [1,2,3,4,5,6,7,8,9]
    assert match_wl(a,b,c) == [2,3,4,5,6,7,8,9,10]
    assert match_wl(np.array(a),np.array(b),np.array(c)) == np.array([2,3,4,5,6,7,8,9,10])


def test_airmass_scaling():
    """ Test of airmass scaling function"""
    assert airmass_scaling(np.array([3]), np.array([2]), 1) == 3/2
    # need mire tests here


    if __name__ == "__main__":
        test_divide_spectra()
        test_match_wl()
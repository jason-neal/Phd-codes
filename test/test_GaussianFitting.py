#!/usr/bin/env python

import hypothesis.strategies as st
import pytest
from hypothesis import given

import GaussianFitting as gf


@given(st.tuples(st.float(), st.float()), st.float())
def test_params_and_coords_reversed(coords, delta):
    # Make sure the x and y coordinates remain the same
    assert gf.params2coords(gf.coords2gaussian_params(coords, delta))[0,2] == coords

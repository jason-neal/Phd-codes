#!/usr/bin/env python
import os
import subprocess
import sys

handy_location = "/home/jneal/.handy_spectra/"

files = [f for f in os.listdir('.') if os.path.isfile(f)]
files = [f for f in files if "wavecal." in f]
files = [f for f in files if "tellcorr.fits" in f]

target_name = os.getcwd().split("/")[-2]

for f in files:
    chip_num = f.split(".nod.ms.norm.")[0][-1]

    if "mixavg" in f:
        comb = "mixavg"
    elif "sum" in f:
        comb = "sum"
    else:
        comb = ""

    if "h2otellcorr" in f:
        tell = "tellcorr"
    else:
        tell = "h2otellcorr"

    new_name = "{0}-{2}-{3}_{1}.fits".format(target_name, chip_num, comb, tell)
    # copy then move
    subprocess.call("cp {0} {0}_cp".format(f), shell=True)
    subprocess.call("mv {0}_cp {1}{2}".format(f, handy_location, new_name), shell=True)

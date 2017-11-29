# Script to produce the output of the spectral differential method:

import argparse
import logging

from astropy.io import fits
from octotribble.Get_filenames import get_filenames
from spectrum_overload import DifferentialSpectrum as DiffSpec
from spectrum_overload import Spectrum

debug = logging.debug


def _parser():
    parser = argparse.ArgumentParser(description='Differential analysis.')
    parser.add_argument('spectrum_1', help='spectrum filename', type=str)
    parser.add_argument('spectrum_2', help='spectrum filename', type=str)
    parser.add_argument("synthetic", help="Synthetic spectrum of expected companion. (Phoenix-ACES)")

    parser.add_argument("--debug", help="Turning on debug output", action='store_true', default=False)
    return parser.parse_args()


def main(spectrum_1, spectrum_2, synthetic=None):
    spec1 = load_spectrum(spectrum_1)
    spec2 = load_spectrum(spectrum_2)

    dspec = DiffSpec(spec1, spec2)


# Pass to the differential class:
#    Calculate RV from date

#    Shift spectra to a common reference
     # dspec.reference_shift()
#    Wavelength match - Interpolate etc
    # dspec.wave_match()
    # result = dspec.subtract()

    # dspec.displayplot()

    # syntheic spectrum

    if synthetic:
        # Perform the differntial on the synthetic spectrum.
        wl = fits.getdata("phoenix_aces wl")
        synth_flux = fits.getdata(synthetic)
        # Same spectrum with different headers. These headers will be used to calcualte the RV shift for the difference.
        synth1 = Spectrum(xaxis=wl, flux=synth_flux, header=spec1.header)
        synth2 = Spectrum(xaxis=wl, flux=synth_flux, header=spec2.header)

        synth_diff = DiffSpec(synth1, synth2)

        # synth_diff.displayplot()

def load_spectrum(name):
    data = fits.getdata(name)
    logging.debug("Columns present in fits file, {}".format(data.columns))
    hdr = fits.getheader(name)
    wl = data["Wavelength"]
    I = data["Corrected_DRACS"]

    spec = Spectrum(flux=I, xaxis=wl, calibrated=True, header=hdr)
    return spec


if __name__ == "__main__":

    # args = vars(_parser())
    # debug_on = args.pop('debug')
    debug_on = True
    # opts = {k: args[k] for k in args}
    if debug_on:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(asctime)s %(levelname)s %(message)s')

    # This stuff will be replace by the parser. included here to test things while creating
    chip_num = 1
    obs_num = "1"
    ref_num = "3"
    target = "HD30501-" + obs_num
    ref_target = "HD30501-" + ref_num    # should be different from target

    if target == ref_target:
        raise ValueError("Reference target should be different from target")

    dracs_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/{0}/Combined_Nods/".format(target)
    tellcorr_name = get_filenames(dracs_path, "CRIRE.*", "*{}.nod.ms.norm.sum.wavecal.tellcorr.fits".format(chip_num))
    tellcorr_name = dracs_path + tellcorr_name[0]

    ref_path = "/home/jneal/Phd/data/Crires/BDs-DRACS/{0}/Combined_Nods/".format(ref_target)
    reftellcorr_name = get_filenames(ref_path, "CRIRE.*", "*{}.nod.ms.norm.sum.wavecal.tellcorr.fits".format(chip_num))
    reftellcorr_name = ref_path + reftellcorr_name[0]

    main(tellcorr_name, reftellcorr_name)
    # main(opts)

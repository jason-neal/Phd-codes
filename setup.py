
"""Setup.py for phd-codes."""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from setuptools import find_packages

config = {
    'description': 'Phd Codes wavelenght calibration and telluric correction.',
    'author': 'Jason Neal',
    'url': 'https://github.com/jason-neal/Phd-codes.git',
    'download_url': 'https://github.com/jason-neal/Phd-codes.git',
    'author_email': 'jason.neal@astro.up.pt',
    'version': "0.1.1",
    'license': 'MIT',
    'setup_requires': ['pytest-runner'],
    'install_requires': ["astropy", "scipy", "numpy", "pyastronomy", 'pytest'],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    'extras_require': {
        'dev': ['check-manifest'],
        'test': ['coverage', 'pytest', 'pytest-cov', 'python-coveralls', 'hypothesis'],
    },
    'packages': find_packages(exclude=['contrib', 'docs', 'tests']),
    #   find_packages("src", exclude=['contrib', 'docs', 'tests'])   'packages': +['utilities', 'simulators'],

    'scripts': ["TelluricSpectra/TellRemoval.py",
                "TelluricSpectra/auto_tellcorrect.sh",
                "handy_spectra.py",
                "WavelengthCalibration/wl_calibrate.py",
                "TelluricSpectra/berv_and_mask.py",
                "TelluricSpectra/auto_berv_and_mask_handy.sh",
                "TelluricSpectra/auto_berv_and_mask.sh",
                ],
    'name': 'phd-codes',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    "classifiers": [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
    ],
    # What does your project relate to?
    "keywords": ['Astronomy', 'Telluric Correction', "Wavelength Calibration"],
}

setup(**config)

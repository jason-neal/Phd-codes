
# Module for loading in the combined norm fits file created by combine_nod_specta.py
# This is to write some ideas for inporting the combined fits data from file.

path = '/home/Phd/data/Crires/BDs-DRACS/HD30501-1/normobs/'
   
 def get_all_norm_spectra(file):
    """ Get normalized/combined spectra"""
    Data = fits.getdata(ThisFile,1)
    Combined = data["Combined"]   
	NodA = data["Nod A"]
	NodB = data["Nod B"]
	return Combined, NodA, NodB       


Combined, NodA, NodB = get_all_norm_spectra(file)

print(Combined)
print(Combined["Combined"])
print(Combined["Nod A"])

Data = fits.getdata(ThisFile,1)
print(Data)
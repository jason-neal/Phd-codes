#

""" Codes for Telluric contamination removal"""

def DivideSpec(SpecA, SpecB):
""" Divide two spectra"""
    assert(len(SpecA) == len(SpecB)), "Not same lenght"
    Divide = SpecA/SpecB
    return None



def match_Wls(Wl, Spec , refWl):
    """Interpolate Wavelengths of spectra to common WL
    Most likely convert telluric to observed spectra wl after wl mapping performed"""
    NewSpec = None
    return NewSpec
 
def plotrecovery():
   """ place for some plotting code"""
    pass

def TelluricCorrect(WLObs,SpecObs,WLTel,SpecTel):
    """Code to contain other functions in this file

 1. Interpolate spectra to same wavelengths with match_WLs()
 2. Divide by Telluric
3.   ... """
    #
    #  WlTel, SpecTell = match_Wls(WLTel,SpecTel,WLObs)
    # 
    WL = None
    Spec = None
    return WL, Spec

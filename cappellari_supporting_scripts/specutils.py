# Utilities for handing spectra
# Bryan Miller

import pyfits
import numpy as np
from pysynphot import observation
from pysynphot import spectrum

def wavehead(hdr,axis=1):
    # Calculates the wavelength array from a FITS header
    # Based on http://mail.scipy.org/pipermail/astropy/2013-July/002635.html
    try:
        cdelt = hdr['CDELT'+str(axis)]
    except:
        cdelt = hdr['CD'+str(axis)+'_'+str(axis)]
    crval = hdr['CRVAL'+str(axis)]
    crpix = hdr['CRPIX'+str(axis)]
    npix = hdr['NAXIS'+str(axis)]
    dcflag = hdr['DC-FLAG']
    print cdelt,crval,crpix,npix
    #start = crval - (1. - crpix)* cdelt
    #end = start + cdelt * npix - cdelt/10.
    w = np.zeros(npix)
    if dcflag == 1:
        w = [10**(crval + ((i+1.) - crpix)*cdelt) for i in range(npix)]
    else:
        w = [crval + ((i+1.) - crpix)*cdelt for i in range(npix)]
    #print start,end
    #w = np.arange(start, end, cdelt)
    return np.asarray(w)

def readspec(fitsfile):
    # Baed on http://mail.scipy.org/pipermail/astropy/2013-July/002635.html
    # Returns wavelength and intensity arrays, assumes simple FITS files
    f = pyfits.open(fitsfile)
    w = wavehead(f[0].header)
    intensity = f[0].data
    f.close()
    return w, intensity

def rebin_spec(wave, specin, wavnew):
    # From http://www.astrobetter.com/blog/2013/08/12/python-tip-re-sampling-spectra-with-pysynphot/
    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')

    return obs.binflux

#----------------------------------------------------------------------------



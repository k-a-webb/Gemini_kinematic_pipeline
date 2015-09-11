__author__ = 'kwebb'

"""
This script follows the examples for the use of pPXF written by M. Cappellari available with the pPXF python download. 
Most of the commented lines pertain to the commands used in the example scripts for each task. 
The gas kinematic task may be implemented with parallel processing if using a large SSP library.
"""

from ppxf import ppxf
import ppxf_util as util
import numpy as np
import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
#from joblib import Parallel, delayed
from scipy import ndimage, signal
from time import clock

def rebin(x, factor):
    """
    Rebin a one-dimensional vector by averaging
    in groups of "factor" adjacent values

    """
    return np.mean(x.reshape(-1, factor), axis=1)


def ppxf_kinematics(bin_sci, ppxf_file, ppxf_bestfit, template_fits, FWHM_tem, FWHM_gal, vel_init=1500., sig_init=100.,
                    bias=0., plot=False, quiet=True, badPixels=[], clean=False):
    """
    Follow the pPXF usage example by Michile Cappellari
    INPUT: DIR_SCI_COMB (comb_fits_sci_{S/N}/bin_sci_{S/N}.fits), TEMPLATE_* (spectra/Mun1.30z*.fits)
    OUTPUT: PPXF_FILE (ppxf_output_sn30.txt), OUT_BESTFIT_FITS (ppxf_fits/ppxf_bestfit_sn30.fits)
    """

    if os.path.exists(ppxf_file):
        print('File {} already exists'.format(ppxf_file))
        return

    # Read a galaxy spectrum and define the wavelength range
    assert len(glob.glob(bin_sci.format('*'))) > 0, 'Binned spectra {} not found'.format(glob.glob(bin_sci.format('*')))

    # hdu = fits.open(in_file[0])
    # gal_lin = hdu[0].data
    # h1 = hdu[0].header

    with fits.open(bin_sci.format(0)) as gal_hdu:
        gal_lin = gal_hdu[0].data
        hdr = gal_hdu[0].header

    # lamRange1 = h1['CRVAL1'] + np.array([0.,h1['CDELT1']*(h1['NAXIS1']-1)])
    lam_range = hdr['CRVAL1'] + np.array([1. - hdr['CRPIX1'], hdr['NAXIS1'] - hdr['CRPIX1']]) * hdr['CD1_1']
    # FWHM_gal = 4.2 # SAURON has an instrumental resolution FWHM of 4.2A.

    # lamRange1 is now variable lam_range (because I was lazy and didnt put headers into the binned spectra)
    # FWHM_gal = 4.6  # GMOS IFU has an instrumental resolution FWHM of 2.3 A

    # If the galaxy is at a significant redshift (z > 0.03), one would need to apply a large velocity shift in PPXF to
    # match the template to the galaxy spectrum.This would require a large initial value for the velocity (V > 1e4 km/s)
    # in the input parameter START = [V,sig]. This can cause PPXF to stop! The solution consists of bringing the galaxy
    # spectrum roughly to the rest-frame wavelength, before calling PPXF. In practice there is no need to modify the
    # spectrum before the usual LOG_REBIN, given that a red shift corresponds to a linear shift of the log-rebinned
    # spectrum. One just needs to compute the wavelength range in the rest-frame and adjust the instrumental resolution
    # of the galaxy observations. This is done with the following three commented lines:
    #
    # z = 1.23 # Initial estimate of the galaxy redshift
    # lamRange1 = lamRange1/(1+z) # Compute approximate restframe wavelength range
    # FWHM_gal = FWHM_gal/(1+z)   # Adjust resolution in Angstrom

    # galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_lin)
    # galaxy = galaxy/np.median(galaxy) # Normalize spectrum to avoid numerical issues
    # noise = galaxy*0 + 0.0049           # Assume constant noise per pixel here

    galaxy, logLam1, velscale = util.log_rebin(lam_range, gal_lin)
    galaxy = galaxy / np.median(galaxy)  # Normalize spectrum to avoid numerical issues

    # Read the list of filenames from the Single Stellar Population library by Vazdekis (1999, ApJ, 513, 224). A subset
    # of the library is included for this example with permission. See http://purl.org/cappellari/software
    # for suggestions of more up-to-date stellar libraries.

    # vazdekis = glob.glob(dir + 'Rbi1.30z*.fits')
    # vazdekis.sort()
    # FWHM_tem = 1.8 # Vazdekis spectra have a resolution FWHM of 1.8A.

    template_spectra = glob.glob(template_fits)
    assert len(template_spectra) > 0, 'Template spectra not found: {}'.format(template_fits)

    # Extract the wavelength range and logarithmically rebin one spectrum to the same velocity scale of the SAURON
    # galaxy spectrum, to determine the size needed for the array which will contain the template spectra.

    # hdu = fits.open(vazdekis[0])
    # ssp = hdu[0].data
    # h2 = hdu[0].header
    # lamRange2 = h2['CRVAL1'] + np.array([0.,h2['CDELT1']*(h2['NAXIS1']-1)])
    # sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
    # templates = np.empty((sspNew.size,len(vazdekis)))

    with fits.open(template_spectra[0]) as temp_hdu:
        ssp = temp_hdu[0].data
        h2 = temp_hdu[0].header
    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1'] * (h2['NAXIS1'] - 1)])
    sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
    templates = np.empty((sspNew.size, len(template_spectra)))

    # Convolve the whole Vazdekis library of spectral templates with the quadratic difference between the SAURON and the
    # Vazdekis instrumental resolution. Logarithmically rebin and store each template as a column in the array TEMPLATES

    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the instrumental spectral profiles are well approximated
    # by Gaussians.

    # FROM GWENS IDL SCRIPT ----------------------------------------------------------
    # Example: BC03 spectra have a resolution of 3A at 5100A, this corresponds to sigma ~ 3.0/5100*3e5/2.35 = 75 km/s.
    # The GMOS IFU has an instrumental resolution of 2.3 A, this corresponds to sigma ~ 2.3/5100*3e5/2.35 = 56.8 km/s.
    # The quadratic difference is sigma = sqrt(56.8^2 - 75^2) = undefined
    # (the above reasoning can be applied if the shape of the instrumental spectral profiles can be well approximated
    # by a Gaussian).
    # For the lower resolution models, we must degrade the DATA to fit the models.
    # Thus: The quadratic difference is sigma = sqrt(75^2 - 56.8^2) = 49.0 km/s
    # ---------------------------------------------------------------------------------

    if FWHM_gal > FWHM_tem:
        FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
    else:
        FWHM_dif = np.sqrt(FWHM_tem ** 2 - FWHM_gal ** 2)
    sigma = FWHM_dif / 2.355 / h2['CDELT1']  # SIGMA DIFFERENCE IN PIXELS, 1.078435697220085

    # Logarithmically rebin the whole Mun library of spectra, and store each template as a column in the array TEMPLATES

    # for j in range(len(vazdekis)):
    # hdu = fits.open(vazdekis[j])
    # ssp = hdu[0].data
    # ssp = ndimage.gaussian_filter1d(ssp, sigma)
    # sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
    # templates[:, j] = sspNew / np.median(sspNew)  # Normalizes templates

    for j in range(len(template_spectra)):
        with fits.open(template_spectra[j]) as temp_hdu_j:
            ssp_j = temp_hdu_j[0].data
        ssp_j = ndimage.gaussian_filter1d(ssp_j, sigma)
        sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp_j, velscale=velscale)

        templates[:, j] = sspNew / np.median(sspNew)  # Normalizes templates

    # The galaxy and the template spectra do not have the same starting wavelength. For this reason an extra velocity
    # shift DV has to be applied to the template to fit the galaxy spectrum. We remove this artificial shift by using
    # the keyword VSYST in the call to PPXF below, so that all velocities are measured with respect to DV. This assume
    # the redshift is negligible.In the case of a high-redshift galaxy one should de-redshift its wavelength to the
    # rest frame before using the line below (see above).

    vel_list = []
    sig_list = []
    dV_list = []
    dsigma_list = []

    h3_list = []
    h4_list = []
    dh3_list = []
    dh4_list = []

    chi2 = []

    for j in range(len(glob.glob(bin_sci.format('*')))):
        b_gal = fits.getdata(bin_sci.format(j), 0)

        b_gal = ndimage.gaussian_filter1d(b_gal, sigma)

        galaxy, logLam1, velscale = util.log_rebin(lam_range, b_gal, velscale=velscale)
        noise = galaxy * 0 + 1  # Assume constant noise per pixel here

        c = 299792.458
        dv = (logLam2[0] - logLam1[0]) * c  # km/s

        # vel = 1500.  # Initial estimate of the galaxy velocity in km/s
        z = np.exp(vel_init / c) - 1  # Relation between velocity and redshift in pPXF

        goodPixels = util.determine_goodpixels(logLam1, lamRange2, z)
        if len(badPixels) > 0:
            indices = []
            for idx, pix in enumerate(goodPixels):
                if pix in badPixels:
                    indices.append(idx)
            goodPixels = np.delete(goodPixels, indices)

        # Here the actual fit starts. The best fit is plotted on the screen.
        # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.

        start = [vel_init, sig_init]  # (km/s), starting guess for [V,sigma]
        t = clock()

        pp = ppxf(templates, galaxy, noise, velscale, start, goodpixels=goodPixels, plot=plot, moments=4,
                  degree=4, vsyst=dv, bias=bias, quiet=False, clean=clean)
        if plot:
            plt.show()


        if not quiet:
            print("Formal errors:")
            print("     dV    dsigma   dh3      dh4")
            print("".join("%8.2g" % f for f in pp.error * np.sqrt(pp.chi2)))

        # If the galaxy is at significant redshift z and the wavelength has been de-redshifted with the three lines
        # "z = 1.23..." near the beginning of this procedure, the best-fitting redshift is now given by the following
        # commented line (equation 2 of Cappellari et al. 2009, ApJ, 704, L34;
        # http://adsabs.harvard.edu/abs/2009ApJ...704L..34C)
        # print, 'Best-fitting redshift z:', (z + 1)*(1 + sol[0]/c) - 1

        # Gwen obtains the velocity and sigma information from the SOL parameter
        # moments = 4 so sol = [vel, sig, h3, h4]
        vel_list.append(pp.sol[0])
        sig_list.append(pp.sol[1])
        dV_list.append((pp.error * np.sqrt(pp.chi2))[0])
        dsigma_list.append((pp.error * np.sqrt(pp.chi2))[1])

        h3_list.append(pp.sol[2])
        h4_list.append(pp.sol[3])
        dh3_list.append((pp.error * np.sqrt(pp.chi2))[2])
        dh4_list.append((pp.error * np.sqrt(pp.chi2))[3])

        chi2.append(pp.chi2)

        hdu_best = fits.PrimaryHDU()
        hdu_best.data = pp.bestfit
        hdu_best.header['CD1_1'] = hdr['CD1_1']
        hdu_best.header['CDELT1'] = hdr['CD1_1']
        hdu_best.header['CRPIX1'] = hdr['CRPIX1']
        hdu_best.header['CRVAL1'] = hdr['CRVAL1']
        hdu_best.header['NAXIS1'] = pp.bestfit.size
        hdu_best.header['CTYPE1'] = 'LINEAR'  # corresponds to sampling of values above
        hdu_best.header['DC-FLAG'] = '1'  # 0 = linear, 1= log-linear sampling
        hdu_best.writeto(ppxf_bestfit.format(j), clobber=True)

    print('Elapsed time in PPXF: %.2f s' % (clock() - t))

    np.savetxt(ppxf_file,
               np.column_stack([vel_list, sig_list, h3_list, h4_list, dV_list, dsigma_list, dh3_list, dh4_list, chi2]),
               fmt=b'%10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f',
               header='velocity    sigma       h3           h4         dV          dsigma       dh3         dh4  chi2')

    return vel_list


def ppxf_simulation(ppxf_bestfit, target_sn, bias=0.6, spaxel=0):
    """
    2. Perform a fit of your kinematics *without* penalty (PPXF keyword BIAS=0).
       The solution will be noisy and may be affected by spurious solutions,
       however this step will allow you to check the expected mean ranges in
       the Gauss-Hermite parameters [h3,h4] for the galaxy under study;

        see ppxf_output_sn30_bias0.txt
        mean(h3), std(h3), max(h3) = -0.2555, 0.09090, 0.007468
         # ignoring endpoints, max(h3[1:-1]) = -0.01376
        mean(h4), std(h4), max(h4) = -0.07712, 0.1423, 0.136607
         # ignoring endpoints, max(h4[1,-1]) = 0.13594
        max(dvel), min(dvel), max(dvel)-np.min(dvel) = 119.2918, 4.08643, 115.20544
        mean(vel) = 1543.0359
        max(sig), min(sig) = 180.3, 36.23



    3. Perform a Monte Carlo simulation of your spectra, following e.g. the
       included ppxf_simulation_example.pro routine. Adopt as S/N in the simulation
       the chosen value (S/N)_min and as input [h3,h4] the maximum representative
       values measured in the non-penalized pPXF fit of the previous step;

    4. Choose as penalty (BIAS) the *largest* value such that, for sigma > 3*velScale,
       the mean difference between the output [h3,h4] and the input [h3,h4]
       is well within the rms scatter of the simulated values
       (see e.g. Fig.2 of Emsellem et al. 2004, MNRAS, 352, 721).
    """

    # dir = 'spectra/'
    # file = dir + 'Rbi1.30z+0.00t12.59.fits'
    # hdu = pyfits.open(file)
    # ssp = hdu[0].data
    # h = hdu[0].header

    bestfit_file = ppxf_bestfit.format(spaxel)
    assert os.path.exists(bestfit_file), 'Best fit spectra not found: {}'.format(bestfit_file)

    with fits.open(bestfit_file) as best_hdu:
        ssp = best_hdu[0].data
        hdr = best_hdu[0].header

    # lamRange = h['CRVAL1'] + np.array([0., h['CDELT1'] * (h['NAXIS1'] - 1)])
    lam_range = hdr['CRVAL1'] + np.array([1. - hdr['CRPIX1'], hdr['NAXIS1'] - hdr['CRPIX1']]) * hdr['CD1_1']
    # star, logLam, velscale = util.log_rebin(lamRange, ssp)

    star, logLam, velscale = util.log_rebin(lam_range, ssp)

    # The finite sampling of the observed spectrum is modeled in detail: the galaxy spectrum is obtained by oversampling
    # the actual observed spectrum to a high resolution. This represents the true spectrum, which is later resampled
    # to lower resolution to simulate the observations on the CCD. Similarly, the convolution with a well-sampled LOSVD
    # is done on the high-resolution spectrum, and later resampled to the observed resolution before fitting with PPXF.

    factor = 10  # Oversampling integer factor for an accurate convolution
    starNew = ndimage.interpolation.zoom(star, factor, order=1)  # The underlying spectrum, known at high resolution
    star = rebin(starNew, factor)  # Make sure that the observed spectrum is the integral over the pixels

    # vel = 0.3  # velocity in *pixels* [=V(km/s)/velScale]
    # h3 = 0.1  # Adopted G-H parameters of the LOSVD
    # h4 = -0.1
    # sn = 60.  # Adopted S/N of the Monte Carlo simulation
    # m = 300  # Number of realizations of the simulation
    # sigmaV = np.linspace(0.8, 4, m)  # Range of sigma in *pixels* [=sigma(km/s)/velScale]

    print('bestfit {} : velscale = {}'.format(spaxel, velscale))
    # bestfit 0 : velscale = 57.44245804
    # max(sig), min(sig) = 164.026343, 11.306016 [km/s] =  2.87765514,  0.19835116 [pix]

    vel = 0.3  # mean vel = 1556.4989, I have no idea why this value is 0.3 ...
    h3 = -0.003146
    h4 = -0.003662
    sn = target_sn
    m = 300  # Number of realizations of the simulation
    sigmaV = np.linspace(0.8, 4, m)  # Range of sigma in *pixels* [=sigma(km/s)/velScale]
    # sigmaV = np.linspace(0.198, 2.877, m)

    result = np.zeros((m, 4))  # This will store the results
    t = clock()
    np.random.seed(123)  # for reproducible results

    for j in range(m):
        sigma = sigmaV[j]
        dx = int(abs(vel) + 4.0 * sigma)  # Sample the Gaussian and GH at least to vel+4*sigma
        x = np.linspace(-dx, dx, 2 * dx * factor + 1)  # Evaluate the Gaussian using steps of 1/factor pixels.
        w = (x - vel) / sigma
        w2 = w ** 2
        gauss = np.exp(-0.5 * w2) / (np.sqrt(2. * np.pi) * sigma * factor)  # Normalized total(gauss)=1
        h3poly = w * (2. * w2 - 3.) / np.sqrt(3.)  # H3(y)
        h4poly = (w2 * (4. * w2 - 12.) + 3.) / np.sqrt(24.)  # H4(y)
        losvd = gauss * (1. + h3 * h3poly + h4 * h4poly)

        galaxy = signal.fftconvolve(starNew, losvd, mode="same")  # Convolve the oversampled spectrum
        galaxy = rebin(galaxy, factor)  # Integrate spectrum into original spectral pixels
        noise = galaxy / sn  # 1sigma error spectrum
        galaxy = np.random.normal(galaxy, noise)  # Add noise to the galaxy spectrum
        start = np.array(
            [vel + np.random.random(), sigma * np.random.uniform(0.85, 1.15)]) * velscale  # Convert to km/s

        pp = ppxf(star, galaxy, noise, velscale, start, goodpixels=np.arange(dx, galaxy.size - dx),
                  plot=False, moments=4, bias=bias, quiet=True)
        result[j, :] = pp.sol

    print('Calculation time: %.2f s' % (clock() - t))

    plt.clf()
    plt.subplot(221)
    plt.plot(sigmaV * velscale, result[:, 0] - vel * velscale, '+k')
    plt.plot(sigmaV * velscale, np.ones(len(sigmaV * velscale)) * np.mean(result[:, 0] - vel * velscale), '-b')
    plt.plot(sigmaV * velscale, sigmaV * velscale * 0, '-r')
    plt.ylim(-40, 40)
    plt.xlabel('$\sigma_{in}\ (km\ s^{-1})$')
    plt.ylabel('$V - V_{in}\ (km\ s^{-1}$)')

    plt.subplot(222)
    plt.plot(sigmaV * velscale, result[:, 1] - sigmaV * velscale, '+k')
    plt.plot(sigmaV * velscale, np.ones(len(sigmaV * velscale)) * np.mean(result[:, 1] - sigmaV * velscale), '-b')
    plt.plot(sigmaV * velscale, sigmaV * velscale * 0, '-r')
    plt.ylim(-40, 40)
    plt.xlabel('$\sigma_{in}\ (km\ s^{-1})$')
    plt.ylabel('$\sigma - \sigma_{in}\ (km\ s^{-1}$)')

    plt.subplot(223)
    plt.plot(sigmaV * velscale, result[:, 2], '+k')
    plt.plot(sigmaV * velscale, sigmaV * velscale * 0 + h3, '-r')
    plt.plot(sigmaV * velscale, np.ones(len(sigmaV * velscale)) * np.mean(result[:, 2]), '-b')
    plt.ylim(-0.2 + h3, 0.2 + h3)
    plt.xlabel('$\sigma_{in}\ (km\ s^{-1})$')
    plt.ylabel('$h_3$')

    plt.subplot(224)
    plt.plot(sigmaV * velscale, result[:, 3], '+k')
    plt.plot(sigmaV * velscale, sigmaV * velscale * 0 + h4, '-r')
    plt.plot(sigmaV * velscale, np.ones(len(sigmaV * velscale)) * np.mean(result[:, 3]), '-b')
    plt.ylim(-0.2 + h4, 0.2 + h4)
    plt.xlabel('$\sigma_{in}\ (km\ s^{-1})$')
    plt.ylabel('$h_4$')

    plt.tight_layout()
    plt.show()


def ppxf_kinematics_gas(bin_sci, ppxf_file, ppxf_bestfit, template_fits, FWHM_tem,
                        vel_init=1500., sig_init=100., bias=0.6, plot=False, quiet=True, FWHM=2.1):
    """
    Follow the pPXF usage example by Michile Cappellari
    INPUT: DIR_SCI_COMB (comb_fits_sci_{S/N}/bin_sci_{S/N}.fits), TEMPLATE_* (spectra/Mun1.30z*.fits)
    OUTPUT: PPXF_FILE (ppxf_output_sn30.txt), OUT_BESTFIT_FITS (ppxf_fits/ppxf_bestfit_sn30.fits)
    """

    if os.path.exists(ppxf_file):
        print('File {} already exists'.format(ppxf_file))
        return

    # Read in the galaxy spectrum, logarithmically rebin
    #
    assert len(glob.glob(bin_sci.format('*'))) > 0, 'Binned spectra {} not found'.format(glob.glob(bin_sci.format('*')))
    with fits.open(bin_sci.format(0)) as gal_hdu:
        gal_data = gal_hdu[0].data
        gal_hdr = gal_hdu[0].header
    lamRange1 = gal_hdr['CRVAL1'] + np.array([1. - gal_hdr['CRPIX1'], gal_hdr['NAXIS1'] - gal_hdr['CRPIX1']]) \
                                    * gal_hdr['CD1_1']

    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_data)
    galaxy = galaxy / np.median(galaxy)  # Normalize spectrum to avoid numerical issues
    wave = np.exp(logLam1)

    # The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0).
    # A constant error is not a bad approximation in the fitted wavelength
    # range and reduces the noise in the fit.
    #
    noise = galaxy * 0 + 0.01528  # Assume constant noise per pixel here

    c = 299792.458  # speed of light in km/s
    # stars_templates, lamRange_temp, logLam_temp = setup_spectral_library(velscale, FWHM_gal)

    # Read in template galaxy spectra
    #
    template_spectra = glob.glob(template_fits)
    assert len(template_spectra) > 0, 'Template spectra not found: {}'.format(template_fits)
    with fits.open(template_spectra[0]) as temp_hdu:
        ssp = temp_hdu[0].data
        h2 = temp_hdu[0].header
    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1'] * (h2['NAXIS1'] - 1)])
    sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
    stars_templates = np.empty((sspNew.size, len(template_spectra)))

    if FWHM_gal > FWHM_tem:
        FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
    else:
        FWHM_dif = np.sqrt(FWHM_tem ** 2 - FWHM_gal ** 2)
    sigma = FWHM_dif / 2.355 / h2['CDELT1']  # SIGMA DIFFERENCE IN PIXELS, 1.078435697220085

    for j in range(len(template_spectra)):
        with fits.open(template_spectra[j]) as temp_hdu_j:
            ssp_j = temp_hdu_j[0].data
        ssp_j = ndimage.gaussian_filter1d(ssp_j, sigma)
        sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp_j, velscale=velscale)
        stars_templates[:, j] = sspNew / np.median(sspNew)  # Normalizes templates

    # we save the original array dimensions, which are needed to specify the regularization dimensions
    #
    reg_dim = stars_templates.shape[1:]

    # See the pPXF documentation for the keyword REGUL,
    # for an explanation of the following two lines
    #
    regul_err = 0.004  # Desired regularization error

    # Construct a set of Gaussian emission line templates.
    # Estimate the wavelength fitted range in the rest frame.
    #
    z = np.exp(vel_init / c) - 1  # Relation between velocity and redshift in pPXF
    lamRange_gal = np.array([lamRange1[0], lamRange1[-1]]) / (1 + z)
    gas_templates, line_names, line_wave = util.emission_lines(logLam2, lamRange_gal, FWHM_gal)

    # Combines the stellar and gaseous templates into a single array
    # during the PPXF fit they will be assigned a different kinematic COMPONENT value
    #
    templates = np.hstack([stars_templates, gas_templates])


    # Here the actual fit starts. The best fit is plotted on the screen.
    #
    # IMPORTANT: Ideally one would like not to use any polynomial in the fit
    # as the continuum shape contains important information on the population.
    # Unfortunately this is often not feasible, due to small calibration
    # uncertainties in the spectral shape. To avoid affecting the line strength of
    # the spectral features, we exclude additive polynomials (DEGREE=-1) and only use
    # multiplicative ones (MDEGREE=10). This is only recommended for population, not
    # for kinematic extraction, where additive polynomials are always recommended.
    #
    # start = [vel, 180.]  # (km/s), starting guess for [V,sigma]
    start = [vel_init, sig_init]  # (km/s), starting guess for [V,sigma]

    # Assign component=0 to the stellar templates and
    # component=1 to the gas emission lines templates.
    # One can easily assign different kinematic components to different gas species
    # e.g. component=1 for the Balmer series, component=2 for the [OIII] doublet, ...)
    # Input a negative MOMENTS value to keep fixed the LOSVD of a component.
    #
    nTemps = stars_templates.shape[1]
    nLines = gas_templates.shape[1]
    component = [0] * nTemps + [1] * nLines
    moments = [4, 2]  # fit (V,sig,h3,h4) for the stars and (V,sig) for the gas
    start = [start, start]  # adopt the same starting value for both gas and stars

    bin_sci_list = glob.glob(bin_sci.format('*'))

    pp_sol = np.zeros([moments[0] + moments[1], len(bin_sci_list)])
    pp_error = np.zeros([moments[0] + moments[1], len(bin_sci_list)])

    for j in range(len(bin_sci_list)):

        gal_data = fits.getdata(bin_sci.format(j), 0)
        gal_data_new = ndimage.gaussian_filter1d(gal_data, sigma)

        galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_data_new, velscale=velscale)
        noise = galaxy * 0 + 1  # Assume constant noise per pixel here

        # The galaxy and the template spectra do not have the same starting wavelength.
        # For this reason an extra velocity shift DV has to be applied to the template
        # to fit the galaxy spectrum. We remove this artificial shift by using the
        # keyword VSYST in the call to PPXF below, so that all velocities are
        # measured with respect to DV. This assume the redshift is negligible.
        # In the case of a high-redshift galaxy one should de-redshift its
        # wavelength to the rest frame before using the line below as described
        # in PPXF_KINEMATICS_EXAMPLE_SAURON.
        #
        c = 299792.458
        dv = (logLam2[0] - logLam1[0]) * c  # km/s

        t = clock()

        pp = ppxf(templates, galaxy, noise, velscale, start, plot=plot, moments=moments, degree=-1, mdegree=10,
                  vsyst=dv, clean=False, regul=1. / regul_err, reg_dim=reg_dim, component=component, bias=bias,
                  quiet=quiet)

        # Save the velocity, sigma, h3, h4 information for both stellar and gas to a table
        for k, sol in enumerate(pp.sol[0]):
            pp_sol[k, j] = pp.sol[0][k]
            pp_error[k, j] = pp.error[0][k]
        for k, sol in enumerate(pp.sol[1]):
            pp_sol[k + len(pp.sol[0]), j] = pp.sol[1][k]
            pp_error[k + len(pp.error[0]), j] = pp.error[1][k]

        # Plot fit results for stars and gas
        #
        if plot:
            plt.clf()
            # plt.subplot(211)
            plt.plot(wave, pp.galaxy, 'k')
            plt.plot(wave, pp.bestfit, 'b', linewidth=2)
            plt.xlabel("Observed Wavelength ($\AA$)")
            plt.ylabel("Relative Flux")
            plt.ylim([-0.1, 1.3])
            plt.xlim([np.min(wave), np.max(wave)])
            plt.plot(wave, pp.galaxy - pp.bestfit, 'd', ms=4, color='LimeGreen', mec='LimeGreen')  # fit residuals
            plt.axhline(y=-0, linestyle='--', color='k', linewidth=2)
            stars = pp.matrix[:, :nTemps].dot(pp.weights[:nTemps])
            plt.plot(wave, stars, 'r', linewidth=2)  # overplot stellar templates alone
            gas = pp.matrix[:, -nLines:].dot(pp.weights[-nLines:])
            plt.plot(wave, gas + 0.15, 'b', linewidth=2)  # overplot emission lines alone
            plt.legend()
            plt.show()

            # When the two Delta Chi^2 below are the same, the solution is the smoothest
            # consistent with the observed spectrum.
            #
            print('Desired Delta Chi^2: %.4g' % np.sqrt(2 * galaxy.size))
            print('Current Delta Chi^2: %.4g' % ((pp.chi2 - 1) * galaxy.size))
            print('Elapsed time in PPXF: %.2f s' % (clock() - t))

            w = np.where(np.array(component) == 1)[0]  # Extract weights of gas emissions
            print('++++++++++++++++++++++++++++++')
            print('Gas V=%.4g and sigma=%.2g km/s' % (pp.sol[1][0], pp.sol[1][1]))
            print('Emission lines peak intensity:')
            for name, weight, line in zip(line_names, pp.weights[w], pp.matrix[:, w].T):
                print('%12s: %.3g' % (name, weight * np.max(line)))
            print('------------------------------')

            # Plot stellar population mass distribution

            # plt.subplot(212)
            # weights = pp.weights[:np.prod(reg_dim)].reshape(reg_dim) / pp.weights.sum()
            # plt.imshow(np.rot90(weights), interpolation='nearest', cmap='gist_heat', aspect='auto', origin='upper',
            # extent=(np.log10(1.0), np.log10(17.7828), -1.9, 0.45))
            # plt.colorbar()
            # plt.title("Mass Fraction")
            # plt.xlabel("log$_{10}$ Age (Gyr)")
            # plt.ylabel("[M/H]")
            # plt.tight_layout()

            plt.legend()
            plt.show()

        if not quiet:
            print("Formal errors:")
            print("     dV    dsigma   dh3      dh4")
            print("".join("%8.2g" % f for f in pp.error * np.sqrt(pp.chi2)))

        hdu_best = fits.PrimaryHDU()
        hdu_best.data = pp.bestfit
        hdu_best.header['CD1_1'] = gal_hdr['CD1_1']
        hdu_best.header['CDELT1'] = gal_hdr['CD1_1']
        hdu_best.header['CRPIX1'] = gal_hdr['CRPIX1']
        hdu_best.header['CRVAL1'] = gal_hdr['CRVAL1']
        hdu_best.header['NAXIS1'] = pp.bestfit.size
        hdu_best.header['CTYPE1'] = 'LINEAR'  # corresponds to sampling of values above
        hdu_best.header['DC-FLAG'] = '1'  # 0 = linear, 1= log-linear sampling
        hdu_best.writeto(ppxf_bestfit.format(j), clobber=True)

    np.savetxt(ppxf_file,
               np.column_stack(
                   [pp_sol[0], pp_error[0], pp_sol[1], pp_error[1], pp_sol[2], pp_error[2],
                    pp_sol[3], pp_error[3], pp_sol[4], pp_error[4], pp_sol[5], pp_error[5]]),
               fmt=b'%10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f %10.4f  %10.4f  %10.4f  %10.4f',
               header='Stellar pop:                                          Gas: \n'
                      'vel  d_vel  sigma  d_sigma  h3  d_h3  h4  d_h4        vel  d_vel  sigma  d_sigma')


def ppxf_kinematics_gas_parallel(bin_sci, ppxf_file, ppxf_bestfit, template_fits, template_res,
                                 vel_init=1500., sig_init=100., bias=0.6, plot=False, quiet=True):
    """
    Follow the pPXF usage example by Michile Cappellari
    INPUT: DIR_SCI_COMB (comb_fits_sci_{S/N}/bin_sci_{S/N}.fits), TEMPLATE_* (spectra/Mun1.30z*.fits)
    OUTPUT: PPXF_FILE (ppxf_output_sn30.txt), OUT_BESTFIT_FITS (ppxf_fits/ppxf_bestfit_sn30.fits)
    """

    if os.path.exists(ppxf_file):
        print('File {} already exists'.format(ppxf_file))
        return

    # Read in the galaxy spectrum, logarithmically rebin
    #
    assert len(glob.glob(bin_sci.format('*'))) > 0, 'Binned spectra {} not found'.format(glob.glob(bin_sci.format('*')))
    with fits.open(bin_sci.format(0)) as gal_hdu:
        gal_data = gal_hdu[0].data
        gal_hdr = gal_hdu[0].header
    lamRange1 = gal_hdr['CRVAL1'] + np.array([1. - gal_hdr['CRPIX1'], gal_hdr['NAXIS1'] - gal_hdr['CRPIX1']]) \
                                    * gal_hdr['CD1_1']

    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_data)
    galaxy = galaxy / np.median(galaxy)  # Normalize spectrum to avoid numerical issues
    wave = np.exp(logLam1)

    # The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0).
    # A constant error is not a bad approximation in the fitted wavelength
    # range and reduces the noise in the fit.
    #
    noise = galaxy * 0 + 0.01528  # Assume constant noise per pixel here

    c = 299792.458  # speed of light in km/s
    FWHM_gal = 2.3  # GMOS IFU has an instrumental resolution FWHM of 2.3 A

    # stars_templates, lamRange_temp, logLam_temp = setup_spectral_library(velscale, FWHM_gal)

    # Read in template galaxy spectra
    #
    template_spectra = glob.glob(template_fits)
    assert len(template_spectra) > 0, 'Template spectra not found: {}'.format(template_fits)
    with fits.open(template_spectra[0]) as temp_hdu:
        ssp = temp_hdu[0].data
        h2 = temp_hdu[0].header
    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1'] * (h2['NAXIS1'] - 1)])
    sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)
    stars_templates = np.empty((sspNew.size, len(template_spectra)))
    FWHM_tem = template_res

    # FWHM_dif = np.sqrt(FWHM_gal ** 2 - template_res ** 2)
    FWHM_dif = np.sqrt(FWHM_tem ** 2 - FWHM_gal ** 2)
    sigma = FWHM_dif / 2.355 / h2['CDELT1']  # SIGMA DIFFERENCE IN PIXELS, 1.078435697220085

    for j in range(len(template_spectra)):
        with fits.open(template_spectra[j]) as temp_hdu_j:
            ssp_j = temp_hdu_j[0].data
        ssp_j = ndimage.gaussian_filter1d(ssp_j, sigma)
        sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp_j, velscale=velscale)
        stars_templates[:, j] = sspNew / np.median(sspNew)  # Normalizes templates

    # we save the original array dimensions, which are needed to specify the regularization dimensions
    #
    reg_dim = stars_templates.shape[1:]

    # See the pPXF documentation for the keyword REGUL,
    # for an explanation of the following two lines
    #
    regul_err = 0.004  # Desired regularization error

    # Construct a set of Gaussian emission line templates.
    # Estimate the wavelength fitted range in the rest frame.
    #
    z = np.exp(vel_init / c) - 1  # Relation between velocity and redshift in pPXF
    lamRange_gal = np.array([lamRange1[0], lamRange1[-1]]) / (1 + z)
    gas_templates, line_names, line_wave = util.emission_lines(logLam2, lamRange_gal, FWHM_gal)

    # Combines the stellar and gaseous templates into a single array
    # during the PPXF fit they will be assigned a different kinematic COMPONENT value
    #
    templates = np.hstack([stars_templates, gas_templates])


    # Here the actual fit starts. The best fit is plotted on the screen.
    #
    # IMPORTANT: Ideally one would like not to use any polynomial in the fit
    # as the continuum shape contains important information on the population.
    # Unfortunately this is often not feasible, due to small calibration
    # uncertainties in the spectral shape. To avoid affecting the line strength of
    # the spectral features, we exclude additive polynomials (DEGREE=-1) and only use
    # multiplicative ones (MDEGREE=10). This is only recommended for population, not
    # for kinematic extraction, where additive polynomials are always recommended.
    #
    # start = [vel, 180.]  # (km/s), starting guess for [V,sigma]
    start = [vel_init, sig_init]  # (km/s), starting guess for [V,sigma]

    # Assign component=0 to the stellar templates and
    # component=1 to the gas emission lines templates.
    # One can easily assign different kinematic components to different gas species
    # e.g. component=1 for the Balmer series, component=2 for the [OIII] doublet, ...)
    # Input a negative MOMENTS value to keep fixed the LOSVD of a component.
    #
    nTemps = stars_templates.shape[1]
    nLines = gas_templates.shape[1]
    component = [0] * nTemps + [1] * nLines
    moments = [4, 2]  # fit (V,sig,h3,h4) for the stars and (V,sig) for the gas
    start = [start, start]  # adopt the same starting value for both gas and stars

    bin_sci_list = glob.glob(bin_sci.format('*'))

    pp_sol = np.zeros([moments[0] + moments[1], len(bin_sci_list)])
    pp_error = np.zeros([moments[0] + moments[1], len(bin_sci_list)])

    pp = Parallel(n_jobs=len(bin_sci_list))( # up to 40 parallel processes, according to blair
        delayed(ppxf_gas_kinematics_parallel_loop)(bin_sci.format(j), ppxf_bestfit.format(j), gal_hdr, templates,
                                                   velscale, sigma, lamRange1, logLam2, start, plot, moments, regul_err,
                                                   reg_dim, component, bias, quiet) for j in range(len(bin_sci_list)))

    # amek sure this file exists beforehand
    with open(ppxf_file, 'a') as outfile:
        outfile.write(pp)


def ppxf_gas_kinematics_parallel_loop(bin_sci_j, ppxf_bestfit_j, gal_hdr, templates, velscale, sigma, lamRange1,
                                      logLam2, start, plot, moments, regul_err, reg_dim, component, bias, quiet):
    gal_data = fits.getdata(bin_sci_j, 0)
    gal_data_new = ndimage.gaussian_filter1d(gal_data, sigma)

    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_data_new, velscale=velscale)
    noise = galaxy * 0 + 1  # Assume constant noise per pixel here

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below as described
    # in PPXF_KINEMATICS_EXAMPLE_SAURON.
    #
    c = 299792.458
    dv = (logLam2[0] - logLam1[0]) * c  # km/s

    t = clock()

    pp = ppxf(templates, galaxy, noise, velscale, start, plot=plot, moments=moments, degree=-1, mdegree=10,
              vsyst=dv, clean=False, regul=1. / regul_err, reg_dim=reg_dim, component=component, bias=bias,
              quiet=quiet)

    print ('pPXF sol of {}: {}'.format(bin_sci_j, pp.sol))
    print ('pPXF error of {}: {}'.format(bin_sci_j, pp.error))

    hdu_best = fits.PrimaryHDU()
    hdu_best.data = pp.bestfit
    hdu_best.header['CD1_1'] = gal_hdr['CD1_1']
    hdu_best.header['CDELT1'] = gal_hdr['CD1_1']
    hdu_best.header['CRPIX1'] = gal_hdr['CRPIX1']
    hdu_best.header['CRVAL1'] = gal_hdr['CRVAL1']
    hdu_best.header['NAXIS1'] = pp.bestfit.size
    hdu_best.header['CTYPE1'] = 'LINEAR'  # corresponds to sampling of values above
    hdu_best.header['DC-FLAG'] = '1'  # 0 = linear, 1= log-linear sampling
    hdu_best.writeto(ppxf_bestfit_j, clobber=True)

    return pp


def setup_spectral_library(velscale, FWHM_gal, template_fits, FWHM_tem):
    # Read the list of filenames from the Single Stellar Population library
    # by Vazdekis et al. (2010, MNRAS, 404, 1639) http://miles.iac.es/.
    #
    # For this example I downloaded from the above website a set of
    # model spectra with default linear sampling of 0.9A/pix and default
    # spectral resolution of FWHM=2.51A. I selected a Salpeter IMF
    # (slope 1.30) and a range of population parameters:
    #
    #     [M/H] = [-1.71, -1.31, -0.71, -0.40, 0.00, 0.22]
    #     Age = range(1.0, 17.7828, 26, /LOG)
    #
    # This leads to a set of 156 model spectra with the file names like
    #
    #     Mun1.30Zm0.40T03.9811.fits
    #
    # IMPORTANT: the selected models form a rectangular grid in [M/H]
    # and Age: for each Age the spectra sample the same set of [M/H].
    #
    # We assume below that the model spectra have been placed in the
    # directory "miles_models" under the current directory.
    #
    vazdekis = glob.glob(template_fits)
    vazdekis.sort()
    # FWHM_tem = 2.51  # Vazdekis+10 spectra have a resolution FWHM of 2.51A.

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the SDSS galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    with fits.open(vazdekis[0]) as hdu:
        ssp = hdu[0].data
        h2 = hdu[0].header
    lamRange_temp = h2['CRVAL1'] + np.array([0., h2['CDELT1'] * (h2['NAXIS1'] - 1)])
    sspNew, logLam_temp, velscale = util.log_rebin(lamRange_temp, ssp, velscale=velscale)

    # Create a three dimensional array to store the
    # two dimensional grid of model spectra
    #
    nAges = 26
    nMetal = 6
    templates = np.empty((sspNew.size, nAges, nMetal))

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SDSS and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SDSS
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    if FWHM_gal > FWHM_tem:
        FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
    else:
        FWHM_dif = np.sqrt(FWHM_tem ** 2 - FWHM_gal ** 2)
    sigma = FWHM_dif / 2.355 / h2['CDELT1']  # Sigma difference in pixels

    # These are the array where we want to store
    # the characteristics of each SSP model
    #
    logAge_grid = np.empty((nAges, nMetal))
    metal_grid = np.empty((nAges, nMetal))

    # These are the characteristics of the adopted rectangular grid of SSP models
    #
    logAge = np.linspace(np.log10(1), np.log10(17.7828), nAges)
    metal = [-1.71, -1.31, -0.71, -0.40, 0.00, 0.22]

    # Here we make sure the spectra are sorted in both [M/H]
    # and Age along the two axes of the rectangular grid of templates.
    # A simple alphabetical ordering of Vazdekis's naming convention
    # does not sort the files by [M/H], so we do it explicitly below
    #
    metal_str = ['m1.71', 'm1.31', 'm0.71', 'm0.40', 'p0.00', 'p0.22']
    for k, mh in enumerate(metal_str):
        files = [s for s in vazdekis if mh in s]
        for j, filename in enumerate(files):
            with fits.open(filename) as hdu:
                ssp = hdu[0].data
            ssp = ndimage.gaussian_filter1d(ssp, sigma)
            sspNew, logLam_temp, velscale = util.log_rebin(lamRange_temp, ssp, velscale=velscale)
            templates[:, j, k] = sspNew  # Templates are *not* normalized here
            logAge_grid[j, k] = logAge[j]
            metal_grid[j, k] = metal[k]

    return templates, lamRange_temp, logLam_temp, logAge_grid, metal_grid


def ppxf_population_gas(bin_sci, FWHM_gal, template_fits, FWHM_tem, ppxf_file, ppxf_bestfit, vel_init=1500, bias=0.6, plot=False):
    """

    """

    if os.path.exists(ppxf_file):
        print('File {} already exists'.format(ppxf_file))
        return



    # Read a galaxy spectrum and define the wavelength range
    bin_sci_list = glob.glob(bin_sci.format('*'))
    assert len(bin_sci_list) > 0, 'Binned spectra {} not found'.format(glob.glob(bin_sci.format('*')))


    # Read SDSS DR8 galaxy spectrum taken from here http://www.sdss3.org/dr8/
    # The spectrum is *already* log rebinned by the SDSS DR8
    # pipeline and log_rebin should not be used in this case.
    #
    # file = 'spectra/NGC3522_SDSS.fits'
    # with fits.open(file) as hdu:
    #     t = hdu[1].data
    #     z = float(hdu[1].header["Z"])  # SDSS redshift estimate
    #
    with fits.open(bin_sci.format(0)) as hdu:
        gal_lin = hdu[0].data
        hdr = hdu[0].header

    # Only use the wavelength range in common between galaxy and stellar library.
    #
    # mask = (t.field('wavelength') > 3540) & (t.field('wavelength') < 7409)
    # galaxy = t[mask].field('flux') / np.median(t[mask].field('flux'))  # Normalize spectrum to avoid numerical issues
    # wave = t[mask].field('wavelength')
    #
    lam_range = hdr['CRVAL1'] + np.array([1. - hdr['CRPIX1'], hdr['NAXIS1'] - hdr['CRPIX1']]) * hdr['CD1_1']
    # log rebin the bins, not done here because SDSS alrady log binned, which is what this example uses
    galaxy, wave, velscale = util.log_rebin(lam_range, gal_lin)
    galaxy = galaxy/np.median(galaxy)
    # our data set does not exceed the range of the template so dont mask

    # The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0).
    # A constant error is not a bad approximation in the fitted wavelength
    # range and reduces the noise in the fit.
    #
    noise = galaxy * 0 + 0.01528  # Assume constant noise per pixel here

    # The velocity step was already chosen by the SDSS pipeline
    # and we convert it below to km/s
    #
    c = 299792.458  # speed of light in km/s
    # velscale = np.log(wave[1] / wave[0]) * c
    z = np.exp(vel_init / c) - 1
    # FWHM_gal = 2.76  # SDSS has an instrumental resolution FWHM of 2.76A.

    #------------------- Setup templates -----------------------

    stars_templates, lamRange_temp, logLam_temp, logAge_grid, metal_grid = setup_spectral_library(velscale, FWHM_gal, template_fits, FWHM_tem)

    # The stellar templates are reshaped into a 2-dim array with each spectrum
    # as a column, however we save the original array dimensions, which are
    # needed to specify the regularization dimensions
    #
    reg_dim = stars_templates.shape[1:]
    stars_templates = stars_templates.reshape(stars_templates.shape[0], -1)

    # See the pPXF documentation for the keyword REGUL,
    # for an explanation of the following two lines
    #
    stars_templates /= np.median(stars_templates)  # Normalizes stellar templates by a scalar
    regul_err = 0.004  # Desired regularization error

    # Construct a set of Gaussian emission line templates.
    # Estimate the wavelength fitted range in the rest frame.
    #
    # lamRange_gal = np.array([np.min(wave), np.max(wave)]) / (1 + z)
    # gas_templates, line_names, line_wave = util.emission_lines(logLam_temp, lamRange_gal, FWHM_gal)
    gas_templates, line_names, line_wave = util.emission_lines(logLam_temp, lam_range, FWHM_gal)

    # Combines the stellar and gaseous templates into a single array
    # during the PPXF fit they will be assigned a different kinematic
    # COMPONENT value
    #
    templates = np.hstack([stars_templates, gas_templates])

    #-----------------------------------------------------------

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below as described
    # in PPXF_KINEMATICS_EXAMPLE_SAURON.
    #
    c = 299792.458
    # dv = (np.log(lamRange_temp[0]) - np.log(wave[0])) * c  # km/s
    dv = (np.log(lamRange_temp[0]) - wave[0]) * c  # km/s
    # vel = c * z  # Initial estimate of the galaxy velocity in km/s
    vel = vel_init

    # Here the actual fit starts. The best fit is plotted on the screen.
    #
    # IMPORTANT: Ideally one would like not to use any polynomial in the fit
    # as the continuum shape contains important information on the population.
    # Unfortunately this is often not feasible, due to small calibration
    # uncertainties in the spectral shape. To avoid affecting the line strength of
    # the spectral features, we exclude additive polynomials (DEGREE=-1) and only use
    # multiplicative ones (MDEGREE=10). This is only recommended for population, not
    # for kinematic extraction, where additive polynomials are always recommended.
    #
    start = [vel, 180.]  # (km/s), starting guess for [V,sigma]

    t = clock()

    # Assign component=0 to the stellar templates and
    # component=1 to the gas emission lines templates.
    # One can easily assign different kinematic components to different gas species
    # e.g. component=1 for the Balmer series, component=2 for the [OIII] doublet, ...)
    # Input a negative MOMENTS value to keep fixed the LOSVD of a component.
    #
    nTemps = stars_templates.shape[1]
    nLines = gas_templates.shape[1]
    component = [0] * nTemps + [1] * nLines
    moments = [4, 2]  # fit (V,sig,h3,h4) for the stars and (V,sig) for the gas
    start = [start, start]  # adopt the same starting value for both gas and stars

    pp_sol = np.zeros([moments[0] + moments[1], len(bin_sci_list)])
    pp_error = np.zeros([moments[0] + moments[1], len(bin_sci_list)])

    for j in range(len(bin_sci_list)):

        gal_data = fits.getdata(bin_sci.format(j), 0)
        galaxy, logLam1, velscale = util.log_rebin(lam_range, gal_data, velscale=velscale)

        pp = ppxf(templates, galaxy, noise, velscale, start,
                  plot=plot, moments=moments, degree=-1, mdegree=10,
                  vsyst=dv, clean=False, regul=1. / regul_err,
                  reg_dim=reg_dim, component=component, bias=bias)

        # Save the velocity, sigma, h3, h4 information for both stellar and gas to a table
        for k, sol in enumerate(pp.sol[0]):
            pp_sol[k, j] = pp.sol[0][k]
            pp_error[k, j] = pp.error[0][k]
        for k, sol in enumerate(pp.sol[1]):
            pp_sol[k + len(pp.sol[0]), j] = pp.sol[1][k]
            pp_error[k + len(pp.error[0]), j] = pp.error[1][k]

        # Plot fit results for stars and gas

        if plot:
            plt.clf()
            plt.subplot(211)
            plt.plot(wave, pp.galaxy, 'k')
            plt.plot(wave, pp.bestfit, 'b', linewidth=2)
            plt.xlabel("Observed Wavelength ($\AA$)")
            plt.ylabel("Relative Flux")
            #plt.ylim([-0.1, 5])
            plt.xlim([np.min(wave), np.max(wave)])
            plt.plot(wave, pp.galaxy - pp.bestfit, 'd', ms=4, color='LimeGreen', mec='LimeGreen')  # fit residuals
            plt.axhline(y=-0, linestyle='--', color='k', linewidth=2)
            stars = pp.matrix[:, :nTemps].dot(pp.weights[:nTemps])
            plt.plot(wave, stars, 'r', linewidth=2)  # overplot stellar templates alone
            gas = pp.matrix[:, -nLines:].dot(pp.weights[-nLines:])
            plt.plot(wave, gas + 0.15, 'b', linewidth=2)  # overplot emission lines alone

            # When the two Delta Chi^2 below are the same, the solution is the smoothest
            # consistent with the observed spectrum.
            #
            print('Desired Delta Chi^2: %.4g' % np.sqrt(2 * galaxy.size))
            print('Current Delta Chi^2: %.4g' % ((pp.chi2 - 1) * galaxy.size))
            print('Elapsed time in PPXF: %.2f s' % (clock() - t))

            w = np.where(np.array(component) == 1)[0]  # Extract weights of gas emissions
            print('++++++++++++++++++++++++++++++')
            print('Gas V=%.4g and sigma=%.2g km/s' % (pp.sol[1][0], pp.sol[1][1]))
            print('Emission lines peak intensity:')
            for name, weight, line in zip(line_names, pp.weights[w], pp.matrix[:, w].T):
                print('%12s: %.3g' % (name, weight * np.max(line)))
            print('------------------------------')

            # Plot stellar population mass distribution

            weight_arr = pp.weights[:stars_templates.shape[1]]

            print('Mass-weighted <logAge> [Gyr]: %.3g' %
                  (np.sum(weight_arr*logAge_grid.ravel())/np.sum(weight_arr)))
            print('Mass-weighted <[M/H]>: %.3g' %
                  (np.sum(weight_arr*metal_grid.ravel())/np.sum(weight_arr)))

            plt.subplot(212)
            weights = pp.weights[:np.prod(reg_dim)].reshape(reg_dim) / pp.weights.sum()
            plt.imshow(np.rot90(weights), interpolation='nearest',
                       cmap='gist_heat', aspect='auto', origin='upper',
                       extent=(np.log10(1.0), np.log10(17.7828), -1.9, 0.45))
            plt.colorbar()
            plt.title("Mass Fraction")
            plt.xlabel("log$_{10}$ Age (Gyr)")
            plt.ylabel("[M/H]")
            plt.tight_layout()
            plt.show()

        np.savetxt(ppxf_bestfit.format(j), pp.bestfit)

    np.savetxt(ppxf_file,
               np.column_stack(
                   [pp_sol[0], pp_error[0], pp_sol[1], pp_error[1], pp_sol[2], pp_error[2],
                    pp_sol[3], pp_error[3], pp_sol[4], pp_error[4], pp_sol[5], pp_error[5]]),
               fmt=b'%10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f %10.4f  %10.4f  %10.4f  %10.4f',
               header='Stellar pop:                                          Gas: \n'
                      'vel  d_vel  sigma  d_sigma  h3  d_h3  h4  d_h4        vel  d_vel  sigma  d_sigma')
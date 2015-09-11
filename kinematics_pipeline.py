# written by Kristi Webb for data of IC225 taken by Bryan Miller
# similar to an earlier method developed by Gwen Rudie in IDL/cl/awk

from time import clock
import glob
import os
import numpy as np
from voronoi_2d_binning import voronoi_2d_binning
from scipy import ndimage, signal
from ppxf import ppxf
import ppxf_util as util
from matplotlib import pyplot as plt
from astropy.io import fits
import pandas as pd
from cap_plot_velfield import plot_velfield
import remove_lines
import bin_by_object
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ppxf_tasks


"""
This script covers all the kinematic analysis of a reduced 3D image cube by the method in reduce_and_cubel.cl for the
  example case of IC225

To visualize the 2D flattened cube for a given spectral range the steps are as follows:
    - Crop the 3D cube for a given spectral range
            scrop_cube(IMAGE_CUBE, SCROP_RANGE, CUBE_SCROPD)
    - Flatten the cropped 3D cubed
            imcombine_flatten(CUBE_SCROPD, SCI_EXT_SCROPD, VAR_EXT_SCROPD)
*** These flattened images are not used in the kinematic analysis, but are useful for visual confirmation of
    relative flux levels between the nucleui and surrounding gas

    The relavent nebular emission lines are:
        - OIII: 4940-5065 A - rest 4958.92, 5006.84
        - H_beta: 4865-4915 A - rest 4861.33
        - H_gamma: 4330-4380 A - rest 4340.47
    The continuum spectrum is:
        - continuum: 4383-4850 A

To run pPXF:
    - Compile coordinates of signal and noise measurements for the binning
            make_table(IMAGE_CUBE, SCI_EXT, VAR_EXT, XYSN_FILE)
    - Preform voronoi 2d binning to acheive desired signal to noise
            voronoi_binning(XYSN_FILE, V2B_FILE, V2B_XY_FILE)
    - Combine spectra of the same bin into fits files
            combine_spectra(V2B_FILE, IMAGE_CUBE, BIN_SCI, FLUX_SCI, BIN_VAR, FLUX_VAR)
    - Determine the optimal BIAS, first run ppxf with BIAS=0 to find the h3 and h4 parameters, put these into
      ppxf_simulation and determine the optimal BIAS. See the information in ppxf.py or in the ppxf readme.txt for
      more information about this step. I have selected BIAS=0.6 as a good value.
            ppxf_simulation(PPXF_BESTFIT.strip('.fits') + '_bias0.fits', LAM_RANGE, TARGET_SN, bias=0.5, spaxel=0)
    - Preform pPXF on the binned spectra
           ppxf_kinematics(BIN_SCI, PPXF_FILE, PPXF_BESTFIT, TEMPLATE_FITS, TEMPLATE_RESOLUTION, LAM_RANGE, VEL_INIT, SIG_INIT, bias=0.6)
To plot pPXF output:
    - Copy the desired spectral range and read mean flux values to be used for plotting
            scopy_flux(FLUX_SCI, FLUX_SCOPY_FITS, FLUX_SCOPY_RANGE, FLUX_SCOPY_FILE)
    - Access arrays to input into cap_plot_velfield(), set vel as velocity information obtained from ppxf
            plot_velfield_setup(vel, V2B_XY_FILE, FLUX_SCOPY_FILE)

As the spectra contain both emission and absorption features, it is important to separate the two before using fxcor
or rvsao as (unlike ppxf) they do not remove the emission lines
    - Create a spectra of the emission lines to study the gas kinematics
            remove_lines.remove_absorp_lines(BIN_SCI, PPXF_BESTFIT, EM_BIN_SCI)
    - Create a spectra of the emission lines to study the stellar kinematics
            remove_lines.remove_emission_lines(BIN_SCI, ABS_BIN_SCI, PPXF_BESTFIT, plot=False)
    *** xcsao is 'supposed' to have the ability to remove emission lines with 'em_chop+' but this kept returning very
        strange results, so I instead use the fits files with the emission lines removed manually.

To run fxcor to preform cross correlation to determine relative velocity:
    - Create input file from list of spectral bins and select the bin to use as the template
    - Absorption lines
            fxcor(EM_BIN_SCI, TEMPLATE_SPECTRA, EM_FXCOR_BIN_LIST, EM_FXCOR_FILE)
    - Emission lines
            fxcor(ABS_BIN_SCI, TEMPLATE_SPECTRA, ABS_FXCOR_BIN_LIST, ABS_FXCOR_FILE)
To plot the fxcor output:
    - Access arrays to input into cap_plot_velfield(), set vel as velocity information obtained from fxcor
            plot_velfield_setup(vel, V2B_XY_FILE, FLUX_SCOPY_FILE)

To run rvsao (with xcsao or emsao) to preform cross correlation to determine relative velocity:
    - Use the absorption lines spectra to study the relvative velocity of the stars
            rvsao(ABS_BIN_SCI, 'xcsao', TEMPLATE_SPECTRA, XCSAO_FILE, XCSAO_BIN_LIST)
    - Use the emission lines spectra to study the relvative velocity of the gas
            rvsao(EM_BIN_SCI, 'emsao', TEMPLATE_SPECTRA, EMSAO_FILE, EMSAO_BIN_LIST)
To plot, same method as above, with vel as the velcity information obtained from rvsao
            plot_velfield_setup(vel, V2B_XY_FILE, FLUX_SCOPY_FILE)

"""

# These are the files to change based on desired use
# --------------------------------------------------

DIR_PATH = ' '  # Working directory (where the 3D cube is)
IMAGE_CUBE = os.path.join(DIR_PATH, '3d_image_cube.fits')
TARGET_SN = 20  # Minimum S/N for Voronoi binning method

# To create 2D flattened science and variance images of specific spectral range
# This is only necessary for visual comparison of the intensity of regions for a given spectral range
SCROP_RANGE = [4330, 4380]  # wavelength range to scrop the image cube to which will be flattened
SCROP_PATH = os.path.join(DIR_PATH, 'scrop_proc')  # contains the output of all the scrop-type methods
CUBE_SCROPD = os.path.join(SCROP_PATH, 'IC225_3D_scropd_{}_{}.fits'.format(SCROP_RANGE[0], SCROP_RANGE[1]))
SCI_EXT_SCROPD = os.path.join(SCROP_PATH, 'IC225_2D_sci_{}_{}.fits'.format(SCROP_RANGE[0], SCROP_RANGE[1]))
VAR_EXT_SCROPD = os.path.join(SCROP_PATH, 'IC225_2D_var_{}_{}.fits'.format(SCROP_RANGE[0], SCROP_RANGE[1]))

# These variables define the chosen file structure I use to organise the output
# -----------------------------------------------------------------------------

# Flattened 3D cubes images - necessary to read the signal and noise for each pixel in binning
SCI_EXT = os.path.join(DIR_PATH, '2D_sci.fits')
VAR_EXT = os.path.join(DIR_PATH, '2D_var.fits')

PROC_PATH = os.path.join(DIR_PATH, 'proc_{}/'.format(TARGET_SN))

# Organise output of Voronoi binning
XYSN_FILE = os.path.join(PROC_PATH, 'y_x_signal_noise.txt')  # output of make_table
V2B_FILE = os.path.join(PROC_PATH, 'v2b_output_sn{}.txt'.format(TARGET_SN))  # output of voronoi binning
V2B_XY_FILE = os.path.join(PROC_PATH, 'v2b_output_xy_sn{}.txt'.format(TARGET_SN))  # output of voronoi binning

# Files with equivalent output as v2b method but with self defined bins
SRCBIN_FILE = os.path.join(PROC_PATH, 'srcbin_output_sn{}.txt'.format(TARGET_SN))
SRCBIN_XY_FILE = os.path.join(PROC_PATH, 'srcbin_output_xy_sn{}.txt'.format(TARGET_SN))

# Organise combined spectra folder and names of binned (summed) and flux'd (averaged) spectra
DIR_SCI_COMB = os.path.join(PROC_PATH, 'comb_fits_sci_{}'.format(TARGET_SN))  # folder with combined spectra
DIR_VAR_COMB = os.path.join(PROC_PATH, 'comb_fits_var_{}'.format(TARGET_SN))  # folder with combined var spec
BIN_SCI = os.path.join(DIR_SCI_COMB, 'bin_sci_{}.fits')  # naming convention of combined (sum) sci spectra
BIN_VAR = os.path.join(DIR_VAR_COMB, 'bin_var_{}.fits')  # naming convention of combined (sum) var spectra
FLUX_SCI = os.path.join(DIR_SCI_COMB, 'flux_sci_{}.fits')  # naming convention of combined (average) sci spectra
FLUX_VAR = os.path.join(DIR_VAR_COMB, 'flux_var_{}.fits')  # naming convention of combined (average) var spectra

EM_BIN_SCI = os.path.join(DIR_SCI_COMB, 'em_bin_sci_{}.fits')
ABS_BIN_SCI = os.path.join(DIR_SCI_COMB, 'abs_bin_sci_{}.fits')

# Organise output of pPXF
PPXF_PATH = os.path.join(PROC_PATH, 'ppxf_proc')
PPXF_FILE = os.path.join(PPXF_PATH, 'ppxf_output_sn{}.txt'.format(TARGET_SN))
PPXF_BESTFIT = os.path.join(PPXF_PATH, 'bestfit_{}.fits')
PPXF_FILE_2CMP = os.path.join(PPXF_PATH, 'ppxf_output_sn{}_2cmp.txt'.format(TARGET_SN))
PPXF_BESTFIT_2CMP = os.path.join(PPXF_PATH, 'bestfit_{}_2cmp.fits')

TEMPLATE_SPECTRA = 8  # Select a particular spectra as the template for cross-correlation, alternatively, adapt 
    # the routine to select a particular spectra for a given template file

# Organise output of fxcor
FXCOR_PATH = os.path.join(PROC_PATH, 'fxcor_proc')
EM_FXCOR_FILE = os.path.join(FXCOR_PATH, 'fxcor_em_bin_sci_sn{}'.format(TARGET_SN,))
EM_FXCOR_BIN_LIST = os.path.join(FXCOR_PATH, 'em_bin_sci_sn{}_list.lis'.format(TARGET_SN))
ABS_FXCOR_FILE = os.path.join(FXCOR_PATH, 'fxcor_abs_bin_sci_sn{}'.format(TARGET_SN,))
ABS_FXCOR_BIN_LIST = os.path.join(FXCOR_PATH, 'abs_bin_sci_sn{}_list.lis'.format(TARGET_SN))

# Organize output of rvsao
RVSAO_PATH = os.path.join(PROC_PATH, 'rvsao_proc')
BIN_LIST = os.path.join(RVSAO_PATH, 'bin_sci_sn{}_list.lis'.format(TARGET_SN))
XCSAO_BIN_LIST = os.path.join(RVSAO_PATH, 'abs_bin_sci_sn{}_list.lis'.format(TARGET_SN))
XCSAO_TEMPLATE = os.path.join(RVSAO_PATH, BIN_SCI.format(TEMPLATE_SPECTRA))
XCSAO_BADLINES = os.path.join(RVSAO_PATH, 'badlines.dat')  # if there are anly regions with large noise features create this file
XCSAO_FILE = os.path.join(RVSAO_PATH, 'xcsao_bin_sci_sn{}.txt'.format(TARGET_SN))
EMSAO_FILE = os.path.join(RVSAO_PATH, 'emsao_bin_sci_sn{}.txt'.format(TARGET_SN))
EMSAO_BIN_LIST = os.path.join(RVSAO_PATH, 'em_bin_sci_sn{}_list.lis'.format(TARGET_SN))

# pPXF parameters
VEL_INIT = 1500.  # initial guess for velocity
SIG_INIT = 150.  # inital guess of sigma distribution
LAM_RANGE = [4173.89, 5404.51]  # wavelength range for logarithmic rebinning (full range)
# template from MILES Library spanning 3540-7410 A, with resolution spectral resolution of 2.5 A (FWHM),
# sigma~64 km/s, R~2000. From Sanchez-Blazquez, et al. (2006)
# (http://www.iac.es/proyecto/miles/pages/stellar-libraries/miles-library.php)
TEMPLATE_FITS = '.../ppxf/miles_models/Mun1.30*.fits'
FWHM_tem = 2.51  # Vazdekis+10 spectra have a resolution FWHM of 2.51A.
FWHM_gal = 2.1  # ~2.28 (measured from FWHM of arc) pix * 0.911 A/pix = 2.1 A FWHM

# To create combined (averaged) spectra to determine mean flux of a specific wavelength range to plot
# YOU DO NOT NEED TO CROP THE SPECTRA HERE, it is just a good idea to check that you measure the same velocity for
# any given spectral range
FLUX_SCOPY_RANGE = [4173.89, 5404.51]
FLUX_SCOPY_FITS_SUFFIX = 'flux_scopy_{}.fits'  # wavelength range combined sci spectra from flux*
FLUX_SCOPY_FILE = os.path.join(PROC_PATH, 'binned_flux_{}.txt'.format(TARGET_SN))
FLUX_SCOPY_FITS = os.path.join(DIR_SCI_COMB, FLUX_SCOPY_FITS_SUFFIX)


def scrop_cube(image_cube, scrop_range, cube_scropd):
    """
    As the output cube of the preprocessing is NOT an MEF but instread a simple fits file with a SCI and VAR extension
    Gwen's scrop task cannot be used. This should replace the functionality.
    This takes a 3D cube and returns a 3D cube cropped to the wavelength specified in scrop_range

    NOTE: Still need to implement changes to header values

    INPUT: IMAGE_CUBE (dcsteqpxbprgN20051205S0006_add.fits), SCROP_RANGE ([4360, 4362])
    OUTPUT: CUBE_SCOPYD
    """

    if os.path.exists(cube_scropd):
        print('File {} already exists'.format(cube_scropd))
        return

    print ('Scropping cube {} to wavelength range {}'.format(image_cube, scrop_range))

    with fits.open(image_cube) as cube_hdu:
        # cube_hdu.info()
        # Filename: dcsteqpxbprgN20051205S0006_add.fits
        # No.    Name         Type      Cards   Dimensions   Format
        # 0    PRIMARY     PrimaryHDU     214   ()
        # 1    SCI         ImageHDU        68   (76, 49, 1300)   float32
        # 2    VAR         ImageHDU        68   (76, 49, 1300)   float32
        cube_data0 = cube_hdu[0].data
        cube_data1 = cube_hdu[1].data
        cube_data2 = cube_hdu[2].data
        cube_hdr0 = cube_hdu[0].header
        cube_hdr1 = cube_hdu[1].header
        cube_hdr2 = cube_hdu[2].header

    crval3 = cube_hdr1['CRVAL3']
    crpix3 = cube_hdr1['CRPIX3']
    cd33 = cube_hdr1['CD3_3']
    npix = cube_hdr1['NAXIS3']

    wmin = crval3 + (1. - crpix3) * cd33
    wmax = crval3 + (npix - crpix3) * cd33
    # dwav = cd33

    assert scrop_range[0] >= wmin, 'Cannot crop spectra outside of wavelength range [{},{}]'.format(wmin, wmax)
    assert scrop_range[1] <= wmax, 'Cannot crop spectra outside of wavelength range [{},{}]'.format(wmin, wmax)

    x1 = (scrop_range[0] - crval3) / cd33 + crpix3
    x2 = (scrop_range[1] - crval3) / cd33 + crpix3

    scrop_cube1 = cube_data1[x1:x2, :, :]
    scrop_cube2 = cube_data2[x1:x2, :, :]

    cube_hdr1['NAXIS3'] = scrop_cube1.shape[0]
    cube_hdr2['NAXIS3'] = scrop_cube2.shape[0]

    hdu_out = fits.HDUList()
    hdu_out.append(fits.PrimaryHDU(data=cube_data0, header=cube_hdr0))
    hdu_out.append(fits.ImageHDU(data=scrop_cube1, name='SCI', header=cube_hdr1))
    hdu_out.append(fits.ImageHDU(data=scrop_cube2, name='VAR', header=cube_hdr2))
    hdu_out.header = cube_hdr0
    hdu_out.writeto(cube_scropd)


def imcombine_flatten(cube, sci_ext, var_ext):
    """
    Imcombine both variance and science extensions to flatten the cube to 2D
    INPUT: CUBE (scrop_proc/IC225_3D_{}_{}.fits)
    OUTPUT: SCI_EXT (scrop_proc/IC225_2D_sci_{}_{}.fits), VAR_EXT (scrop_proc/IC225_2D_var_{}_{}.fits)
    """

    if os.path.exists(sci_ext):
        print('File {} already exists'.format(sci_ext))
        return
    if os.path.exists(var_ext):
        print('File {} already exists'.format(var_ext))
        return

    from pyraf import iraf

    iraf.imcombine('{}[sci]'.format(cube), sci_ext, project="yes")
    iraf.imcombine('{}[var]'.format(cube), var_ext, project="yes")


def make_table(image_cube, sci_ext, var_ext, xysn_file):
    """
    Read in the pixel values of the science and varience planes of the image cube and make a table of the
    coordinates with the respective signal and noise measurements
    INPUT: SCI_EXT (IC225_2D_sci.fits), VAR_EXT (IC225_2D_var.fits)
    OUTPUT: XYSN_FILE (y_x_signal_noise.txt)
    """

    if os.path.exists(xysn_file):
        print('File {} already exists'.format(xysn_file))
        return

    if not os.path.exists(sci_ext):
        imcombine_flatten(image_cube, sci_ext, var_ext)

    assert os.path.exists(sci_ext), 'Image {} does not exist'.format(sci_ext)
    assert os.path.exists(var_ext), 'Image {} does not exist'.format(var_ext)

    with fits.open(sci_ext) as sci_hdu:
        sci_data = sci_hdu[0].data
        sci_xaxis = sci_hdu[0].header['NAXIS1']
        sci_yaxis = sci_hdu[0].header['NAXIS2']

    with fits.open(var_ext) as var_hdu:
        var_data = var_hdu[0].data
        var_xaxis = var_hdu[0].header['NAXIS1']
        var_yaxis = var_hdu[0].header['NAXIS2']

    assert sci_xaxis == var_xaxis, 'Sci and var planes have diffent x dimensions'
    assert sci_yaxis == var_yaxis, 'Sci and var planes have diffent y dimensions'

    with open(xysn_file, 'w') as outfile:
        for i in range(sci_yaxis - 1):
            for j in range(sci_xaxis - 1):
                noise = np.sqrt(var_data[i, j])
                outfile.write('   {}   {}   {}   {}\n'.format(i, j, sci_data[i, j], noise))


def voronoi_binning(xysn_file, v2b_file, v2b_xy_file):
    """
    Follows example script provided from Michele Cappellari for voronoi 2d binning
    INPUT: XYSN_FILE (x_y_signal_noise.txt)
    OUTPUT: V2B_FILE (v2b_output_sn30.txt), V2B_XY_FILE (v2b_output_xy_sn30.txt)

    Output variables of voronoi_2d_binning (copied straight from the description in the script):
        BINNUMBER: Vector (same size as X) containing the bin number assigned to each input pixel. The index goes from
            zero to Nbins-1. This vector alone is enough to make *any* subsequent computation on the binned data.
            Everything else is optional!
        XBIN: Vector (size Nbins) of the X coordinates of the bin generators. These generators uniquely define the
            Voronoi tessellation.
        YBIN: Vector (size Nbins) of Y coordinates of the bin generators.
        XBAR: Vector (size Nbins) of X coordinates of the bins luminosity weighted centroids. Useful for plotting
            interpolated data.
        YBAR: Vector (size Nbins) of Y coordinates of the bins luminosity weighted centroids.
        SN: Vector (size Nbins) with the final SN of each bin.
        NPIXELS: Vector (size Nbins) with the number of pixels of each bin.
        SCALE: Vector (size Nbins) with the scale length of the Weighted Voronoi Tessellation, when the /WVT keyword is
            set. In that case SCALE is *needed* together with the coordinates XBIN and YBIN of the generators, to
            compute the tessellation (but one can also simply use the BINNUMBER vector).
    """

    if os.path.exists(v2b_file):
        print('File {} already exists'.format(v2b_file))
        return

    y, x, signal, noise = np.loadtxt(xysn_file, unpack=True)  # , skiprows=3)

    # Only select pixels where the signal and noise is nonzero, this often results in a few missed pixels
    # around the border of the image. This method is easier than parsing the xysn_file to manually remove pixels with
    # noise values <= 0.
    #
    noise = noise[noise > 0.]
    signal = signal[noise > 0.]
    x = x[noise > 0.]
    y = y[noise > 0.]

    # Perform the actual computation. The vectors (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    #
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x, y, signal, noise, TARGET_SN,
                                                                              plot=1, quiet=0)

    # Save to a text file the initial coordinates of each pixel together with the corresponding bin number computed
    # by this procedure. binNum uniquely specifies the bins and for this reason it is the only
    # number required for any subsequent calculation on the bins.
    #
    np.savetxt(v2b_file, np.column_stack([x, y, binNum]), header='x  y  binNum', fmt=b'%10.6f %10.6f %8i')
    np.savetxt(v2b_xy_file, np.column_stack([xBar, yBar, xNode, yNode]), header='xBar  yBar  xNode   yNode',
               fmt=b'%10.6f %10.6f %10.6f %10.6f')


def combine_spectra(v2b_file, image_cube, bin_sci, flux_sci, bin_var, flux_var):
    """
    Combine each pixel of the same bin (according to the lists output by voronoi_binning) into a single fits file
    INPUT: V2B_FILE (v2b_output_sn30.txt)
    OUTPUT: DIR_SCI_COMB (comb_fits_sci_{S/N}/bin_sci_{S/N}.fits), DIR_VAR_COMB (comb_fis_var_{S/N}/bin_var_{S/N}.fits)
    """

    # read in the output of the voronoi binning
    v2b_output = pd.read_table(v2b_file, sep=r"\s*", engine='python', skiprows=1, names=["x", "y", "bin"])

    # for each bin, combine the spectra of the same bin via numpy operations to avoiding floating point error in
    # IRAF which occurs when combining too many spectra
    #
    for i in range(np.amax(v2b_output['bin'].values) + 1):
        bin_sci_i = bin_sci.format(i)
        bin_var_i = bin_var.format(i)
        flux_sci_i = flux_sci.format(i)
        flux_var_i = flux_var.format(i)

        if not os.path.exists(bin_sci_i):
            # Check to see how many pixels are in this bin
            spaxel_list = v2b_output.query('bin == {}'.format(i))
            print('Number of pixels in bin {}: {}'.format(i, len(spaxel_list)))

            imcombine(image_cube, bin_sci_i, bin_var_i, spaxel_list, 'sum')
            imcombine(image_cube, flux_sci_i, flux_var_i, spaxel_list, 'average')


def imcombine(image_cube, outsci, outvar, spaxel_list, combine_method):
    """
    In order to avoid a 'floating point error' in imexamine I'm going to try and combine the fits sections as
    numpy arrays instead and manually copy the header values. Only works for combine='average'|'sum' at the moment
    """

    assert combine_method is 'sum' or 'average', 'Combine is not "sum" or "average"'

    # IMAGE_CUBE[int(line["x"].values[j - 1]) + 1, int(line["y"].values[j - 1]) + 1, *]

    with fits.open(image_cube) as image_cube_hdu:
        # Filename: dcsteqpxbprgN20051205S0006_add.fits
        # No.    Name         Type      Cards   Dimensions   Format
        # 0    PRIMARY     PrimaryHDU     214   ()
        # 1    SCI         ImageHDU        68   (76, 49, 1300)   float32
        # 2    VAR         ImageHDU        68   (76, 49, 1300)   float32
        image_cube_sci = image_cube_hdu[1].data
        image_cube_var = image_cube_hdu[2].data
        cube_header = image_cube_hdu['SCI'].header
        cube_header_0 = image_cube_hdu[0].header

    imcomb_sci = []
    imcomb_var = []

    # image_cube_sci.shape = (1300, 49, 76) ie (z,y,x)
    for k in range(image_cube_sci.shape[0]):
        pix_sci = []
        pix_var = []
        for j in range(len(spaxel_list)):
            pix_sci.append(image_cube_sci[k, spaxel_list["y"].values[j - 1] + 1, spaxel_list["x"].values[j - 1] + 1])
            pix_var.append(image_cube_var[k, spaxel_list["y"].values[j - 1] + 1, spaxel_list["x"].values[j - 1] + 1])

        if combine_method == 'sum':
            imcomb_sci.append(np.array(pix_sci).sum())
            imcomb_var.append(np.array(pix_var).sum())
        elif combine_method == 'average':
            imcomb_sci.append(np.array(pix_sci).mean())
            imcomb_var.append(np.array(pix_var).mean())

    write_imcomb_fits(imcomb_sci, outsci, cube_header, cube_header_0)
    write_imcomb_fits(imcomb_var, outvar, cube_header, cube_header_0)


def write_imcomb_fits(outdata, outfile, cube_header, cube_header_0):
    """
    Write the combined spectra to a fits file with headers from the wavelength dimension of the 3D cube
    """
    outfile_hdu = fits.PrimaryHDU()
    outfile_hdu.header['OBSERVAT'] = cube_header_0['OBSERVAT']
    outfile_hdu.header['CD1_1'] = cube_header['CD3_3']
    outfile_hdu.header['CRPIX1'] = cube_header['CRPIX3']
    outfile_hdu.header['CRVAL1'] = cube_header['CRVAL3']

    outfile_hdu.header['DATE'] = cube_header_0['DATE']
    outfile_hdu.header['INSTRUME'] = cube_header_0['INSTRUME']
    outfile_hdu.header['OBJECT'] = cube_header_0['OBJECT']
    outfile_hdu.header['GEMPRGID'] = cube_header_0['GEMPRGID']
    outfile_hdu.header['OBSID'] = cube_header_0['OBSID']
    outfile_hdu.header['OBSERVAT'] = cube_header_0['OBSERVAT']
    outfile_hdu.header['TELESCOP'] = cube_header_0['TELESCOP']
    outfile_hdu.header['EQUINOX'] = cube_header_0['EQUINOX']
    outfile_hdu.header['EPOCH'] = cube_header_0['EPOCH']
    outfile_hdu.header['RA'] = cube_header_0['RA']
    outfile_hdu.header['DEC'] = cube_header_0['DEC']
    outfile_hdu.header['DATE-OBS'] = cube_header_0['DATE-OBS']
    outfile_hdu.header['TIME-OBS'] = cube_header_0['TIME-OBS']
    outfile_hdu.header['UTSTART'] = cube_header_0['UTSTART']
    outfile_hdu.header['UTEND'] = cube_header_0['UTEND']
    outfile_hdu.header['EXPTIME'] = cube_header_0['EXPTIME']

    outfile_hdu.header['DC-FLAG'] = 0  # ie linear binning (1 = linear-log binning)
    # Header values readable by rvsao
    outfile_hdu.header['DEC--TAN'] = 'LAMBDA'
    outfile_hdu.data = outdata
    outfile_hdu.writeto(outfile, clobber=True)


def scopy_flux(flux_sci, flux_scopy_fits, flux_scopy_range, flux_scopy_file):
    """
    Combine (average) all spectra (according to bin) in the image for a given spectral range, calculate mean flux.
    This is used for 1 mag contours when plotting the velocity fields.
    INPUT: FLUX_SCI, FLUX_SCOPY_FITS, FLUX_SCOPY_RANGE
    OUTPUT: FLUX_SCOPY_FILE
    """

    if os.path.exists(flux_scopy_file):
        print('File {} already exists'.format(flux_scopy_file))
        return

    files_in_dir = glob.glob(flux_sci.format('*'))
    assert len(files_in_dir) > 0, 'No files match {}'.format(flux_sci.format('*'))

    from pyraf import iraf

    iraf.noao()
    iraf.onedspec()

    flux_scopy_fits_i_data_mean = []

    for i in range(len(files_in_dir)):

        flux_sci_i = flux_sci.format(i)
        flux_scopy_fits_i = flux_scopy_fits.format(i)

        if not os.path.exists(flux_scopy_fits_i):
            iraf.scopy(flux_sci_i, flux_scopy_fits_i, w1=flux_scopy_range[0], w2=flux_scopy_range[1])

        flux_scopy_fits_i_data = fits.getdata(flux_scopy_fits_i, 0)
        assert flux_scopy_fits_i_data.ndim != 0, "Scrop'd array is empty"

        flux_scopy_fits_i_data_mean.append(flux_scopy_fits_i_data.mean())

    np.array(flux_scopy_fits_i_data_mean).tofile(flux_scopy_file, sep='\n')


def fxcor(spec, task, template_spec, spec_list_file, fxcor_output_file, fxcor_range="4173-5447", interactive='no'):
    """
    Run fxcor on the binned spectra to determine the relative velocity (with respect to the template spectra)
    INPUT: BIN_SCI, FXCOR_TEMPLATE
    OUTPUT: FXCOR_BIN_LIST, FXCOR_FILE(s)

    Note: make sure you select the parameters for your desired use. Refer to the IRAF rv package help for description
    of what each parameter is/does
    """

    if os.path.exists(fxcor_output_file + '.txt'):
        print('File {} already exists'.format(fxcor_output_file + '.txt'))
        return

    assert task == "abs" or task == "ems", "'task' is neight 'ab' or 'ems'"

    spec_list = []
    for i in range(len(glob.glob(spec.format('*')))):
        spec_list.append(spec.format(i))
    assert len(spec_list) > 0, 'Input files {} do not exist'.format(spec.format('*'))
    np.array(spec_list).tofile(spec_list_file, sep='\n')

    from pyraf import iraf

    iraf.noao()
    iraf.rv()

    if task == 'ems':
        iraf.fxcor('@{}'.format(spec_list_file), spec.format(template_spec), output=fxcor_output_file, continuum="both",
                   interactive=interactive, order=1, high_rej=2, low_rej=2, osample=fxcor_range, rsample=fxcor_range,
                   rebin="smallest", imupdate="no", pixcorr="no", filter="both", f_type="welch", cuton=20, cutoff=1000,
                   fullon=30, fulloff=800, ra="RA", dec="DEC", ut="UTSTART", epoch="EQUINOX", verbose="txtonly")

    elif task == 'abs':
        # Run interactively to make sure not fitting noise features, adapet osample/rsample to avoid such noise
        iraf.fxcor('@{}'.format(spec_list_file), spec.format(template_spec), output=fxcor_output_file, continuum="both",
                   interactive=interactive, order=1, high_rej=2, low_rej=2, osample=fxcor_range, rsample=fxcor_range,
                   rebin="smallest", imupdate="no", pixcorr="no", filter="both", f_type="welch", cuton=20, cutoff=800,
                   fullon=30, fulloff=800, ra="RA", dec="DEC", ut="UTSTART", epoch="EQUINOX", verbose="txtonly")

    assert os.path.exists(fxcor_output_file + '.txt'), 'Error in iraf.fxcor: File {} was not created'.format(
        fxcor_output_file + '.txt')


def rvsao(bin_sci, task, template_spectra, rvsao_file, rvsao_bin_list, interactive="no", linesig=1.5,
          czguess=0., st_lambda="INDEF", end_lambda="INDEF", badlines=None):
    """
    Use the rvsao task emsao or xcsao to measure relative velocity from emission or absoption spectra
    Note: make sure you select the parameters for your desired use. Refer to the IRAF rvsao package help for description
    of what each parameter is/does
    """

    assert task == 'xcsao' or task == 'emsao', "task is not either 'xcsao' or 'emsao'"

    if os.path.exists(rvsao_file):
        print('File {} already exists'.format(rvsao_file))
        return

    fixbad = "no"
    if badlines:
        fixbad = "yes"

    bin_list = []
    for i in range(len(glob.glob(bin_sci.format('*')))):  # to ensure order is 0-61 (not 0, 1, 10, 11, etc)
        bin_list.append(bin_sci.format(i))
    assert len(bin_list) > 0, 'Absorption/emission(?) bin spectra do not exist: {}'.format(em_bin_sci.format('*'))
    np.array(bin_list).tofile(rvsao_bin_list, sep='\n')

    from pyraf import iraf

    iraf.images()
    iraf.rvsao()

    if task == 'xcsao':
        iraf.xcsao('@{}'.format(rvsao_bin_list), templates=bin_sci.format(template_spectra), report_mode=2,
                   logfiles=rvsao_file, displot=interactive, low_bin=10, top_low=20, top_nrun=80, nrun=211,
                   zeropad="yes", nzpass=1, curmode="no", pkmode=2, s_emchop="no", vel_init="guess", czguess=czguess,
                   st_lambda=st_lambda, end_lambda=end_lambda, fixbad=fixbad, badlines=badlines)

    elif task == 'emsao':

        # Run this interactively as the fit can fail often. Depending on S/N you may have to decrease the linesig
        # to detect the lines, or if lines are matched inconsistently play with the st_lambda, end_lambda parameters
        iraf.emsao('@{}'.format(rvsao_bin_list), logfiles=rvsao_file, displot=interactive, report_mode=2,
                   contsub_plot="no", st_lambda=st_lambda, end_lambda=end_lambda,  # vel_init="guess", czguess=czguess,
                   linesig=linesig, fixbad=fixbad, badlines=badlines)
        # linesig=0.92, vel_init="guess", czguess=1500,


def plot_velfield_setup(vel, v2b_xy_file, flux_scopy_file):
    """
    Use default plotting scheme of pPXF to visualize the velocity field with 1 mag flux contours
    """

    assert os.path.exists(v2b_xy_file), 'File {} does not exist'.format(v2b_xy_file)
    assert os.path.exists(flux_scopy_file), 'File {} does not exist'.format(flux_scopy_file)
    assert len(vel) > 0, 'Input velocity does not make sense'

    xbar, ybar, xnode, ynode = np.loadtxt(v2b_xy_file, unpack=True, skiprows=1)
    flux = np.loadtxt(flux_scopy_file, unpack=True)

    assert len(xbar) == len(ybar), 'Xbar is not the same length as Ybar'
    assert len(xbar) == len(vel), 'Xbar is not the same length as vel'
    assert len(xbar) == len(flux), 'Xbar is not the same length as flux'

    plt.clf()
    plt.title('Velocity')
    plot_velfield(xbar, ybar, vel, flux=flux, colorbar=True, label='km/s')
    plt.show()


def plot_velfield_bybin(vel, v2b_file, vmin=None, vmax=None):
    """
    Read in the velocity and spatial information and plot with a colourbar
    to change range of colourbar use vmin and vmax parameters.
    Pixels without a bin will be plotted with the mean velocity value.
    """

    if vmin is None:
        vmin = np.min(vel)

    if vmax is None:
        vmax = np.max(vel)

    bin_table = pd.read_table(v2b_file, sep=r"\s*", engine='python', names=["x", "y", "binNum"], skiprows=1)

    vel_arr = np.zeros(shape=len(bin_table.index))

    for i in range(len(vel)):
        query = bin_table.query('binNum == {}'.format(i))
        for idx in query.index:
            vel_arr[idx] = vel[i]

    x_uq = np.unique(bin_table.x.values)
    y_uq = np.unique(bin_table.y.values)

    table = pd.DataFrame(data={'x': bin_table.x.values, 'y': bin_table.y.values, 'vel': vel_arr})

    grid = np.zeros(shape=(y_uq.size, x_uq.size))
    for x in range(x_uq.size):
        for y in range(y_uq.size):
            query = table.query('x == {}'.format(x + x_uq[0])).query('y == {}'.format(y + y_uq[0]))
            if len(query.vel.values) == 0:
                print ('ERROR: coordinate ({},{}) does not have bin number'.format(x + x_uq[0], y + y_uq[0]))
                grid[y, x] = np.mean(table.vel.values)
            else:
                grid[y, x] = query.vel.values[0]

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(grid, vmin=vmin, vmax=vmax)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def read_emsao_output(emaso_input):
    """
    the output text file of emsao in report_mode=2 is conveniently difficult to parse as the rows are of variable length
    """

    emsao_vel = []
    with open(emaso_input) as infile:
        for line in infile:
            emsao_vel.append(float(line.split()[4]))
    return emsao_vel


def read_xcsao_output(xcsao_input):
    return pd.read_table(xcsao_input, sep=r"\s*", engine='python', usecols=[3], names=["vrel"], squeeze=True).values


def read_fxcor_output(fxcor_file):
    return pd.read_table(fxcor_file + '.txt', sep=r"\s*", engine='python', skiprows=16, usecols=[10],
                         names=["vrel"], squeeze=True).values


def crop_table(xysn_file, xysn_file_cropped, x=[25, 45], y=[20, 40]):
    """
    This is useful for if you want to completely ignore a particular region. I don't use this for anything though.
    """

    if os.path.exists(xysn_file_cropped):
        print ('File {} already exists'.format(xysn_file_cropped))
        return
    table = pd.read_table(xysn_file, sep=r"\s*", engine='python', names=["y", "x", "sig", "noise"])
    table_cropped = table.query("{} < x < {}".format(x[0], x[1])).query("{} < y < {}".format(y[0], y[1]))

    table_cropped.to_csv(xysn_file_cropped, sep=" ", header=False, index=False)


def bin_sources(sci_ext, xysn_file, v2b_file, v2b_xy_file, plot=False):
    """
    Create bins specifically about the three sources we observe in the 2d flatted image. This method was written
    specifically for observations of IC 225 where three flux sources were seen in the center of the galaxy. We modeled
    these sources as ellipses, and split the background into equal regions in a pinwheel patters for the velocity
    field analysis. Keep in mind this method does not take into consideration the S/N of each bin. We chose this pattern
    because it was close to the Voronoi bin pattern for S/N 40, but when later doing the stellar population history
    measurements we binned up the backgroung bins again to increase the S/N.
    """

    if os.path.exists(v2b_file) & os.path.exists(v2b_xy_file):
        print ('Files {}, {} already exists'.format(v2b_file, v2b_xy_file))
        return

    data = fits.getdata(sci_ext)

    # Find centercoordinates of each object (in this case we have three sources)
    c1 = [35.84, 29.03]  # center nucleas
    c2 = [48.47, 25.32]  # off center nucleus
    c3 = [31.16, 17.48]  # blob features
    # find elliptical parameters of each object
    a1 = 5.
    b1 = 5.44
    a2 = 4.23
    b2 = 5.2
    a3 = 5.
    b3 = 3.6

    cby, cbx = data.shape
    arr = np.zeros(shape=(cby, cbx))

    # Cut background into four squares
    arr[0:int(cby / 2), 0:int(cbx / 2)] = 0.
    arr[int(cby / 2):cby, 0:int(cbx / 2)] = 6.
    arr[0:int(cby / 2), int(cbx / 2):cbx] = 2.
    arr[int(cby / 2):cby, int(cbx / 2):cbx] = 4.

    # Cut each of those squares in half to create a pinwheel pattern
    m = float(cby) / float(cbx)
    for x in range(int(cbx / 2)):
        for y in range(int(cby / 2)):
            if x > y / m:
                arr[y, x] = 1.
    for x in range(int(cbx / 2), cbx):
        for y in range(int(cby / 2)):
            if x > (y - cby) / (-1 * m):
                arr[y, x] = 3.
    for x in range(int(cbx / 2)):
        for y in range(int(cby / 2), cby):
            if x < (y - cby) / (-1 * m):
                arr[y, x] = 7.
    for x in range(int(cbx / 2), cbx):
        for y in range(int(cby / 2), cby):
            if x < y / m:
                arr[y, x] = 5.

    # Define ellipses over the sources
    arr = bin_by_object.ellipse1(arr, c1[0], c1[1], 6. + a1, 6. + b1, 8.)
    arr = bin_by_object.ellipse2(arr, c2[0], c2[1], 3. + a2, 3. + b2, 9.)
    arr = bin_by_object.ellipse3(arr, c3[0], c3[1], 4. + a3, 4. + b3, 10.)

    xx, yy, signal, noise = np.loadtxt(xysn_file, unpack=True)

    x = []
    y = []
    classe = []
    with open(v2b_file, 'w') as outfile:
        outfile.write('# x  y  binNum\n')
        for j in range(arr.shape[0] - 1):
            for i in range(arr.shape[1] - 1):
                if not (i in xx[noise == 0.]) and not (j in yy[noise == 0.]):
                    outfile.write('{}  {}  {}\n'.format(i, j, int(arr[j, i])))
                    x.append(i)
                    y.append(j)
                    classe.append(int(arr[j, i]))

    x = np.array(x)
    y = np.array(y)
    classe = np.array(classe)

    good = np.unique(classe)
    xNode = ndimage.mean(x, labels=classe, index=good)
    yNode = ndimage.mean(y, labels=classe, index=good)

    if plot:
        weights = [xNode, yNode]
        plt.imshow(arr)
        plt.scatter(*weights)
        plt.show()

    # I have issues here creating the barycentric nodes, so instead i choose to only use the centerpoints already found
    # It seems that inputting the xNode and yNode into the _cvt_equal_mass function changes the values
    # This is only useful for obesrving the center of the bins anyway for plotting purposes, and should not affect our
    # measurements/analysis in the least
    #
    # cvt=True
    # pixelsize=None
    # plot=True
    # quiet=True
    # wvt=True
    # xNode2, yNode2, scale, it = voronoi_2d_binning._cvt_equal_mass(x, y, signal, noise, xNode, yNode, quiet, wvt)
    # classe, xBar, yBar, sn, area = voronoi_2d_binning._compute_useful_bin_quantities(x, y, signal, noise, xNode2, yNode2, scale)

    xBar = xNode
    yBar = yNode
    np.savetxt(v2b_xy_file, np.column_stack([xBar, yBar, xNode, yNode]), header='xBar  yBar  xNode   yNode',
               fmt=b'%10.6f %10.6f %10.6f %10.6f')


def remove_emission_lines(bin_sci, ppxf_bestfit, abs_bin_sci, plot=True, bad_lines=[]):
    """
    Fit the absorption spectra with a pVoight function by parameterizing the bestfit template as output from pPXF,
    fit the emission lines over the absorption features with a sum of a gaussian and lorentz and subtract from the
    original spectra. Where the emission lines are isolated, just interpolate the continuum values from the best fit
    template and replace.
    This requires a very good fit between the galaxy spectra and the bestfit spectra. I recommend running this with
    plot=True to check for any residuals.
    """

    for j in range(len(glob.glob(bin_sci.format('*')))):

        bin_sci_j = bin_sci.format(j)
        ppxf_bestfit_j = ppxf_bestfit.format(j)

        if not os.path.exists(abs_bin_sci.format(j)):

            if plot:
                print('>>>>> Removing emission lines from spectra {}'.format(j))

            with fits.open(bin_sci_j) as hdu:
                odata = hdu[0].data
                ohdr = hdu[0].header

            bestfit = fits.getdata(ppxf_bestfit_j)

            lamRange = ohdr['CRVAL1'] + np.array([1. - ohdr['CRPIX1'], ohdr['NAXIS1'] - ohdr['CRPIX1']]) * ohdr['CD1_1']
            lin_bins = np.linspace(lamRange[0], lamRange[1], ohdr['NAXIS1'])

            # Log bin the spectra to match the best fit absorption template
            galaxy, logLam1, velscale = util.log_rebin(lamRange, odata)
            log_bins = np.exp(logLam1)
            emlns, lnames, lwave = util.emission_lines(logLam1, lamRange, 2.3)

            # Hgamma 4340.47, Hbeta 4861.33, OIII [4958.92, 5006.84]
            # lwave = [4340.47, 4861.33, 4958.92, 5006.84]

            # find the index of the emission lines
            iHg = (np.abs(log_bins - lwave[0])).argmin()
            iHb = (np.abs(log_bins - lwave[1])).argmin()
            iOIII = (np.abs(log_bins - lwave[2])).argmin()
            iOIIIb = (np.abs(log_bins - (lwave[2] - 47.92))).argmin()

            # There are BOTH absorption and emission features about the wavelength of Hgamma and Hbeta, so we need
            # to use a specialized fitting function (convolved Guassian and Lorentzian -> pVoight) to remove the
            # emission lines
            popt_Hg, pcov_Hg = remove_lines.fit_ems_pvoightcont(log_bins, galaxy, lin_bins, odata, iHg, bestfit)
            popt_Hb, pcov_Hb = remove_lines.fit_ems_pvoightcont(log_bins, galaxy, lin_bins, odata, iHb, bestfit)

            em_fit = remove_lines.gauss_lorentz(lin_bins, popt_Hg[0], popt_Hg[1], popt_Hg[2], popt_Hg[3], popt_Hg[4],
                                                popt_Hg[5]) + \
                     remove_lines.gauss_lorentz(lin_bins, popt_Hb[0], popt_Hb[1], popt_Hb[2], popt_Hb[3], popt_Hb[4],
                                                popt_Hb[5])

            abs_feat_fit = odata - em_fit

            abs_fit = remove_lines.em_chop(lin_bins, abs_feat_fit, iOIII, log_bins, bestfit)
            abs_fit = remove_lines.em_chop(lin_bins, abs_fit, iOIIIb, log_bins, bestfit)

            for lin in bad_lines:
                ibad = (np.abs(lin_bins - lin)).argmin()
                abs_fit = remove_lines.em_chop(lin_bins, abs_fit, ibad, log_bins, bestfit)

            if plot:
                plt.plot(lin_bins, odata, '-k', label="spectra")
                # plt.plot(lin_bins, bestfit, '--r', label="bestfit absorption line")
                plt.plot(lin_bins, abs_fit, '-b', label="absorption spectra")
                plt.legend()
                plt.show()

            abs_hdu = fits.PrimaryHDU()
            abs_hdu.data = abs_fit
            abs_hdu.header = fits.getheader(bin_sci.format(j), 0)
            abs_hdu.writeto(abs_bin_sci.format(j))


if __name__ == '__main__':
    """
    See detailed information at the top of the script
    """

    if not os.path.exists(PROC_PATH):
        os.mkdir(PROC_PATH)
    if not os.path.exists(SCROP_PATH):
        os.mkdir(SCROP_PATH)
    if not os.path.exists(FXCOR_PATH):
        os.makedirs(FXCOR_PATH)
    if not os.path.exists(PPXF_PATH):
        os.mkdir(PPXF_PATH)
    if not os.path.exists(RVSAO_PATH):
        os.mkdir(RVSAO_PATH)
    if not os.path.exists(DIR_SCI_COMB):
        os.mkdir(DIR_SCI_COMB)
    if not os.path.exists(DIR_VAR_COMB):
        os.mkdir(DIR_VAR_COMB)

    # UNCOMMENT AS NEEDED (not the fanciest system I know...)

    # ## Optional:
    # To flatten 3D cube to 2D in specific wavelengths to visualize gas emission maps
    # (Don't use these flattend cubes otherwise)
    #
    # scrop_cube(IMAGE_CUBE, SCROP_RANGE, CUBE_SCROPD)
    # imcombine_flatten(CUBE_SCROPD, SCI_EXT_SCROPD, VAR_EXT_SCROPD)

    # Prepare the input: flatted the cube to 2D to read signal and noise planes, then apply voronoi binning
    #
    # make_table(IMAGE_CUBE, SCI_EXT, VAR_EXT, XYSN_FILE)
    # voronoi_binning(XYSN_FILE, V2B_FILE, V2B_XY_FILE)

    # ## Optional: define bins around the sources instead of using voronoi binning method, built for S/N of ~ 30-50
    #
    # bin_sources(SCI_EXT, XYSN_FILE, SRCBIN_FILE, SRCBIN_XY_FILE) # ELLIPSE VALUES ARE HARDCODED
    # V2B_FILE = SRCBIN_FILE
    # V2B_XY_FILE = SRCBIN_XY_FILE

    # Combine spectra into thier respective bin, and calculate average flux of each bin (used for plotting):
    #
    # combine_spectra(V2B_FILE, IMAGE_CUBE, BIN_SCI, FLUX_SCI, BIN_VAR, FLUX_VAR)
    # scopy_flux(FLUX_SCI, FLUX_SCOPY_FITS, FLUX_SCOPY_RANGE, FLUX_SCOPY_FILE)

    # ## Optional: clean spectra of large noise features. Could define range of pixes (in log bins) to use with the
    # badPixel array below with pPXF if consistent across all spectra, or use clean=True with pPXF to remove outliers
    # from the fit (make sure to do plot=True when doing this)
    #
    # clean_spec(BIN_SCI, feat_sci, bad_region)

    # ## Do once:
    # To determine optimal penalty (BIAS) first run with BIAS=0 then preform monte carlo simulation
    # See information in ppxf.py (or the readme which comes with the ppxf download) for more details
    # The chosen penalty for IC 225 is BIAS=0.6
    #
    #PPXF_FILE_BIAS = PPXF_FILE.strip('.txt') + '_bias0.txt'
    #PPXF_BESTFIT_BIAS = PPXF_BESTFIT.strip('.fits') + '_bias0.fits'
    #ppxf_tasks.ppxf_kinematics(BIN_SCI, PPXF_FILE_BIAS, PPXF_BESTFIT_BIAS, TEMPLATE_FITS, FWHM_tem, FWHM_gal, VEL_INIT, SIG_INIT, 0)
    #ppxf_tasks.ppxf_simulation(PPXF_BESTFIT, TARGET_SN, bias=0.6, spaxel=0)

    # Run pPXF to measure stellar population kinematics (from absorption features) and plot results
    # Assign bad pixel range to spurious consistent noise features from imperfect interpolation across the chip gap.
    # If there are strong noise featues which effect the fit that are not constant, use clean=yes instead.
    #
    # badPixels = np.append(np.arange(384., 395.), np.arange(984., 996.))
    # ppxf_vel = ppxf_tasks.ppxf_kinematics(BIN_SCI, PPXF_FILE, PPXF_BESTFIT, TEMPLATE_FITS, FWHM_tem, FWHM_gal,
    #           VEL_INIT, SIG_INIT, bias=0.6, plot=False, badPixels=badPixels, clean=True)
    # ppxf_vel, ppxf_sig, h3, h4, ppxf_dvel, ppxf_dsig, dh3, dh4, chi2 = np.loadtxt(PPXF_FILE, unpack=True)
    # plot_velfield_setup(ppxf_vel, V2B_XY_FILE, FLUX_SCOPY_FILE)
    #
    # ## Optional: If using self-defined bins, plot with this method instead
    #
    # plot_velfield_bybin(ppxf_vel, V2B_FILE)
    
    # ## Optional: pPXF can also be used to measure the gas kinematics by fitting a 2 composite model, however the
    # fit of the stellar kinematics does not match what it measured by the method above, and doesn't make much sense.
    # Despite this, the gas kinematics match well the output from fxcor so they look a-okay
    #
    # ppxf_tasks.ppxf_population_gas(BIN_SCI, FWHM_gal, TEMPLATE_FITS, FWHM_tem, PPXF_FILE_2CMP, PPXF_BESTFIT_2CMP, VEL_INIT, 0.6, plot=False, clean=True)
    # ppxf_2cmp = pd.read_table(PPXF_FILE_2CMP, sep=r"\s*", skiprows=2, usecols=[0, 8], names=["vel","velg"])
    #
    # plot_velfield_setup(ppxf_2cmp.vel.values, V2B_XY_FILE, FLUX_SCOPY_FILE)
    # plot_velfield_setup(ppxf_2cmp.velg.values, V2B_XY_FILE, FLUX_SCOPY_FILE)

    # Make spectra of just emission and just absorption lines, necessary to have isolated absorption spectra for
    # cross-correlation fitting techniques, use plot=yes to check
    # As the smission lines in our case are quite strong, we can just use the galaxy spectra for the corss correlation.
    # You may have to adjsut the rvsao.emsao.lingsig parameter (default 1.5) to 1.0 to detect the lines if you're not
    # rebinning to S/N that's high enough for a 1.5sigma detection.
    #
    # This method fits the absorption spectra when coincident with emission lines, and fits the bestfit template
    # where the emission lines are isolated
    #
    # remove_emission_lines(BIN_SCI, PPXF_BESTFIT, ABS_BIN_SCI, plot=True, bad_lines=[4496])
    #
    # This method fits the absorption spectra when coincident with emission lines, and fits the isolated emission lines
    # to a sum of a gaussian and lorentz, and subtracts that fitted function to remove the emission line.
    #
    # remove_lines.fit_absorp_spectra(BIN_SCI, PPXF_BESTFIT, ABS_BIN_SCI, plot=True)
    # remove_lines.fit_emission_spectra(BIN_SCI, PPXF_BESTFIT, EM_BIN_SCI, plot=False)
    #
    # If this does not work (for example, if the emission features are not symmetric, or too close to each other)
    # you can isolate the emission spectra with the function remove_lines.subtract_bestfit and use the bestfit as the
    # absorption spectra. This method will give you a LOG binned spectra though.
    #
    # EM_BIN_SCI = EM_BIN_SCI.split('.fits')[0] + '_log.fits'
    # ABS_BIN_SCI = PPXF_BESTFIT
    # remove_lines.subtract_besftfit(BIN_SCI, PPXF_BESTFIT, EM_BIN_SCI)

    # Run rv fxcor for stellar population velocites, then gas velocities
    # select an appropriate range to cross-correlation i.e. not including the chip gap feature, run interactively
    # to see if this is an issue or not.
    #
    # fxcor(ABS_BIN_SCI, 'abs', TEMPLATE_SPECTRA, ABS_FXCOR_BIN_LIST, ABS_FXCOR_FILE, "4337-5000", "no")
    # plot_velfield_setup(read_fxcor_output(ABS_FXCOR_FILE), V2B_XY_FILE, FLUX_SCOPY_FILE)
    # plot_velfield_bybin(read_fxcor_output(ABS_FXCOR_FILE), V2B_FILE)
    #
    # fxcor(EM_BIN_SCI, 'ems', TEMPLATE_SPECTRA, EM_FXCOR_BIN_LIST, EM_FXCOR_FILE, "4337-5021", "no")
    # plot_velfield_setup(read_fxcor_output(EM_FXCOR_FILE), V2B_XY_FILE, FLUX_SCOPY_FILE)
    # plot_velfield_bybin(read_fxcor_output(EM_FXCOR_FILE), V2B_FILE)

    # Run rvsao xcsao for stellar population velocites, then gas velocities. If the S/N isn't high enough for a 1.5 sig
    # detection in emsao, adjust parameter linesig.
    # If the emission line fitting procedure does not work, use the original spectra instead.
    # rvsao.xcsao as the option (s_emchop, t_emchpp) to remove emission or absorption lines, but I haven't had much
    # success using it. Instead I use the technique above
    # adjust st_lambda and end_lambda to avoid regions of strong noise that give spurious measurements
    #
    # Create list of bad lines, which will be eliminated from the cross correlation. lambda step is 0.9110890030860901
    # np.savetxt(XCSAO_BADLINES, np.arange(4489, 4505, 0.9110890030860901))
    # rvsao(ABS_BIN_SCI, 'xcsao', TEMPLATE_SPECTRA, XCSAO_FILE, XCSAO_BIN_LIST, interactive="yes", st_lambda=4300.,
    #       end_lambda=5000., badlines=XCSAO_BADLINES)
    # plot_velfield_setup(read_xcsao_output(XCSAO_FILE), V2B_XY_FILE, FLUX_SCOPY_FILE)
    # plot_velfield_bybin(read_xcsao_output(XCSAO_FILE), V2B_FILE)
    #
    # rvsao(EM_BIN_SCI, 'emsao', TEMPLATE_SPECTRA, EMSAO_FILE, EMSAO_BIN_LIST, "no", linesig=1., czguess=1550.,
    # st_lambda=4600., end_lambda=5300.)
    # It looks a lot clearer using just the original spectra
    # rvsao(BIN_SCI, 'emsao', TEMPLATE_SPECTRA, EMSAO_FILE, EMSAO_BIN_LIST, "yes", 1., 1550., 4300.)
    # plot_velfield_setup(read_emsao_output(EMSAO_FILE), V2B_XY_FILE, FLUX_SCOPY_FILE)
    # plot_velfield_bybin(read_emsao_output(EMSAO_FILE), V2B_FILE)



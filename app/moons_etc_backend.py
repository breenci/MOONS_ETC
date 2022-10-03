#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:23:24 2021

@author: oscar.gonzalez

This is the backend

"""

import numpy as np
from numpy.random import rand
from astropy.io import fits, ascii
from scipy import ndimage
from scipy import interpolate
import matplotlib.pyplot as plt
import math
import csv
import configparser

# Load the configuration file
# No changes needed
def getConfig(ConfigFile):
    config = configparser.ConfigParser()
    read=config.read(ConfigFile)
    print('Succesfully loaded Configuration file: ',read)
    print('Succesfully loaded Configuration file: ',read)
    return config 


# load the parameter file generated from form. 
# Can this be sent without file? DONE
def getParams(ParamFile):
    params = configparser.ConfigParser()
    read=params.read(ParamFile)
    print('Succesfully loaded Parameter file: ',read)
    return params 


# get vals from paramfile
def get_input(params):
    uservals={}
    #hardcoded
    uservals['telescope']='VLT'#config['general']['telescope']
    uservals['instrument']='MOONS'#config['general']['instrument']
    uservals['set_line_profile']='NO'
    #target
    uservals['template_name']=str(params['target']['template_name'])
    uservals['filter']=str(params['target']['filter'])
    uservals['magnitude']=float(params['target']['magnitude'])
    uservals['mag_system']=str(params['target']['system'])
    uservals['reddening']=float(params['target']['reddening'])
    #instrument
    uservals['moons_mode']=str(params['instrument']['moons_mode'])
    uservals['strategy']=str(params['instrument']['strategy'])
    uservals['atm_dif_ref']=float(params['instrument']['adc_refwave'])
    uservals['exptime']=float(params['instrument']['exptime'])
#    uservals['dit']=float(params['instrument']['dit'])
    uservals['N_exp']=float(params['instrument']['nexp'])
    #conditions
    uservals['seeing']=float(params['conditions']['seeing'])
    uservals['airmass']=float(params['conditions']['airmass'])
    uservals['airmass_fl']=uservals['airmass']
    #simulations
    uservals['sky_residual']=float(params['simulation']['sky_residual'])
    uservals['sky_template']=params['simulation']['sky_template']
    uservals['telluric']=float(params['simulation']['telluric'])
    uservals['flux_calib']=float(params['simulation']['flux_calib'])

    #configuration
    return uservals


def calculate_ndits(uservals):
    uservals['N_dit']=int(uservals['exptime']/uservals['dit'])
    uservals['effective_exposure']=uservals['N_dit']*uservals['dit']
    print('Number of IR detector DITs: %s'%uservals['N_dit'])
    print('Total on source time in IR detectors: %s'%uservals['effective_exposure'])
    return uservals

def calculate_texp(uservals):
    uservals['texp']=uservals['exptime']/uservals['N_exp']
    return uservals

def setup_moons(uservals,config,bandpass,device):
    instrumentconfig={}
    instrumentconfig['wlr']=[float(config[bandpass]['wlr_i']),float(config[bandpass]['wlr_f'])]
    instrumentconfig['RON'] = float(config[device]['RON'])
    instrumentconfig['DKsec'] = float(config[device]['DK'])/3600.
    instrumentconfig['gain'] = float(config[device]['gain']) #ADU/e- for 4RGs consistent with Gianluca's model
    instrumentconfig['saturation_level'] = float(config[device]['saturation_level']) #ADU
    instrumentconfig['pix_size']= float(config[device]['pix_size'])
    instrumentconfig['spec_sampling'] = float(config[bandpass]['spec_sampling'])
    instrumentconfig['resolution']= float(config[bandpass]['resolution'])
    #instrumentconfig['template_wl_norm']= float(config[bandpass]['wav_ref'])
    QE_file='app/static/Inst_setup/%s'%(config[device]['QE'])
    instrumentconfig['QE_wav'], instrumentconfig['QE_eff']  = np.loadtxt(QE_file, unpack = True)
    instrumentconfig['sky_aperture']=1.1
    instrumentconfig['throughput']=config['instrument']['throughput']
    instrumentconfig['telescope_eff']=config['instrument']['telescope_eff']
    instrumentconfig['adc_table']= config['instrument']['adc_table']
    print('')
    return instrumentconfig

def set_telescope(uservals):
    telescopeconfig={}
    if (uservals['telescope']=='ELT'):
        telescopeconfig['t_aperture']=39.0
    if (uservals['telescope']=='VLT'):
        telescopeconfig['t_aperture']=8.2
    return telescopeconfig

def set_detector(telescopeconfig,uservals,instrumentconfig, device):
    #equivalent pixel sizes in arcsec
    detectorconfig={}
    detectorconfig['ypix_fwhm']=3.0  #projected sky size of pixel from Tino's paper design 0.35, fibre size (no field stop)=1.1"
    print(' ')
    print('Spectral sampling for current configuration: ',str(round(instrumentconfig['spec_sampling'],1)))
    print(' ')
    print('Spectral resolving power for current configuration: ',str(round(instrumentconfig['resolution'],1)))
    print(' ')
    detectorconfig['disp']=(instrumentconfig['wlr'][1]+instrumentconfig['wlr'][0])/2.0*1.0e4/instrumentconfig['resolution']/instrumentconfig['spec_sampling']
    detectorconfig['npix']=int(1.0e4*(instrumentconfig['wlr'][1]-instrumentconfig['wlr'][0])/detectorconfig['disp']) # Total number of pixels by fixing spectral range and sampling requirements in current baseline design
    print('Spectral dispersion for current configuration: ',str(round(detectorconfig['disp'],2)))
    print(' ')
    #uservals=calculate_ndits(uservals)
    uservals=calculate_texp(uservals)
    if device=='CCD':
        detectorconfig['dit']=uservals['texp']
        detectorconfig['N_dit']=uservals['N_exp']
    if device=='IR':
        detectorconfig['dit']=uservals['texp']
        detectorconfig['N_dit']=uservals['N_exp']
    return detectorconfig

def get_template(uservals):
    template_name=uservals['template_name']
    try:
        print(' ')
        print("Reading FITS template spectrum: %s " %template_name)
        fits.getdata(str(template_name))
    except FileNotFoundError:
        print("FITS file not found or not valid input file")
        exit()
    hdu=fits.open(str(template_name))
    spec=hdu[0].data
    spec_header=hdu[0].header
    naxis=spec_header['naxis1']
    crval=spec_header['crval1']
    cdelt=spec_header['cdelt1']
    tunit_wav=spec_header['TUNIT1']
    respow_tpl=spec_header['R']
    template_data={}
    template_data['header']=spec_header
    temp_pix_array=np.arange(0,naxis,1)
    # These are the wave and flux arrays of the templates
    unitsok=0
    if (tunit_wav=='Angstroms'):
        wl_tpl=crval+temp_pix_array*cdelt
        fl_tpl=spec
        template_data['wave']=wl_tpl # in Angstroms
        template_data['cdelt']=cdelt
        template_data['flux']=fl_tpl # in ergs/s/cm2/A
        template_data['respow']=respow_tpl
        unitsok=1
    if (unitsok==0):
        print('ERROR: Wavelength unit expect to be Angstroms')
        exit()
    return template_data

def get_template_galsim(uservals):
    template_name=uservals['template_name']
    rv=0.0
    #template_name=uservals['template']
    try:
        print(' ')
        print("Reading FITS template spectrum: %s " %template_name)
        fits.getdata(str(template_name))
    except FileNotFoundError:
        print("FITS file not found or not valid input file")
        exit()
    hdu=fits.open(str(template_name))
    spec=hdu[0].data
    spec_header=hdu[0].header
    naxis=spec_header['naxis1']
    crval=spec_header['crval1']
    cdelt=spec_header['cdelt1']
    tunit_wav='Angstroms'
    respow_tpl=200000.00 #spec_header['R']
    template_data={}
    template_data['header']=spec_header
    temp_pix_array=np.arange(0,naxis,1)
    # This are the wave and flux arrays of the templates
    unitsok=0
    if (tunit_wav=='Angstroms'):
        wl_tpl=(crval+temp_pix_array*cdelt)*(1+rv/299792.0)
        fl_tpl=spec
        template_data['wave']=wl_tpl # in Angstroms
        template_data['cdelt']=cdelt
        template_data['flux']=fl_tpl # in ergs/s/cm2/A
        template_data['respow']=respow_tpl
        unitsok=1
    if (unitsok==0):
        print('ERROR: Wavelength unit expect to be Angstroms')
        exit()
    return template_data  

def get_RV(number_of_spectra,uservals):
    s = np.random.normal(uservals['rv'][0], uservals['rv'][1], number_of_spectra)
    return s

def get_diffraction(Lambda,airmass,atm_ref_wav):
    Lambda0=float(atm_ref_wav)
    TC=11.5			#Temperature [C]
    RH=14.5			#Relative Humidity [%]
    P=743.0			#Pressure [mbar]
    Z=np.arccos(1.0/float(airmass))*57.2958
    ZD=Z*0.0174533
    T=TC+273.16
    PS=-10474.0+116.43*T-0.43284*T**2+0.00053840*T**3
    P2=RH/100.0*PS
    P1=P-P2
    D1=P1/T*(1.0+P1*(57.90*1.0e-8-(9.3250*1.0e-4/T)+(0.25844/T**2)))
    D2=P2/T*(1.0+P2*(1.0+3.7e-4*P2)*(-2.37321e-3+(2.23366/T)-(710.792/T**2)+(7.75141e4/T**3)))
    S0=1.0/Lambda0
    S=1.0/Lambda
    N0_1=1.0E-8*((2371.34+683939.7/(130.0-S0**2)+4547.3/(38.9-S0**2))*D1+(6487.31+58.058*S0**2-0.71150*S0**4+0.08851*S0**6)*D2)
    N_1=1.0E-8*((2371.34+683939.7/(130.0-S**2)+4547.3/(38.9-S**2))*D1+(6487.31+58.058*S**2-0.71150*S**4+0.08851*S**6)*D2)
    DR=np.tan(ZD)*(N0_1-N_1)*206264.8
    return DR

def get_line_profile(filename, wave_spec, flux_spec, disp):
    profile_angs, profile_line  = np.loadtxt(filename, unpack=True)
    k_x=np.arange(0,len(profile_angs),1)
    k_x_wlr=profile_angs[0]+k_x*(profile_angs.max()-profile_angs.min())/len(k_x)
    k_x_sim = np.interp(k_x_wlr,profile_angs,profile_line)
    flux_conv=np.convolve(flux_spec,k_x_sim,mode="same") #Convolve with line profile kernel, resampled to simulation dispersion steps
    return flux_conv

def get_sky_model(sky_template, airmass_fl, folder):
    if sky_template=='eso_skycalc':
        print('Using ESO SkyCalc template for Mean Sky conditions')
        available_airmass=np.array([1.0,1.2,1.4,1.6,1.8,2.0])
        closest_airmass=available_airmass[np.argmin(np.abs(available_airmass-airmass_fl))]
        print('Skymodel selected: SkyTemplate_ESO_a'+str(closest_airmass)+'.fits')
        spec_hdu=fits.open('app/static/Skymodel/SkyTemplate_ESO_a'+str(closest_airmass)+'.fits')
        spec=spec_hdu[0].data
        trans_spec=spec_hdu[1].data
        spec_header=spec_hdu[0].header
        naxis_sf=spec_header['naxis1']
        crval_sf=spec_header['crval1']
        cdelt_sf=spec_header['cdelt1']
        temp_pix_array_sf=np.arange(0,naxis_sf,1)
        naxis_st=spec_header['naxis1']
        crval_st=spec_header['crval1']
        cdelt_st=spec_header['cdelt1']
        temp_pix_array_st=np.arange(0,naxis_st,1)
        # This are the wave and flux arrays of the sky templates
        rwl0=crval_sf+temp_pix_array_sf*cdelt_sf
        rfn0=spec
        atmwl=crval_st+temp_pix_array_st*cdelt_st
        atmtr=trans_spec
        oh_f={}
        oh_f['rwl0']=rwl0
        oh_f['rfn0']=rfn0
        oh_f['atmwl']=atmwl
        oh_f['atmtr']=atmtr
        oh_f['rdwl']=cdelt_sf
    else:
        skyfile=sky_template
        try:
            fits.getdata('app/static/user_files/' + folder + '/uploaded_sky_template.fits')
        except FileNotFoundError:
            print("FITS file not found or not valid input file")
            exit()
        spec_hdu=fits.open(str(skyfile))
        print('Skymodel selected: User upload')
        spec=spec_hdu[0].data
        trans_spec=spec_hdu[1].data
        spec_header=spec_hdu[0].header
        naxis_sf=spec_header['naxis1']
        crval_sf=spec_header['crval1']
        cdelt_sf=spec_header['cdelt1']
        temp_pix_array_sf=np.arange(0,naxis_sf,1)
        naxis_st=spec_header['naxis1']
        crval_st=spec_header['crval1']
        cdelt_st=spec_header['cdelt1']
        temp_pix_array_st=np.arange(0,naxis_st,1)
        # This are the wave and flux arrays of the sky templates
        rwl0=crval_sf+temp_pix_array_sf*cdelt_sf
        rfn0=spec
        atmwl=crval_st+temp_pix_array_st*cdelt_st
        atmtr=trans_spec
        oh_f={}
        oh_f['rwl0']=rwl0
        oh_f['rfn0']=rfn0
        oh_f['atmwl']=atmwl
        oh_f['atmtr']=atmtr
        oh_f['rdwl']=cdelt_sf
    return oh_f

def get_efficiency(bandpass,outputwl, min_wl, max_wl,instrumentconfig, uservals):
    # Add Telescope transmission
    if (uservals['telescope']=='VLT'):
        telescope_file='app/static/Inst_setup/%s'%instrumentconfig['telescope_eff']
    telescope_wav_micron, telescope_eff  = np.loadtxt(telescope_file, unpack = True)
    telescope_wav=telescope_wav_micron*1.0e04
    # Resample telescope transmission on wlgrid
    tel_eff = np.interp(outputwl,telescope_wav,telescope_eff)
    # Add transmission curve from overall instrument+telescope efficiency
    #data_dir='data_dir/Inst_setup'
    if (uservals['instrument']=='MOONS'):
        if (bandpass == "LR-RI" or bandpass == "HR-RI"):
            trans_file='app/static/Inst_setup/throughput_RI_%s.txt'%instrumentconfig['throughput'] #select if best or worst curves
        if (bandpass == "YJ"):
            trans_file='app/static/Inst_setup/throughput_YJ_%s.txt'%instrumentconfig['throughput'] #select if best or worst curves
        if (bandpass == "LR-H" or bandpass == "HR-H"):
            trans_file='app/static/Inst_setup/throughput_H_%s.txt'%instrumentconfig['throughput'] #select if best or worst curves
        weff,lreff,hreff = np.loadtxt(trans_file, unpack = True)
        eff_lr0=lreff
        eff_hr0=hreff
        eff_wl0=weff*10.0
        effok = np.where((eff_wl0 > min_wl) & (eff_wl0 < max_wl))[0] #Selecting instrument transmission in Setup
        eff_lr=eff_lr0[effok]
        eff_hr=eff_hr0[effok]
        eff_wl=eff_wl0[effok]
        # Resample instrument efficiency on wlgrid and detector QE (updated to detector FDR)
        if (uservals['moons_mode'] == "HR") or (uservals['moons_mode'] == "HR-sim"):
            eff_ins = np.interp(outputwl,eff_wl,eff_hr)
            QE_detector=np.interp(outputwl,instrumentconfig['QE_wav']*1.e4,instrumentconfig['QE_eff'])
            eff = QE_detector/100.0*eff_ins*tel_eff
            response= eff/np.max(eff) #QE_detector/100.0*eff/np.max(QE_detector/100.0*eff)
        if (uservals['moons_mode'] == "LR") or (uservals['moons_mode'] == "LR-sim"):
            eff_ins = np.interp(outputwl,eff_wl,eff_lr)
            QE_detector=np.interp(outputwl,instrumentconfig['QE_wav']*1.e4,instrumentconfig['QE_eff'])
            eff = QE_detector/100.0*eff_ins*tel_eff
            response=eff/np.max(eff) #QE_detector/100.0*eff/np.max(QE_detector/100.0*eff)
    print('Mean total efficiency (Telescope+Instrument+Detector): ',str(round(np.mean(eff),2)))
    print(' ')
    return eff,response,eff_ins,tel_eff,QE_detector

def get_sky_spectrum(uservals,instrumentconfig,telescopeconfig, min_wl, max_wl, folder):
    ####################################################################
    #obtain Sky spectrum from template (either provided or ESO sky_calc)
    sky_data={}
    oh_f=get_sky_model(uservals['sky_template'], uservals['airmass_fl'], folder)
    sky_data['rdwl']=oh_f['rdwl']
    rwl0=oh_f['rwl0'] # wavelength of sky model in Angstroms
    rfn0=oh_f['rfn0']*math.pi*(telescopeconfig['t_aperture']*1.e2/2.0)**2*sky_data['rdwl'] #Photons/s/arcsec2 per pixel of the OH grid
    atmtr0=oh_f['atmtr'] # atmospheric transmission
    atmwl0=oh_f['atmwl']
    iok_sky = np.where((rwl0 > min_wl) & (rwl0 < max_wl))[0]
    iok_atm = np.where((atmwl0 > min_wl) & (atmwl0 < max_wl))[0]
    rfn = rfn0[iok_sky] # flux of OH spectrum
    sky_data['rwl'] = rwl0[iok_sky] # wave of OH spectrum
    sky_data['atmwl']=atmwl0[iok_atm]
    sky_data['atmtr']=atmtr0[iok_atm]
    sky_data['rfn_sky'] = (rfn)*math.pi*(instrumentconfig['sky_aperture']/2.0)**2 # Photons/s/pix of the OH spectrum ALONE
    ####################################################################
    return sky_data

def seeing_to_ImageQ(seeing,cen_wav,Lambda,airmass_fl):
    r_0=0.100/seeing*(cen_wav/500.0)**(1.2)*airmass_fl**(-0.6)
    F_kolb=-0.981644
    fwhm_atm=seeing*airmass_fl**(0.6)*(cen_wav/500.0)**(-0.2)*np.sqrt(1.0+F_kolb*2.183*(r_0/46.0)**(0.356))
    fwhm_tel=0.000212*(cen_wav/8.2)
    fwhm_iq=np.sqrt(fwhm_tel**2+fwhm_atm**2)
    print(" ")
    print('Expected Image Quality in selected band:', round(fwhm_iq,2))
    print(" ")
    r_0=0.100/seeing*(Lambda/500.0)**(1.2)*airmass_fl**(-0.6)
    F_kolb=-0.981644
    fwhm_atm=seeing*airmass_fl**(0.6)*(Lambda/500.0)**(-0.2)*np.sqrt(1.0+F_kolb*2.183*(r_0/46.0)**(0.356))
    fwhm_tel=0.000212*(Lambda/8.2)
    fwhm_iq_arr=np.sqrt(fwhm_tel**2+fwhm_atm**2)
    return fwhm_iq,fwhm_iq_arr

def get_saturation(count_level,saturation_level):
    check_level=np.where(count_level>saturation_level)
    if (np.size(check_level)>0):
        return True
    else:
        return False

def get_PeakIntensity(spec,npix_y,DK_perpix):
    spec2d_central=np.zeros([int(npix_y)*2,np.size(spec)])
    central_pix=int(npix_y-1)
    spec2d_central[central_pix]=spec
    spec2d=ndimage.gaussian_filter1d(spec2d_central,npix_y/2.0/2.355,axis=0)
    peak_intensity=spec2d[central_pix]+DK_perpix
    return peak_intensity

def get_dispaxis(instrumentconfig, telescopeconfig, detectorconfig):
    ####################################################################
    #Set dispersion axis
    pixel_data={}
    pixel_data['cen_wav']=(instrumentconfig['wlr'][1]+instrumentconfig['wlr'][0])/2.0*1.0e3
    pixel_data['wav_range_length']=(instrumentconfig['wlr'][1]-instrumentconfig['wlr'][0])*1.0e3
    pix_arr=np.arange(0,detectorconfig['npix'],1)
    pixel_data['outputwl'] = instrumentconfig['wlr'][0]*1.0e4+pix_arr*detectorconfig['disp']
    ####################################################################
    return pixel_data

def get_respow(instrumentconfig, telescopeconfig, template_data):
    #Set central wavelength in Angstrom
    cen_wav_a=(instrumentconfig['wlr'][1]+instrumentconfig['wlr'][0])/2.0*1.0e4
    #FWHM for kernel
    fwhm_kernel=np.sqrt((cen_wav_a/instrumentconfig['resolution'])**2-(cen_wav_a/template_data['respow'])**2)
    respow_kernel=cen_wav_a/fwhm_kernel
    instrumentconfig['respow_kernel']=respow_kernel # in R
    instrumentconfig['fwhm_kernel']=fwhm_kernel # in angstroms
    ####################################################################
    return template_data, instrumentconfig

def conv_sky(sky_data,min_wl,max_wl, template_data, uservals,pixel_data, instrumentconfig, detectorconfig):
    #respow_kernel=instrumentconfig['respow_kernel']
    #outputwl=pixel_data['outputwl']
    #transmission:
    sigma=instrumentconfig['fwhm_kernel']/2.355/sky_data['rdwl']
    atminterp_res=ndimage.gaussian_filter1d(sky_data['atmtr'], sigma)
    #atminterp_res, fwhm = pyasl.instrBroadGaussFast(atmwl, atmtr, respow_kernel,edgeHandling="firstlast", fullout=True)
    atminterp = np.interp(pixel_data['outputwl'],sky_data['atmwl'],atminterp_res)# resample atmospheric transmission to detector pixels

    #sky emission:
    sp_conv_sky_res=ndimage.gaussian_filter1d(sky_data['rfn_sky'], sigma)
    #sp_conv_sky_res, fwhm = pyasl.instrBroadGaussFast(rwl, rfn_sky, respow_kernel,edgeHandling="firstlast", fullout=True)
    sp_det_sky = np.interp(pixel_data['outputwl'],sky_data['rwl'],sp_conv_sky_res)
    inband = np.where((sky_data['rwl'] >= pixel_data['outputwl'][0]) & (sky_data['rwl'] <= pixel_data['outputwl'][detectorconfig['npix']-1]))[0]
    sp_conv_sky_sum=sp_conv_sky_res[inband].sum()
    sp_det_sky_sum=sp_det_sky.sum()
    renorm_sky=sp_conv_sky_sum/sp_det_sky_sum
    sp_det_sky_rn = sp_det_sky*renorm_sky # for sky only

    if (uservals['set_line_profile']=='YES'):
        line_profile_file='IP_HIRES_m68_wave17834.1A.txt' # this needs to change to the MOONS LSF!
        sp_det_sky_rn=get_line_profile(line_profile_file,pixel_data['outputwl'],sp_det_sky_rn,detectorconfig['disp'])

    sky_data['atminterp']=atminterp
    sky_data['sp_det_sky_rn']=sp_det_sky_rn
    return sky_data

def write_out_file(x_print, y_print, name_x, name_y, sens_out_file):
    name_x=['# '+name_x]
    name_y=[name_y]
    with open('app/static/user_files/'+sens_out_file,'w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(zip(name_x,name_y))
        writer.writerows(zip(x_print,y_print))

def apply_reddening(uservals,templ_phot_fl,wl_tpl, config):
    print('Applying reddening normalisation')
    #coeff_r2_1=[1.410,0.764,0.502]
    coeff_r3_1=np.array([1.516,0.709,0.449])
    wl_ref_array=np.array([0.797,1.22,1.63])
    a_lambda=coeff_r3_1*uservals['reddening'] # A_lambda=R_lambda x E(B-V) with R_l from Schafly et al.
    # calculate the relative magnitude difference from reddening with respect to reference band
    # then calculate the normalised flux ratios in the bands red_flux_shape=F_I:F_J:F_H with the reference one always in 1.

    if uservals['filter']=='I':
        mag_reldif=a_lambda
        red_flux_ratios=(10.0**(-(mag_reldif[0]+57.5)/2.5)/(wl_ref_array[0]*1.0e4))/(10.0**(-(mag_reldif+57.5)/2.5)/(wl_ref_array*1.0e4)) # in flux units, ignoring the constants.

    if uservals['filter']=='J':
        mag_reldif=a_lambda
        red_flux_ratios=(10.0**(-(mag_reldif[1]+57.5)/2.5)/(wl_ref_array[1]*1.0e4))/(10.0**(-(mag_reldif+57.5)/2.5)/(wl_ref_array*1.0e4)) # in flux units, ignoring the constants.

    if uservals['filter']=='H':
        mag_reldif=a_lambda
        red_flux_ratios=(10.0**(-(mag_reldif[2]+57.5)/2.5)/(wl_ref_array[2]*1.0e4))/(10.0**(-(mag_reldif+57.5)/2.5)/(wl_ref_array*1.0e4)) # in flux units, ignoring the constants.
    rfl_factor=np.interp(wl_tpl,wl_ref_array*1.e4,red_flux_ratios)
    return templ_phot_fl*rfl_factor

def setup_template(template_data, uservals, telescopeconfig, config):
    ####################################################################
    #Prepare the input template
    fl_tpl=template_data['flux']
    wl_tpl=template_data['wave']
    # normalising the template source spectrum to the magnitude
    i = np.where(wl_tpl >= uservals['templ_wl_norm']*1.0e4)[0]
    templ_fl_ref = fl_tpl[i[0]]
#    templ_wl_ref = wl_tpl[i[0]]
    fl_tpl_normalized_to_one = fl_tpl/templ_fl_ref
    # template photon flux per A in cgs assuming a telescope effective diameter
    templ_phot_fl = fl_tpl_normalized_to_one*3.63e3*10.0**(-(uservals['ab']+57.5)/2.5)*math.pi*(telescopeconfig['t_aperture']*1.0e2/2.0)**2/(uservals['templ_wl_norm']*1.0e4*6.626e-27)
    if uservals['reddening']>0:
        templ_phot_fl=apply_reddening(uservals,templ_phot_fl,wl_tpl,config)
    # template photon flux per PER SPECTRAL PIXEL OF THE TEMPLATE
    templ_phot_fl_pix = templ_phot_fl*template_data['cdelt']
    template_norm={}
    template_norm['wl_tpl']=wl_tpl
    template_norm['fl_tpl']=templ_phot_fl_pix
    template_norm['flux_at_ref']=templ_fl_ref
    return template_norm
    ####################################################################

def get_fibreloss(iq_fwhm,adisp, instrumentconfig):
    adc_lookupTable='app/static/Inst_setup/%s'%instrumentconfig['adc_table']
    adc_lt=fits.open(adc_lookupTable)
    ad=adc_lt[1].header['CRVAL1']+np.arange(0,adc_lt[1].header['NAXIS1'])*adc_lt[1].header['CDELT1']
    iq=adc_lt[1].header['CRVAL2']+np.arange(0,adc_lt[1].header['NAXIS2'])*adc_lt[1].header['CDELT2']
    F_loss=adc_lt[1].data
    f_frac=iq_fwhm*1.0
    f=interpolate.interp2d(ad, iq, F_loss, kind='cubic')
    for num,value in enumerate(iq_fwhm):
        f_frac[num]=f(adisp[num],iq_fwhm[num])[0]
    return(f_frac)

def make_simulation(exp_type,bandpass, pixel_data,sky_data,template_norm, template_data, uservals, detectorconfig, telescopeconfig, instrumentconfig, min_wl, max_wl, folder):#template_data,resolution,wlr,t_aperture,sky_aperture,template_wl_norm,ab,airmass,npix,Instrument,aper_sampl,dit,eff_opt,seeing,atm_dif_ref):
#    respow_kernel=instrumentconfig['respow_kernel']
    outputwl=pixel_data['outputwl']
    cen_wav=pixel_data['cen_wav']
    sp_det_sky_rn=sky_data['sp_det_sky_rn']
    atminterp=sky_data['atminterp']
    wav_range_length=pixel_data['wav_range_length']
    npix_y=detectorconfig['ypix_fwhm']*2.0 # adopting 2xfwhm for extraction

    ####################################################################
    #Prepare the input template
    wl_tpl_in,fl_tpl_in=template_norm['wl_tpl'],template_norm['fl_tpl']
    setup_range_tpl=np.where((wl_tpl_in>min_wl) & (wl_tpl_in<max_wl))[0]
    templ_phot_fl_pix=fl_tpl_in[setup_range_tpl]
    wl_tpl=wl_tpl_in[setup_range_tpl]
    ####################################################################

    ####################################################################
    # Modelling Fibre Injection
    #Obtain Image Quality in corresponding band from seeing provided (seeing defined in zenith at 500nm)
    Lambda=np.arange(0.5,1.9,0.1)
    fwhm_iq,fwhm_iq_arr=seeing_to_ImageQ(uservals['seeing'], cen_wav, Lambda*1.0e3, uservals['airmass_fl'])
    #fib_frac=1.0 # allowing for tuning of fraction loss manually, keep as 1.0 for no additional loss
    #Calculate atmospheric difraction effect
    atm_diff=get_diffraction(Lambda, uservals['airmass_fl'], uservals['atm_dif_ref'])
    atm_diff_wl=np.interp(wl_tpl,Lambda*1.0e4,atm_diff)
    fib_frac=get_fibreloss(fwhm_iq_arr, atm_diff_wl, instrumentconfig)
    fib_frac_wl=np.interp(wl_tpl,Lambda*1.0e4,fib_frac)
    sp_conv_src = fib_frac_wl*templ_phot_fl_pix
    ####################################################################

    ####################################################################
    #resampling to detector pixels, conserving total flux
    #source:
    sigma=instrumentconfig['fwhm_kernel']/2.355/template_data['cdelt']
    sp_conv_src_res=ndimage.gaussian_filter1d(sp_conv_src, sigma)
    sp_det_src = np.interp(outputwl,wl_tpl,sp_conv_src_res)
    inband = np.where((wl_tpl >= outputwl[0]) & (wl_tpl <= outputwl[detectorconfig['npix']-1]))[0]
    sp_conv_src_sum=sp_conv_src_res[inband].sum()
    sp_det_src_sum=sp_det_src.sum()
    renorm=sp_conv_src_sum/sp_det_src_sum
    sp_det_src_rn = sp_det_src*renorm
    if (uservals['set_line_profile']=='YES'):
        line_profile_file='IP_HIRES_m68_wave17834.1A.txt'
        print(' ')
        print('Applying line profile convolution')
        print('Extracting line profile from file: ',line_profile_file)
        sp_det_src_rn=get_line_profile(line_profile_file,outputwl,sp_det_src_rn,detectorconfig['disp'])
    else:
        print('No LSF provided, adopting Gaussian kernel convolution')
        print(' ')
    ####################################################################

    ####################################################################
    #Modelling overall efficiency (not currently enabled for MOONS!)
    eff,flatfield,eff_ins,tel_eff,QE_detector=get_efficiency(bandpass,outputwl,min_wl, max_wl, instrumentconfig, uservals)
    #Calculate telescope and instrument emissivity (not critical for H band)
    ThBK=283.00
    EBK=0.00 #36.0*selector #selector matching temperature of the telescope
    ThBK_ins=283.00
    EBK_ins=0.10 #36.0*selector #selector matching temperature of the telescope
    t_em=1.4*10.0**12*EBK*np.exp(-14388.0/(outputwl/1.0e4*ThBK))/((outputwl/1.0e4)**3/instrumentconfig['resolution'])*detectorconfig['disp']
    ins_em=1.4*10.0**12*EBK_ins*np.exp(-14388.0/(outputwl/1.0e4*ThBK_ins))/((outputwl/1.0e4)**3/instrumentconfig['resolution'])*detectorconfig['disp']
    NBK_tel=(t_em)*math.pi*(instrumentconfig['sky_aperture']/2.0)**2
    NBK_ins=(ins_em)*math.pi*(instrumentconfig['sky_aperture']/2.0)**2
    NBK=NBK_tel+NBK_ins
    #print("Thermal background emissivity Telescope [e-/s]: ",str(round(np.max(NBK_tel*eff),3)))
    #print("Thermal background emissivity Instrument [e-/s]: ",str(round(np.max(NBK_ins*eff),3)))
    #sp_det_sky_rn=sp_det_sky_rn+NBK commented out for the moment.
    ####################################################################
    
    ####################################################################
    #Modelling detector influx
    sp_det = sp_det_src_rn+sp_det_sky_rn # total detector flux from both sky and source
    sp_eff=sp_det*eff*atminterp
    sp_eff_src = sp_det_src_rn*eff*atminterp
    sp_eff_sky = sp_det_sky_rn*eff#*atminterp
    # number of pixels COLLAPSED ALONG Y DIRECTION for MOS
    #sp_dk = sp_eff+DKsec*npix_y # from total insident flux
#    sp_dk_sky = sp_eff_sky+DKsec*npix_y #from only sky
    # scale by integration time
    spec_total = sp_eff*detectorconfig['dit']
    spec_2d_peak_intensity=get_PeakIntensity(spec_total,npix_y,instrumentconfig['DKsec']*detectorconfig['dit'])
    # the following is for the sky only
    spec_sky = sp_eff_sky*detectorconfig['dit']
    # the following is for the source alone
    spec_source = sp_eff_src*detectorconfig['dit']
    ####################################################################
    
    ####################################################################
    # Add stray light contribution
    stray=1.0 # 1% difusse stray light contribution as per latest optical modelling
    total_stray=spec_sky.sum()*detectorconfig['N_dit']*500.0/detectorconfig['npix']**2
    spec_stray=total_stray*float(stray)/100.0*npix_y
    ####################################################################

    ####################################################################    
    #Add residual sky subtraction (optional, if you want then change skyres to percentage):
    if ((uservals['sky_residual'] >= 0) & (uservals['sky_residual'] <= 100)):
        skyres=uservals['sky_residual']
    else:
        if (uservals['sky_residual'] == -1):
            print('Simulation with sky-subtraction OFF')
            print(' ')
            skyres=0
        else:
            print(' Not a valid sky residual value (0-100). Adopting 0.00 percent')
            print(' ')
            skyres=0
    noiseskyres=float(skyres)/100.0*spec_sky*detectorconfig['N_dit']
    ####################################################################

    ####################################################################
    #calculate all noise contributions to total    
    #detector NOISE:
    npix_x=2.7
    if uservals['strategy']=='Stare':
        noisedet=np.sqrt(npix_y*(detectorconfig['N_dit']*instrumentconfig['RON']**2+instrumentconfig['DKsec']*detectorconfig['dit']*detectorconfig['N_dit']))
        noisedet_reselem=np.sqrt(npix_y*npix_x*(detectorconfig['N_dit']*instrumentconfig['RON']**2+instrumentconfig['DKsec']*detectorconfig['dit']*detectorconfig['N_dit']))
    if uservals['strategy']=='Xswitch':
        noisedet=np.sqrt(2*(npix_y*(detectorconfig['N_dit']*instrumentconfig['RON']**2+instrumentconfig['DKsec']*detectorconfig['dit']*detectorconfig['N_dit'])))
        noisedet_reselem=np.sqrt(2*(npix_y*npix_x*(detectorconfig['N_dit']*instrumentconfig['RON']**2+instrumentconfig['DKsec']*detectorconfig['dit']*detectorconfig['N_dit'])))
    #background NOISE (including stray):
    if uservals['strategy']=='Stare':
        noiseback=np.sqrt(spec_sky*detectorconfig['N_dit']+spec_stray+noiseskyres**2)
        noiseback_reselem=np.sqrt(npix_x*(spec_sky*detectorconfig['N_dit']+spec_stray+noiseskyres**2))        
        #noiseback_nores=np.sqrt(spec_sky*detectorconfig['N_dit']+spec_stray)
    if uservals['strategy']=='Xswitch':
        noiseback=np.sqrt(2*(spec_sky*detectorconfig['N_dit']+spec_stray)+noiseskyres**2)
        noiseback_reselem=np.sqrt(2*npix_x*(spec_sky*detectorconfig['N_dit']+spec_stray)+noiseskyres**2)
        
        #noiseback_nores=np.sqrt(2*(spec_sky*detectorconfig['N_dit']+spec_stray))
    #total NOISE: comment out when necessary
    #outnoise=np.sqrt(noiseback**2+noiseskyres**2+noisedet**2+spec_source*detectorconfig['N_dit'])
    outnoise=np.sqrt(noiseback**2+noisedet**2+spec_source*detectorconfig['N_dit'])
    outnoise_reselem=np.sqrt(noiseback_reselem**2+noisedet_reselem**2+npix_x*spec_source*detectorconfig['N_dit'])
    #outnoise_nosky=np.sqrt(noiseback_nores**2+noisedet**2+spec_source*detectorconfig['N_dit'])
    #noise_sim_withsky=np.sqrt(noiseback**2+noisedet**2+spec_source*detectorconfig['N_dit'])
    #outnoise=noisetot+noiseskyres
    #outnoise = np.sqrt(outspec+outspec_sky+RON**2*npix_y*2.)
    sn_cont_all = spec_source/outnoise*detectorconfig['N_dit']
    sn_cont_all_reselem = spec_source/outnoise_reselem*detectorconfig['N_dit']*npix_x
    cent_range=np.where((outputwl >= (cen_wav-wav_range_length*0.05)*10.0) & (outputwl <= (cen_wav+wav_range_length*0.05)*10.0))[0]
    sn_central=sn_cont_all[cent_range].max()
    sn_central_reselem=sn_cont_all_reselem[cent_range].max()
    print("**** S/N at central wavelength = %.2f per pix / %.2f per res. elem. ****"%(sn_central,sn_central_reselem))
    print(" ")
    count_level=spec_2d_peak_intensity*instrumentconfig['gain']
    saturated=get_saturation(count_level,instrumentconfig['saturation_level'])

    if saturated:
        print('WARNING!!! Counts above saturation/linearity regime!!!')
    #Get figure with summary of results
    #SNR figure
    # f=plt.figure(figsize=(10,8),dpi=100)
    # ax1=f.add_subplot(221)
    # ax1.plot(outputwl/1.0e4, sn_cont_all,label='SNR per pixel')
    # ax1.axis([instrumentconfig['wlr'][0], instrumentconfig['wlr'][1], np.nanmin(sn_cont_all), np.nanmax(sn_cont_all)])
    # ax1.set_xlabel('Wavelength [um]')
    # ax1.set_ylabel('SNR (/pix)')
    # plt.legend(loc='upper right',prop={'size':8}, numpoints=1)

    # #Sky spectrum
    # ax1=f.add_subplot(222)
    # ax1.plot(outputwl/1.0e4, spec_2d_peak_intensity,label='Peak Intensity')
    # ax1.axis([instrumentconfig['wlr'][0], instrumentconfig['wlr'][1], np.nanmin(spec_2d_peak_intensity), np.nanmax(spec_2d_peak_intensity)])
    # if saturated:
    #     ax1.plot(outputwl/1.0e4,spec_2d_peak_intensity*0.0+instrumentconfig['saturation_level']/instrumentconfig['gain'],color='red',label='Saturation')
    # plt.legend(loc='upper right',prop={'size':8}, numpoints=1)
    # ax1.set_xlabel('Wavelength [um]')
    # ax1.set_ylabel('Counts (e-)')

    # #atmospheric transmission
    # ax1=f.add_subplot(223)
    # ax1.plot(outputwl/1.0e4, atminterp ,label='Atmospheric transmission')
    # ax1.axis([instrumentconfig['wlr'][0], instrumentconfig['wlr'][1],np.nanmin(atminterp), np.nanmax(atminterp)] )# np.nanmin(eff*atminterp), np.nanmax(eff*atminterp)])
    # ax1.set_xlabel('Wavelength [um]')
    # ax1.set_ylabel('Transmission fraction')
    # plt.legend(loc='lower left',prop={'size':8}, numpoints=1)

    #Simulated spectrum
    # ax1=f.add_subplot(224)
    res_noise=np.random.normal(outnoise*0.0,outnoise,size=None)
    res_sky=np.random.normal(outnoise*0.0,noiseskyres,size=None)    
    sim_spectrum={}
    if uservals['sky_residual']==-1:
        sim_spectrum_flux=spec_total*detectorconfig['N_dit']+res_noise
    else:
        sim_spectrum_flux=spec_source*detectorconfig['N_dit']+res_noise#+res_sky/np.sqrt(detectorconfig['N_dit'])

    if ((uservals['telluric']==1) & (exp_type=='target')):
        print('Telluric correction applied (error in correction set to <3%)')
        print(' ')
        # fraction corresponding to a % of correction
        telluric_percent=1.0 # fixed to a 4% telluric correction
        tell_sig=(1-atminterp)*telluric_percent/100.0 # random values between - and + telluric percent
        tell_res=np.random.normal(outnoise*0.0,tell_sig,size=None) 
        telluric=atminterp+tell_res
        sim_spectrum_flux=sim_spectrum_flux/telluric
        
    if exp_type=='flux_calib':
        # fraction corresponding to a % of correction
        telluric_percent=1.0 # fixed to a 4% telluric correction
        tell_sig=(1-atminterp)*telluric_percent/100.0 # random values between - and + telluric percent
        tell_res=np.random.normal(outnoise*0.0,tell_sig,size=None) 
        telluric=atminterp+tell_res
        sim_spectrum_flux=sim_spectrum_flux/telluric
        
    sim_spectrum['flux']=sim_spectrum_flux        
    sim_spectrum['noise']=outnoise
    sim_spectrum['model']=sp_det_src_rn
    sim_spectrum['wave']=outputwl
    sim_spectrum['cdelt']=detectorconfig['disp']
    sim_spectrum['crval']=outputwl[0]
    sim_spectrum['SNR']=sn_central
    sim_spectrum['resolving_power']=instrumentconfig['resolution']
    # ax1.plot(outputwl/1.0e4, sim_spectrum_flux,label='Sim spectrum (not flux calib)')
    # ax1.plot(outputwl/1.0e4, spec_source*detectorconfig['N_dit']/atminterp,label='Observed target (no noise)',alpha=0.6)
    # ax1.axis([instrumentconfig['wlr'][0], instrumentconfig['wlr'][1], np.nanmin(sim_spectrum_flux[(sim_spectrum_flux != -np.inf) & (sim_spectrum_flux != np.inf)]), np.nanmax(sim_spectrum_flux[(sim_spectrum_flux != -np.inf) & (sim_spectrum_flux != np.inf)])])
    # ax1.set_xlabel('Wavelength [um]')
    # ax1.set_ylabel('Counts (e-)')
    # plt.legend(loc='upper right',prop={'size':8}, numpoints=1)
    # plt.tight_layout()
    # #f.savefig("app/static/ETC_results_%s_%s.pdf"%(bandpass,exp_type), bbox_inches='tight')
    # plt.close()

    if uservals['flux_calib']==1:
        if exp_type=='flux_calib':
            x_print=outputwl/1.0e4
            y_print=sim_spectrum_flux #sim_spectrum_flux
            sens_out_file= folder + "/standard_flux_%s.txt"%bandpass
            name_x='Lambda'
            name_y='Flux'
            wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)   
            
            ratio_flux=get_standard_calibration(template_data,bandpass, folder)
            x_print=outputwl/1.0e4
            y_print=sim_spectrum_flux/ratio_flux #sim_spectrum_flux
            sens_out_file= folder + "/sim_spectrum_fluxcalibrated_%s.txt"%bandpass
            name_x='Lambda'
            name_y='Flux'
            wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)                        
            
        if exp_type=='target':   
            
            x_print=outputwl/1.0e4
            y_print=spec_2d_peak_intensity
            sens_out_file= folder + "/max_intensity_%s.txt"%bandpass
            name_x='Lambda'
            name_y='Flux'
            wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
            
            x_print=outputwl/1.0e4
            y_print=sn_cont_all
            sens_out_file= folder + "/SN/signal_to_noise_%s.txt"%bandpass
            name_x='Lambda'
            name_y='SNR'
            wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
        
            x_print=outputwl/1.0e4
            y_print=atminterp
            sens_out_file= folder + "/transmission/Transmission_%s.txt"%bandpass
            name_x='Lambda'
            name_y='Transmission'
            wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
            
            x_print=outputwl/1.0e4
            y_print=sim_spectrum['flux'] #sim_spectrum_flux
            sens_out_file= folder + "/sim_spectrum_%s.txt"%bandpass
            name_x='Lambda'
            name_y='Flux'
            wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
        
            x_print=outputwl/1.0e4
            y_print=1.0/outnoise**2 #error on the simulated spec -> inverted square of the RMS noise
            sens_out_file= folder + "/sim_spectrum_ivar_%s.txt"%bandpass
            name_x='Lambda'
            name_y='ivar'
            wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
                    
            x_print=outputwl/1.0e4
            y_print=spec_source*detectorconfig['N_dit']
            sens_out_file= folder + "/obj_spec/object_spectrum_%s.txt"%bandpass
            name_x='Lambda'
            name_y='Flux'
            wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)

    if uservals['flux_calib']!=1:

        print('test')
        
        x_print=outputwl/1.0e4
        y_print=sim_spectrum_flux #sim_spectrum_flux
        sens_out_file= folder + "/standard_flux_%s.txt"%bandpass
        name_x='Lambda'
        name_y='Flux'
        wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)   
            
        ratio_flux=get_standard_calibration(template_data,bandpass, folder)
        x_print=outputwl/1.0e4
        y_print=sim_spectrum_flux/ratio_flux #sim_spectrum_flux
        sens_out_file=folder + "/sim_spectrum_fluxcalibrated_%s.txt"%bandpass
        name_x='Lambda'
        name_y='Flux'
        wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)                        
            
        x_print=outputwl/1.0e4
        y_print=spec_2d_peak_intensity
        sens_out_file=folder + "/max_intensity_%s.txt"%bandpass
        name_x='Lambda'
        name_y='Flux'
        wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
            
        x_print=outputwl/1.0e4
        y_print=sn_cont_all
        sens_out_file=folder + "/SN/signal_to_noise_%s.txt"%bandpass
        name_x='Lambda'
        name_y='SNR'
        wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
        
        x_print=outputwl/1.0e4
        y_print=atminterp
        sens_out_file=folder + "/transmission/Transmission_%s.txt"%bandpass
        name_x='Lambda'
        name_y='Transmission'
        wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
            
        x_print=outputwl/1.0e4
        y_print=sim_spectrum['flux'] #sim_spectrum_flux
        sens_out_file=folder + "/sim_spectrum_%s.txt"%bandpass
        name_x='Lambda'
        name_y='Flux'
        wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
        
        x_print=outputwl/1.0e4
        y_print=1.0/outnoise**2 #error on the simulated spec -> inverted square of the RMS noise
        sens_out_file=folder + "/sim_spectrum_ivar_%s.txt"%bandpass
        name_x='Lambda'
        name_y='ivar'
        wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)
                    
        x_print=outputwl/1.0e4
        y_print=spec_source*detectorconfig['N_dit']
        sens_out_file=folder + "/obj_spec/object_spectrum_%s.txt"%bandpass
        name_x='Lambda'
        name_y='Flux'
        wrotefile=write_out_file(x_print, y_print, name_x, name_y, sens_out_file)

    return sn_central

def get_standard_calibration(template_data, bandpass, folder):

    standard_obs=ascii.read('app/static/user_files/' + folder +'/standard_flux_%s.txt'%bandpass)
    standard_obs_wave=standard_obs['Lambda']
    standard_obs_flux=standard_obs['Flux']    
    standard_tab_wave=template_data['wave']
    standard_tab_flux=template_data['flux']*template_data['cdelt'] # flux per template pixel
    standard_tab_flux_inwav=np.interp(standard_obs_wave,standard_tab_wave,standard_tab_flux)/(standard_obs_wave[1]-standard_obs_wave[0]) 
    ratio=standard_obs_flux/standard_tab_flux_inwav
    ratio_fit=np.polyfit(standard_obs_wave,ratio,3)
    return ratio_fit[0]*standard_obs_wave**3+ratio_fit[1]*standard_obs_wave**2+ratio_fit[2]*standard_obs_wave+ratio_fit[3]
    
def get_minmax(instrumentconfig):
    tol = (instrumentconfig['wlr'][1]-instrumentconfig['wlr'][0])*0.05
    min_wl=(instrumentconfig['wlr'][0]-tol)*1.0e4
    max_wl=(instrumentconfig['wlr'][1]+tol)*1.0e4
    return min_wl, max_wl

def setup_magnitude(uservals):
    if uservals['mag_system']=='AB':
        print('magnitude in AB system')
        if uservals['filter']=='I':
            uservals['ab']=uservals['magnitude']
            uservals['templ_wl_norm']=0.797
        if uservals['filter']=='J':
            uservals['ab']=uservals['magnitude']
            uservals['templ_wl_norm']=1.22
        if uservals['filter']=='H':
            uservals['ab']=uservals['magnitude']
            uservals['templ_wl_norm']=1.63
    if uservals['mag_system']=='Vega':
        print('magnitude in Vega system')
        if uservals['filter']=='I':
            uservals['ab']=uservals['magnitude']+0.45
            uservals['templ_wl_norm']=0.797
        if uservals['filter']=='J':
            uservals['ab']=uservals['magnitude']+0.91
            uservals['templ_wl_norm']=1.22
        if uservals['filter']=='H':
            uservals['ab']=uservals['magnitude']+1.39
            uservals['templ_wl_norm']=1.63
    return uservals

def make_exposure(exp_type,uservals, config, folder):
    if uservals['moons_mode']=='HR':
        bands=['HR-RI','YJ','HR-H']
    if uservals['moons_mode']=='LR':
        bands=['LR-RI','YJ','LR-H']
    if uservals['moons_mode']=='HR-sim':
        if uservals['filter']=='I':
            bands=['HR-RI']
        if uservals['filter']=='J':
            bands=['YJ']
        if uservals['filter']=='H':
            bands=['HR-H']
    if uservals['moons_mode']=='LR-sim':
        if uservals['filter']=='I':
            bands=['LR-RI']
        if uservals['filter']=='J':
            bands=['YJ']
        if uservals['filter']=='H':
            bands=['LR-H']
    telescopeconfig=set_telescope(uservals)
    uservals=setup_magnitude(uservals)
    if uservals['moons_mode']=='HR-sim' or uservals['moons_mode']=='LR-sim':
        template_data=get_template_galsim(uservals)
    else:
        template_data=get_template(uservals)

    template_norm=setup_template(template_data, uservals, telescopeconfig, config)
    SNR={}
    for bandpass in bands:
        if bandpass=='HR-RI' or bandpass=='LR-RI':
            device='CCD'
        else:
            device='IR'
        instrumentconfig=setup_moons(uservals,config,bandpass,device)
        min_wl, max_wl=get_minmax(instrumentconfig)
        detectorconfig=set_detector(telescopeconfig,uservals,instrumentconfig,device)
        sky_data=get_sky_spectrum(uservals,instrumentconfig,telescopeconfig, min_wl, max_wl, folder)
        pixel_data=get_dispaxis(instrumentconfig,telescopeconfig,detectorconfig)
        number_of_pixels=np.size(pixel_data['outputwl'])
        template_data,instrumentconfig=get_respow(instrumentconfig,telescopeconfig,template_data)
        sky_data=conv_sky(sky_data, min_wl, max_wl, template_data, uservals, pixel_data, instrumentconfig, detectorconfig)
        SNR=make_simulation(exp_type,bandpass, pixel_data,sky_data, template_norm, template_data, uservals, detectorconfig, telescopeconfig, instrumentconfig,min_wl, max_wl, folder) # resolution,wlr,t_aperture,sky_aperture,template_wl_norm,ab,airmass,npix,Instrument,ypix_fwhm,dit,eff_opt,seeing,atm_dif_ref)
        print('Calculation completed for %s'%bandpass)


def do_etc_calc(folder, uservals):
    config=getConfig('app/static/Inst_setup/ConfigFile.ini')
    params= getParams('app/static/user_files/' + folder +'/ParamFile.ini')   
    exp_type='target'
    uservals['airmass_fl'] = uservals['airmass']
    make_exposure(exp_type,uservals,config,folder)

if __name__ == "__main__":
    config=getConfig('Inst_setup/ConfigFile.ini')
    params= getParams('ParamFile.ini')

    uservals=get_input(params)    
    exp_type='target'
    make_exposure(exp_type,uservals,params,config)
    if uservals['flux_calib']==1:
        exp_type='flux_calib'
        #target
        uservals['template_name']='Inst_setup/standard.fits'
        uservals['filter']='H'
        uservals['magnitude']=8.0
        uservals['mag_system']='Vega'
        #instrument
        uservals['strategy']='Stare'
        uservals['exptime']=15.0
        uservals['dit']=5
        uservals['N_exp']=3
        make_exposure(exp_type,uservals,params,config)
    
    



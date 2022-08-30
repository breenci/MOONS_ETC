from os import system
from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, SubmitField, FloatField, BooleanField, HiddenField
from wtforms.validators import DataRequired, NumberRange


class ETC_form(FlaskForm):
    # target variables
    template_name = SelectField('Template', choices=[('app/static/Example_spectra/constant_in_wav.fits', 'constant_in_wav'), ('app/static/Example_spectra/input_stellar_template.fits', 'stellar')], validators=[DataRequired()])
    magnitude = FloatField('Magnitude', validators=[DataRequired()])
    filter = SelectField('Filter', choices=['H', 'J', 'I'], validators=[DataRequired()])
    mag_system = SelectField('System', choices=['AB', 'Vega'])
    reddening = FloatField('Reddening E(B-V)')
    displacement = FloatField('Fibre to Target Displacement')

    # instrument variables
    moons_mode = SelectField('Mode', choices=['HR', 'LR'], validators=[DataRequired()])
    strategy = SelectField('Strategy', choices=['Xswitch', 'Stare'], validators=[DataRequired()])
    atm_dif_ref = FloatField('Ref. Wavelength for AD Correction', validators=[DataRequired(), NumberRange(min=0.57, max=1.8)])
    exptime = IntegerField('Exposure Time (s)')
    N_exp = IntegerField('No. Exposures')
    dit = IntegerField('NDIT')

    #conditions
    seeing = FloatField('Seeing (arcsec)', validators=[DataRequired()])
    airmass = FloatField('Airmass',validators=[DataRequired()])

    #simulation
    sky_template = SelectField('Sky Template', choices=['eso_skycalc'], validators=[DataRequired()])
    sky_residual = IntegerField('Additional Sky Residual (%)',validators=[DataRequired(), NumberRange(min=0, max=100)])
    telluric = IntegerField('Telluric Correction')
    flux_calib = SelectField('Flux Calib', choices=[0,1])

    # hardcoded values
    telescope = HiddenField('Telescope', default='VLT')
    instrument = HiddenField('Instrument', default='MOONS')
    set_line_profile = HiddenField('Set Line Profile', default='NO')

    #plot selection
    trans = BooleanField('Transmission')
    sn = BooleanField('Signal-to-Noise')
    obj_spec = BooleanField('Object Spectrum')

    #control
    submit = SubmitField('Submit')
    


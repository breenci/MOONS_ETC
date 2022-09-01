from email.policy import default
from os import system
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import SelectField, IntegerField, SubmitField, FloatField, BooleanField, HiddenField, FormField, FieldList
from wtforms.validators import DataRequired, NumberRange


class ETC_form(FlaskForm):
    # target variables
    template_name = SelectField('Template (Defaults)', choices=[('app/static/Example_spectra/constant_in_wav.fits', 'constant_in_wav'), 
                    ('app/static/Example_spectra/input_stellar_template.fits', 'stellar')], validators=[DataRequired()])
    upload_template = FileField('Upload Template (overides default)')

    magnitude = FloatField('Magnitude', validators=[DataRequired()])
    filter = SelectField('Magnitude Band', choices=['H', 'J', 'I'], validators=[DataRequired()])
    mag_system = SelectField('Photometric System', choices=['AB', 'Vega'])
    reddening = FloatField('Reddening E(B-V)')
    displacement = FloatField('Fibre-to-Target Displacement (arcsec)', default=0)

    # instrument variables
    moons_mode = SelectField('Mode', choices=['HR', 'LR'], validators=[DataRequired()])
    strategy = SelectField('Strategy', choices=['Xswitch', 'Stare'], validators=[DataRequired()])
    atm_dif_ref = FloatField('Ref. Wavelength for AD Correction', validators=[DataRequired(), NumberRange(min=0.57, max=1.8)])
    exptime = FloatField('Exposure Time (s)')
    N_exp = IntegerField('No. Exposures')
    dit = IntegerField('NDIT')

    #conditions
    seeing = FloatField('Seeing (arcsec)', validators=[DataRequired(), NumberRange(min=0, max=4)])
    airmass = FloatField('Airmass',validators=[DataRequired(), NumberRange(min=1)])

    #simulation
    sky_template = SelectField('Sky Template', choices=[('eso_skycalc', 'eso sky calc')], validators=[DataRequired()])
    sky_residual = IntegerField('Additional Sky Residual (%)',validators=[DataRequired(), NumberRange(min=0, max=100)])
    telluric = SelectField('Telluric Correction', choices=[(0, 'No'), (1, 'Yes')])
    flux_calib = SelectField('Flux Calib', choices=[(0, 'No'), (1, 'Yes')])

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
    


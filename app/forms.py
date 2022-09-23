from email.policy import default
from os import system
from typing import Optional
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SelectField, IntegerField, SubmitField, FloatField, BooleanField, HiddenField, FormField, FieldList, RadioField
from wtforms.validators import DataRequired, NumberRange, ValidationError, Optional


class UploadRequired:

    def __init__(self, fieldname, message=None):
        self.fieldname = fieldname
        self.message = message

    def __call__(self, form, field):
        try:
            other = form[self.fieldname]
        except KeyError as exc:
            raise ValidationError(
                field.gettext("Invalid field name '%s'.") % self.fieldname
            ) from exc
        if other.data == False:
            print('Not selected')
            return
        elif field.data:
            print('There is data')
            return

        message = self.message
        if message is None:
            message = field.gettext("Upload file or use a default")

        raise ValidationError(message)


class ETC_form(FlaskForm):
    # target variables
    template_type = BooleanField('')
    template_name = SelectField('Template', choices=[('app/static/Example_spectra/constant_in_wav.fits', 'constant_in_wav'), 
                    ('app/static/Example_spectra/input_stellar_template.fits', 'stellar')], validators=[Optional()])
    upload_template = FileField('', validators=[ UploadRequired('template_type'), FileAllowed(['.fits'])])

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

    #control
    submit = SubmitField('Submit')




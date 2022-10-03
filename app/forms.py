from email.policy import default
from os import system
from typing import Optional
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SelectField, IntegerField, SubmitField, FloatField, BooleanField, HiddenField
from wtforms.validators import DataRequired, NumberRange, ValidationError, Optional, InputRequired


class RequiredIfTrue:
    """Validator to require input if another field is used"""

    def __init__(self, fieldname, message=None):
        self.fieldname = fieldname
        self.message = message

    def __call__(self, form, field):
        # make sure other field is a valid field
        try:
            other = form[self.fieldname]
        except KeyError as exc:
            raise ValidationError(
                field.gettext("Invalid field name '%s'.") % self.fieldname
            ) from exc
        # return if other field is not used
        if other.data == False:
            return
        # elif other field is used return if input is given
        elif field.data:
            return

        # otherwise raise an error
        message = self.message
        if message is None:
            message = field.gettext("Upload file or use a default")

        raise ValidationError(message)


class ETC_form(FlaskForm):
    # target variables
    template_type = BooleanField('')
    template_name = SelectField('Template', choices=[('app/static/Example_spectra/constant_in_wav.fits', 'constant_in_wav'), 
                    ('app/static/Example_spectra/input_stellar_template.fits', 'stellar')])
    upload_template = FileField('', validators=[RequiredIfTrue('template_type'), FileAllowed(['fits'])])

    magnitude = FloatField('Magnitude', default=10,validators=[InputRequired()])
    filter = SelectField('Magnitude Band', choices=['H', 'J', 'I'])
    mag_system = SelectField('Photometric System', choices=['AB', 'Vega'])
    reddening = FloatField('Reddening E(B-V)', default=1, validators=[InputRequired()])
    displacement = FloatField('Fibre-to-Target Displacement (arcsec)', default=0, validators=[InputRequired()])

    # instrument variables
    moons_mode = SelectField('Mode', choices=['HR', 'LR'])
    strategy = SelectField('Strategy', choices=['Xswitch', 'Stare'])
    atm_dif_ref = FloatField('Ref. Wavelength for AD Correction', default=1.2, validators=[InputRequired(), NumberRange(min=0.57, max=1.8)])
    exptime = FloatField('Exposure Time (s)', default=300, validators=[InputRequired(), NumberRange(min=0)])
    N_exp = IntegerField('No. Exposures', default=1, validators=[InputRequired(), NumberRange(min=0)])
    dit = IntegerField('NDIT', default=1, validators=[InputRequired(), NumberRange(min=0)])

    #conditions
    seeing = FloatField('Seeing (arcsec)', default=2, validators=[InputRequired(), NumberRange(min=0, max=4)])
    airmass = FloatField('Airmass', default=1, validators=[InputRequired(), NumberRange(min=1)])

    #simulation
    sky_type = BooleanField('')
    sky_template = SelectField('Sky Template', choices=[('eso_skycalc', 'eso sky calc')], validators=[Optional()])
    upload_sky = FileField('', validators=[RequiredIfTrue('sky_type')])
    sky_residual = IntegerField('Additional Sky Residual (%)', default=0, validators=[InputRequired(), NumberRange(min=0, max=100)])
    telluric = SelectField('Telluric Correction', choices=[(0, 'No'), (1, 'Yes')])
    flux_calib = SelectField('Flux Calib', choices=[(0, 'No'), (1, 'Yes')])

    # hardcoded values
    telescope = HiddenField('Telescope', default='VLT')
    instrument = HiddenField('Instrument', default='MOONS')
    set_line_profile = HiddenField('Set Line Profile', default='NO')

    #control
    submit = SubmitField('Submit')




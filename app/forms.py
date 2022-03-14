from os import system
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, SubmitField, FloatField
from wtforms.validators import DataRequired

class ETC_form(FlaskForm):
    # target variables
    template_name = SelectField('Template', choices=[('Example_spectra/constant_in_wav.fits', 'constant_in_wav'), ('Example_spectra/input_stellar_template.fits', 'stellar')], validators=[DataRequired()])
    magnitude = FloatField('Magnitude', validators=[DataRequired()])
    filter = SelectField('Filter', choices=['H', 'J', 'I'], validators=[DataRequired()])
    system = SelectField('System', choices=['AB', 'Vega'])
    reddening = FloatField('Reddening')
    displacement = FloatField('Displacement')

    # instrument variables
    mode = SelectField('Mode', choices=['HR', 'LR'],validators=[DataRequired()])
    strategy = SelectField('Strategy', choices=['Xswitch', 'Stare'], validators=[DataRequired()])
    adc_refwave = FloatField('ADC_refwave')
    exp_time = IntegerField('Exposure Time')
    nexp = IntegerField('No. Exposures')
    dit = IntegerField('dit')

    #conditions
    seeing = FloatField('Seeing', validators=[DataRequired()])
    airmass = FloatField('Airmass',validators=[DataRequired()])

    #simulation
    sky_temp = SelectField('Sky Template', choices=['eso_skycalc'], validators=[DataRequired()])
    sky_res = IntegerField('Sky Residual',validators=[DataRequired()])
    tell = IntegerField('Telluric')
    flux_cal = FloatField('Flux Calib')

    #control
    submit = SubmitField('Submit')
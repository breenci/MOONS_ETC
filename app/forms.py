from os import system
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, SubmitField, FloatField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    # target variables
    template_name = StringField('Template', validators=[DataRequired()])
    magnitude = FloatField('Magnitude', validators=[DataRequired()])
    filter = SelectField('Filter', choices=['H', 'YJ'], validators=[DataRequired()])
    system = SelectField('System', choices=['AB', 'BA'])
    reddening = FloatField('Reddening')
    displacement = FloatField('Displacement')

    # instrument variables
    mode = SelectField('Mode', choices=['HR', 'LR'],validators=[DataRequired()])
    strategy = SelectField('Strategy', choices=['Xswitch', 'STARE'], validators=[DataRequired()])
    adc_refwave = FloatField('ADC_refwave')
    exp_time = FloatField('Exposure Time')
    nexp = IntegerField('No. Exposures')
    dit = IntegerField('dit')

    #conditions
    seeing = FloatField('Seeing', validators=[DataRequired()])
    airmass = FloatField('Airmass',validators=[DataRequired()])

    #simulation
    sky_temp = StringField('Sky Template',validators=[DataRequired()])
    sky_res = IntegerField('Sky Residual',validators=[DataRequired()])
    tell = IntegerField('Telluric')
    flux_cal = FloatField('Flux Calib')

    #control
    submit = SubmitField('Submit')
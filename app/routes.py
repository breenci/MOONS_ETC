from app import app
from flask import render_template
import configparser

from app.forms import LoginForm

@app.route('/', methods=['GET', 'POST'])
def index():
    form = LoginForm()
    if form.validate_on_submit():
        config = configparser.ConfigParser()
        config.read('ParamFile.ini')
        config.set('target', 'template_name', form.template_name.data)
        config.set('target', 'magnitude', str(form.magnitude.data))
        config.set('instrument', 'moons_mode', form.mode.data)
        config.set('instrument', 'strategy', form.strategy.data)
        config.set('conditions', 'seeing', str(form.seeing.data))
        config.set('conditions', 'airmass', str(form.airmass.data))
        config.set('simulation', 'sky_template', form.sky_temp.data)
        config.set('simulation', 'sky_residual', str(form.sky_res.data))

        with open('ParamFile.ini', 'w') as configfile:
            config.write(configfile)

    return render_template('form.html', form=form)

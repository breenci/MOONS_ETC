from app import app
from flask import render_template, send_file
import configparser

from app.forms import ETC_form

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ETC_form(template_name = 'test1')
    if form.validate_on_submit():
        config = configparser.ConfigParser()
        config.read('ParamFile.ini')

        config.set('target', 'template_name', form.template_name.data)
        config.set('target', 'magnitude', str(form.magnitude.data))
        config.set('target', 'filter', form.filter.data)
        config.set('target', 'system', form.system.data)
        config.set('target', 'reddening', str(form.reddening.data))
        config.set('target', 'displacement', str(form.displacement.data))

        config.set('instrument', 'moons_mode', form.mode.data)
        config.set('instrument', 'strategy', form.strategy.data)
        config.set('instrument', 'adc_refwave', str(form.adc_refwave.data))
        config.set('instrument', 'exp_time', str(form.exp_time.data))
        config.set('instrument', 'nexp', str(form.nexp.data))
        config.set('instrument', 'dit', str(form.dit.data))

        config.set('conditions', 'seeing', str(form.seeing.data))
        config.set('conditions', 'airmass', str(form.airmass.data))

        config.set('simulation', 'sky_template', form.sky_temp.data)
        config.set('simulation', 'sky_residual', str(form.sky_res.data))
        config.set('simulation', 'telluric', str(form.tell.data))
        config.set('simulation', 'flux_calib', str(form.flux_cal.data))

        with open('ParamFile.ini', 'w') as configfile:
            config.write(configfile)

    return render_template('form.html', form=form)


@app.route('/getTxtFile')
def download():
    return send_file('/Users/ciaran.breen/Documents/MOONS_ETC/app/outputs/signal_to_noise_YJ.txt', as_attachment=True)
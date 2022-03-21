from fileinput import filename
from app import app
from flask import redirect, render_template, send_file, send_from_directory
import configparser
from moons_etc_backend import do_etc_calc
import glob
import os
import numpy as np

from app.make_plots import plot_folder
from app.forms import ETC_form

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ETC_form()
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
        config.set('instrument', 'exptime', str(form.exp_time.data))
        config.set('instrument', 'nexp', str(form.nexp.data))
        config.set('instrument', 'dit', str(form.dit.data))

        config.set('conditions', 'seeing', str(form.seeing.data))
        config.set('conditions', 'airmass', str(form.airmass.data))

        config.set('simulation', 'sky_template', form.sky_temp.data)
        config.set('simulation', 'sky_residual', str(form.sky_res.data))
        config.set('simulation', 'telluric', str(form.tell.data))
        config.set('simulation', 'flux_calib', str(form.flux_cal.data))

        plot_list = np.array([form.sn.data, form.trans.data, form.obj_spec.data])
        
        with open('ParamFile.ini', 'w') as configfile:
            config.write(configfile)
        
        # for file in glob.glob('app/static/*'):
        #     os.remove(file)

        np.savetxt('app/static/plot_selection/plot_selection.txt', plot_list)

        do_etc_calc()

        plot_folder('app/static/SN')
        plot_folder('app/static/obj_spec')
        plot_folder('app/static/transmission')

        return redirect('/results')

    return render_template('form.html', form=form)


@app.route('/results', methods=['GET'])
def results():
    is_plot = np.loadtxt('app/static/plot_selection/plot_selection.txt')
    return render_template('results.html', is_plot=is_plot)


@app.route('/get-txt/<path:filename>')
def send_report(filename):
    return send_from_directory('static', filename, as_attachment=True)


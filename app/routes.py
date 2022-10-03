#from urllib import request
from app import app
from flask import redirect, render_template, send_from_directory, request
from uuid import uuid4
import os
import shutil
import numpy as np
from app.forms import ETC_form
from app.moons_etc_backend import do_etc_calc
import plotly.graph_objects as go
import glob
import json
import plotly


def cleanup(path, file_limit):
    '''Basic dir cleanup'''
    # list all dirs 
    dir_list = os.listdir(path)

    # make paths absolute
    full_dlist = [path + dir for dir in dir_list]
    fcount = len(full_dlist)
    # if there are too many files remove the oldest until under the limit
    while fcount > file_limit:
        oldest_dir = min(full_dlist, key=os.path.getctime)
        shutil.rmtree(oldest_dir, ignore_errors=True)
        full_dlist.remove(oldest_dir)
        fcount -= 1


# Homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    # create instance of ETC form to capture input parameters
    form = ETC_form()

    # if the form runs without error to the etc backend calcs and store result
    if form.validate_on_submit():

        # do cleanup if required
        cleanup('app/static/user_files/', 5)

        # unique id for the user
        id = str(uuid4())

        # make user directory
        fldr_list = ['SN', 'obj_spec', 'transmission']
        for folder in fldr_list:
            path = 'app/static/user_files/' + id + '/' + folder
            if os.path.exists(path) == False:
                os.makedirs(path)


        if form.template_type and form.upload_template.data != None:
            # get uploaded file
            uploaded_template = request.files[form.upload_template.name]
            uploaded_template.save('app/static/user_files/' + id + '/uploaded_template.fits')
            form.template_name.data = 'app/static/user_files/' + id + '/uploaded_template.fits'

        if form.sky_type and form.upload_sky.data != None:
            # get uploaded file
            uploaded_sky = request.files[form.upload_sky.name]
            uploaded_sky.save('app/static/user_files/' + id + '/uploaded_sky_template.fits')
            form.sky_template.data = 'app/static/user_files/' + id + '/uploaded_sky_template.fits'


        # do etc calculations with form input
        do_etc_calc(id, form.data)

        # redirect to results page
        return redirect('/results/' + id)

    return render_template('form.html', form=form)


# make data avaialble for download
@app.route('/get-txt/<path:filename>')
def send_report(filename):
    rel_path = os.path.relpath(glob.glob('app/static/user_files/' + filename)[0], start='app/static/user_files')
    return send_from_directory('static/user_files', rel_path, as_attachment=True)


# dynamically update results based on plot selection 
@app.route('/callback', methods=['POST', 'GET'])
def cb():
    print(request.args.get('data'))
    return make_plot(request.args.get('folder'), request.args.get('data'))


# results page
@app.route('/results/<user_folder>')
def results(user_folder):
    return render_template('results.html', graphJSON=make_plot(user_folder), folder=user_folder)


def make_plot(user_folder, plot_type='SN'):
    '''Make an interactive plot'''

    # gather all the bands
    file_list = glob.glob('app/static/user_files/'+user_folder+'/'+plot_type+'/*.txt')

    # create figure
    fig = go.Figure()
    # plot each band (bands are identified by starting wavelength)
    for file in file_list:
        data = np.loadtxt(file)
        wl = data[1:,0]
        snr = data[1:,1]
        if wl[0] < 0.934:
            fig.add_trace(go.Scatter(x=wl, y=snr, mode='lines', name='RI',
            line=dict(color='red', width=.5)))
        elif wl[0] > 1.45:
            fig.add_trace(go.Scatter(x=wl, y=snr, mode='lines', name='H',
            line=dict(color='blue', width=.5)))
        else:
            fig.add_trace(go.Scatter(x=wl, y=snr, mode='lines', name='YJ',
            line=dict(color='magenta', width=.5)))
    
    # add axis labels
    if plot_type == 'SN':
        fig.update_layout(xaxis_title='Wavelength / micron',
                    yaxis_title='SNR', height=800, width=1200)
    if plot_type == 'transmission':
        fig.update_layout(xaxis_title='Wavelength / micron',
                    yaxis_title='Transmission', height=800, width=1200)
    if plot_type == 'obj_spec':
        fig.update_layout(xaxis_title='Wavelength / micron',
                    yaxis_title='Object Spectrum', height=800, width=1200)

    # output as JSON  
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
from app import app
from flask import redirect, render_template, send_from_directory
from uuid import uuid4
import os
import shutil
import numpy as np
from app.make_plots import plot_folder
from app.forms import ETC_form
from app.moons_etc_backend import do_etc_calc


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

    # if the form runs without error to the etc calcs and store result
    if form.validate_on_submit():
        
        # unique id for the user
        id = str(uuid4())[:5]

        # make user directory
        fldr_list = ['SN', 'obj_spec', 'transmission']
        for folder in fldr_list:
            path = 'app/static/user_files/' + id + '/' + folder
            if os.path.exists(path) == False:
                os.makedirs(path)

        # check which data needs to be plotted and save the result
        plot_list = np.array([form.sn.data, form.trans.data, form.obj_spec.data])
        np.savetxt('app/static/user_files/'+ id +'/plot_selection.txt', plot_list)

        # do cleanup if required
        cleanup('app/static/user_files/', 5)

        # do etc calculations with form input
        do_etc_calc(id, form.data)

        # plot the necessary data
        for fldr in fldr_list:
            plot_folder('app/static/user_files/'+ id +'/' + fldr)

        # redirect to results page
        return redirect('/' + id)

    return render_template('form.html', form=form)


# results page. Unique to user
@app.route('/<user_folder>')
def results(user_folder):
    is_plot = np.loadtxt('app/static/user_files/' + user_folder + '/plot_selection.txt')
    return render_template('results.html', is_plot=is_plot, folder=user_folder)

# make data avaialble for download
@app.route('/get-txt/<path:filename>')
def send_report(filename):
    return send_from_directory('static/user_files', filename, as_attachment=True)


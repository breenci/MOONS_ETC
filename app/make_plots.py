import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import glob


def plot_folder(folder):
    file_list = glob.glob(folder + '/*.txt')
    for file in file_list:    
        data = np.loadtxt(file)
        fig, ax = plt.subplots()
        ax.plot(data[:,0], data[:,1])
        ax.set_xlabel('Wavelength ($\mu$m)')
        plt.savefig(file[:-4] + '_plot.png')




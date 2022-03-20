from scipy.integrate import cumtrapz
from constants import SAMPLE_FREQUENCY
import matplotlib.pyplot as plt

plt.style.use('ggplot')


plot_types = ['plot_3d_trajectory', 'plot_frequency_spectrum']


def generate_plots(x, y, z, plots):
    request = bin(plots)[2:].zfill(len(plot_types))[::-1]
    if request[0] == '1':
        plot_3d_trajectory(x, y, z)
    if request[1] == '1':
        plot_frequency_spectrum(x, y, z)
    plt.show()


def plot_3d_trajectory(x, y, z):
    x = cumtrapz(x)
    y = cumtrapz(y)
    z = cumtrapz(z)
    fig, ax = plt.subplots()
    fig.suptitle('3D Trajectory over ' + str(x.size * SAMPLE_FREQUENCY) + ' seconds', fontsize=20)
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, c='red', lw=1, label='trajectory')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_zlabel('Z position')


def plot_frequency_spectrum(x, y, z):
    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, sharey=True)
    fig.suptitle('Spectrum over ' + str(x.size * SAMPLE_FREQUENCY) + ' seconds', fontsize=20)
    ax1.plot(x, c='r', label='x')
    ax1.legend()
    ax2.plot(y, c='b', label='y')
    ax2.legend()
    ax3.plot(z, c='g', label='z')
    ax3.legend()
    ax3.set_xlabel('Frequency (Hz)')

# -*- coding: utf-8 -*-
# @Author: E. G. Patrick Bos
# @Date:   2017-09-20 11:28:36
# @Last Modified by:   E. G. Patrick Bos
# @Last Modified time: 2017-09-20 11:39:45

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_timeseries(timeseries, dt, ax=None, minimal_layout=True, draw_labels=True, title=None, draw_colorbar=True):
    """
    @param timeseries: numpy array of shape (T, C) (T timebins by C channels).
    @param dt: time step (in seconds) between timeseries bins.
    """
    if ax is None:
        if minimal_layout:
            fig, ax = plt.subplots(1, frameon=False)
        else:
            fig, ax = plt.subplots(1)
    else:
        fig = ax.figure

    if not minimal_layout and draw_labels:
        ax.set_xlabel('time [s]')
        ax.set_ylabel('channel')

    N_timebins, N_channels = timeseries.shape

    im = ax.imshow(timeseries.T, aspect='auto', interpolation='nearest', cmap=cm.RdYlBu,
                   extent=(0, N_timebins * dt, N_channels, 0))

    if not minimal_layout and title is not None:
        ax.set_title(title)

    if draw_colorbar:
        cb = fig.colorbar(im, ax=ax)
    else:
        cb = None

    if minimal_layout:
        ax.axis('off')

    plt.tight_layout()

    return {'ax': ax, 'fig': fig, 'im': im, 'cb': cb}


def plot_timeseries_fourier_amplitudes(timeseries, dt, ax=None, minimal_layout=True,
                                       draw_labels=True, title=None, draw_colorbar=True):
    """
    @param timeseries: numpy array of shape (T, C) (T timebins by C channels).
    @param dt: time step (in seconds) between timeseries bins.
    """
    if ax is None:
        if minimal_layout:
            fig, ax = plt.subplots(1, frameon=False)
        else:
            fig, ax = plt.subplots(1)
    else:
        fig = ax.figure

    if not minimal_layout and draw_labels:
        ax.set_xlabel('frequency [s^{-1}]')
        ax.set_ylabel('channel')

    N_timebins, N_channels = timeseries.shape

    frequencies = np.fft.fftfreq(N_timebins, d=dt)  # s^{-1}
    central_timebin = N_timebins // 2
    if N_timebins % 2 == 0:
        central_timebin -= 1
    fmin = frequencies[0]
    fmax = frequencies[central_timebin]

    fourier_transform = np.fft.fft(timeseries, axis=0)[:central_timebin + 1]

    im = ax.imshow(np.abs(fourier_transform).T, aspect='auto', interpolation='nearest',
                   extent=(fmin, fmax, N_channels, 0))

    if not minimal_layout and title is not None:
        ax.set_title(title)

    if draw_colorbar:
        cb = fig.colorbar(im, ax=ax)
    else:
        cb = None

    if minimal_layout:
        ax.axis('off')

    plt.tight_layout()

    return {'ax': ax, 'fig': fig, 'im': im, 'cb': cb}

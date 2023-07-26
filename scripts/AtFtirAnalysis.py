#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:30:03 2017

@author: moritz
"""


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from scipy import signal

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from scipy.optimize import nnls



class AtFtirAnalysis():
    """
    A class to process and analyze Attenuated Total Reflectance Fourier Transform Infrared (ATR-FTIR) spectra data.
    
    Parameters
    -----
    dataframe: pandas DataFrame
        The input DataFrame containing the ATR-FTIR spectra data. Wavenumbers are stored in the column headers of the dataframe
    """

    def __init__(self, dataframe):
        assert isinstance(dataframe, pd.core.frame.DataFrame), 'data object you provided is not a pandas data frame'

        self.dataframe = dataframe

        # holds the original data
        self.data = self.dataframe.iloc[:, 3:]
        # convert wavenumbers from string to integers
        self.data.columns = [int(item) for item in self.data.columns]

        self.wavenumbers = np.asarray(self.data.columns, dtype=float)
        # holds the pre processed data
        self._preprocessed = self.data.values
        
        # info should contain "time", "name" and "date" for all rows
        self.info = self.dataframe.iloc[:,:3]
        # convert everything to lower case
        self.info.name = self.info.name.str.lower()   

    '''
    Preprocessing methods
    '''

    def take_slice(self, low=1300, high=1800):
        """
        Take a slice of the spectrum within the range specified.

        Parameters
        -----
        low: float, default=1300
            The lower limit of the slice.
        
        high: float, default=1800
            The upper limit of the slice.

        Returns
        -----
        Writes processed dataframe to self._preprocessed
        """

        self._low = low
        self._high = high
        w = self.wavenumbers
        ind = np.where(((w > low) & (w < high)))

        self._preprocessed = self._preprocessed.T[ind].T
        self._multiple_slice = 0

    def take_multiple_slice(self, low=[1100, 1250], high=[1600, 1700]):
        """
        Take multiple slices of the spectrum within the ranges specified.

        Parameters
        -----
        low: list of float, default=[1100, 1250]
            The lower limits of the slices.
        
        high: list of float, default=[1600, 1700]
            The upper limits of the slices.

        Sets
        -----
        self._preprocessed (numpy.ndarray): IThe preprocessed data
        """
        self._low = low
        self._high = high
        w = self.wavenumbers
        ind = np.where(((w > self._low[0]) & (w < self._high[0])) | ((w > self._low[1]) & (w < self._high[1])) )

        self._preprocessed = self._preprocessed.T[ind].T
        self._multiple_slice = 1

    def savitzky_golay(self):
        """

        :param self:
        :return:
        """
        raise NotImplementedError

    def baseline_correction(self, reference):
        """
        Corrects the baseline by subtracting the reference spectrum.

        Parameters:
        -----
        reference (numpy.ndarray): The reference spectrum.

        Sets
        -----
        self._preprocessed (numpy.ndarray): IThe preprocessed data
        """
        # Make sure the reference is a numpy array
        assert isinstance(reference, np.ndarray), "The reference spectrum must be a numpy array."

        # Normalize the reference and the spectra to be corrected
        reference_norm = reference / np.linalg.norm(reference)
        self._preprocessed = self._preprocessed / np.linalg.norm(self._preprocessed, axis=1, keepdims=True)

        # Subtract the reference from the spectra
        self._preprocessed -= reference_norm

    def low_pass_filter(self, cutoff=3, f_sampling=0.1, order=2):
        """
        Apply a low pass Butterworth filter to the spectrum.

        Parameters
        -----
        cutoff: float, default=3
            The cutoff frequency of the filter.
        
        f_sampling: float, default=0.1
            The sampling frequency of the filter.
        
        order: int, default=2
            The order of the filter.

        Sets
        -----
        self._preprocessed (numpy.ndarray): IThe preprocessed data
        """

        b, a = signal.butter(cutoff, f_sampling, btype='low', analog=False)
        y = signal.lfilter(b, a, self._preprocessed)
        self._preprocessed = y


    def normalize_std(self):
        norm = StandardScaler().fit_transform(self._preprocessed)
        self._preprocessed = norm


    def gradient(self, order=2, axis=1):
        """
        Takes the gradient along axis (dfault: rows) and stores it in _preprocessed
        :param order:
        :return: gradient along rows
        """
        self._preprocessed = np.gradient(self._preprocessed, order)[axis]


    '''
    Collection of cluster analysis alogorithms which use self._preprocessed (2D Array)
    '''

    def pca(self, n_components=3):
        """
        Principal component analysis of all rows in self._preprocessed
        :param self:
        :return: dataframe of principal components
        """

        self._pca = PCA(n_components=n_components)
        self._pca.fit(self._preprocessed)
        components = self._pca.transform(self._preprocessed)

        # make the column names
        columns = ['PCA{}'.format(i) for i in range(n_components)]
        df_pca = pd.DataFrame(components, columns=columns)
        df_pca = pd.concat((self.info, df_pca), axis=1)
        df_pca.time = [float(item) for item in df_pca.time]
        self._pca.df = df_pca

    def mcr(self, n_components=3, max_iter=500, tol=1e-6):
        """
        Performs Multivariate Curve Resolution using Alternating Least Squares (MCR-ALS).

        MCR-ALS is a technique used to decompose complex mixtures into their pure components
        when the number of components is known a priori. It alternates between fitting the original
        data and minimizing the residuals using non-negative least squares until convergence.

        Parameters:
        -----
        n_components (int): The number of components to extract. Default is 3.
        max_iter (int): Maximum number of iterations for the optimization algorithm. Default is 500.
        tol (float): Tolerance for the stopping criteria. Default is 1e-6.

        Sets:
        -----
        self._mcr (numpy.ndarray): The extracted components after performing MCR-ALS.
        """
        pls = PLSRegression(n_components=n_components)
        C_old = np.zeros(self._preprocessed.shape)
        
        for i in range(max_iter):
            pls.fit(self._preprocessed, C_old)
            C = pls.predict(self._preprocessed)
            C, _ = nnls(C, self._preprocessed)
            if np.sqrt(np.mean((C - C_old)**2)) < tol:
                break
            C_old = C
        
        self._mcr = C


    ''' Plot methods'''
    def plot_pca(self):
        """
        Plot a 2D scatter plot of the first two principal components for each experiment over time.

        Parameters
        ----------
        None

        Returns
        -------
        matplotlib.axes.Axes: The matplotlib Axes object with the plot.
        """
        gr = self._pca.df.groupby('date')

        color = plt.cm.tab10

        fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
        for index, item in enumerate(gr):
            name, val = item

            x = val.PCA0
            y = val.PCA1
            c = val.time

            if any(val.name.str.contains('blank')):
                marker = 'x'
                color = plt.cm.tab20c_r
            else:
                marker = '.'

            ax.scatter(x, y, c=color(index), label=name, marker=marker)
            ax.plot(x, y, 'k-', alpha=0.5, linewidth=0.5, label='')
            ax.scatter(x.values[-1], y.values[-1], marker='o', color=color(index), s=300, edgecolor='k')

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        v = self._pca.explained_variance_ratio_ * 100
        ax.set_xlabel('PC 0 : explains {:.1f} % variance'.format(v[0]))
        ax.set_ylabel('PC 1 : explains {:.1f} % variance'.format(v[1]))

        vs = v[0:2].sum()
        ax.set_title('Score plot: explained variance {:.1f} %'.format(vs))
        return ax

    def plot_pca_3d(self):
        """
        Plot a 3D scatter plot of the first two principal components for each experiment over time.

        Parameters
        ----------
        None

        Returns
        -------
        matplotlib.axes.Axes: The matplotlib Axes object with the plot.
        """
        gr = self._pca.df.groupby('date')

        color = plt.cm.tab10

        fig = plt.figure(1, figsize=(8, 8))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=45)

        for index, item in enumerate(gr):
            name, val = item

            x = val.PCA0
            y = val.PCA1
            z = val.PCA2

            if any(val.name.str.contains('blank')):
                marker = 'x'
                color = plt.cm.tab20c_r
            else:
                marker = '.'

            ax.scatter(x, y, z, c=color(index), label=name, marker=marker)

            ax.scatter(x.values[-1], y.values[-1], z.values[-1], marker='o',
                       color=color(index), s=300, edgecolor='k')
            ax.scatter(x.values[0], y.values[0], z.values[0], marker='o',
                       color=color(index), s=25, edgecolor='k')

        ax.legend()
        v = self._pca.explained_variance_ratio_ * 100
        ax.set_xlabel('PC 0 : explains {:.1f} % variance'.format(v[0]))
        ax.set_ylabel('PC 1 : explains {:.1f} % variance'.format(v[1]))
        ax.set_zlabel('PC 2 : explains {:.1f} % variance'.format(v[2]))
        vs = v.sum()
        ax.set_title('Score plot: explained variance {:.1f} %'.format(vs))

        return ax


    def plot_pca_loadings(self):
        """
        Plot the loadings of the PCA components.

        Parameters
        ----------
        None

        Returns
        -------
        matplotlib.axes.Axes: The matplotlib Axes object with the plot.
        """
        mask = (self.data.columns > self._low) & (self.data.columns < self._high)
        b = self.data.loc[:, mask]

        n = len(self._pca.components_)
        fig, ax = plt.subplots(nrows=n, figsize=(8,3*n))
        fig.subplots_adjust(hspace=0.5)
        # the rows of compontenst_ are the loadings
        for i, loading in enumerate(self._pca.components_):

            x = b.columns.values
            y = loading
            ax[i].hlines(xmin=self._low, xmax=self._high, y=0, linestyle=':')
            ax[i].plot(x, y, label='loadings on PCA{}'.format(i))
            ax[i].set_title('Loading plot')
            ax[i].set_xlabel('Wavenumber')
            ax[i].set_ylabel('Magnitude on unit vector')
            ax[i].legend(loc='best')

        return ax

    def plot_raw_spectra(self, low=1300, high=1800, step=5, as_analysis=0):
        """
        Plot raw spectra of the data.

        Parameters
        ----------
        low : int, optional
            Lower limit of the plot, by default 1300
        high : int, optional
            Upper limit of the plot, by default 1800
        step : int, optional
            Step size for plotting, by default 5
        as_analysis : int, optional
            Whether to plot as analysis, by default 0

        Returns
        -------
        matplotlib.axes.Axes: The matplotlib Axes object with the plot.
        """
        if as_analysis == 1:
            low = self._low
            high = self._high

        mask = (self.data.columns > low) & (self.data.columns < high)
        b = self.data.loc[:, mask]

        df = pd.concat((self.info, b), axis=1)
        self._df = df
        gr = df.groupby('date')

        fig, ax = plt.subplots(nrows=len(gr), figsize=(8, 4 * len(gr)))
        fig.subplots_adjust(hspace=0.5)

        for index, item in enumerate(gr):
            name, val = item

            t = val.time[::step].values
            x = b.columns.values
            y = val.iloc[::step, 3:].values

            color = plt.cm.viridis(np.linspace(0, 1, len(y)))

            for i, yl in enumerate(y):
                ax[index].plot(x, yl, label=t[i], color=color[i])

            #ax[index].legend()
            ax[index].set_title(name)

        return ax

    def plot_processed_spectra(self, low=1300, high=1800, step=5):
        """
        Plot processed spectra of the data.

        Parameters
        ----------
        low : int, optional
            Lower limit of the plot, by default 1300
        high : int, optional
            Upper limit of the plot, by default 1800
        step : int, optional
            Step size for plotting, by default 5

        Returns
        -------
        matplotlib.axes.Axes: The matplotlib Axes object with the plot.
        """
        mask = (self.data.columns > low) & (self.data.columns < high)
        b = self.data.loc[:, mask]

        df = pd.concat((self.info, b), axis=1)
        self._df = df
        gr = df.groupby('date')

        fig, ax = plt.subplots(nrows=len(gr), figsize=(8, 4 * len(gr)))
        fig.subplots_adjust(hspace=0.5)

        for index, item in enumerate(gr):
            name, val = item

            t = val.time[::step].values
            x = b.columns.values
            y = val.iloc[::step, 3:].values

            color = plt.cm.viridis(np.linspace(0, 1, len(y)))

            for i, yl in enumerate(y):
                ax[index].plot(x, yl, label=t[i], color=color[i])

            #ax[index].legend()
            ax[index].set_title(name)

        return ax

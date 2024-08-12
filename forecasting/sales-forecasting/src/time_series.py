####################################
# Time series analysis
####################################

# Filter warnings
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


class TimeSeriesForecasting:
    """A class for analyzing time series data."""

    def __init__(self) -> None:
        pass

    def error_metrics(self, y_true, y_pred):
        """Print out error metrics."""
        mape = self.mape(y_true, y_pred)
        wmape = self.wmape(y_true, y_pred)
        mase = self.mase(y_true, y_pred)
        r2 = self.r_squared(y_true, y_pred)
        mae = self.mae(y_true, y_pred)
        rmse = self.rmse(y_true, y_pred)

        errors = {
            'MAPE' : np.round(mape, 3),
            'WMAPE' : np.round(wmape, 3),
            'MASE' : np.round(mase,3),
            'MAE' : np.round(mae, 3),
            'RMSE': np.round(rmse, 3),
            'R^2': np.round(r2, 3),
        }
        return errors
        
    def mape(self, y_true, y_pred):
        """Mean absolute percentage error."""
        _mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return _mape

    def wmape(self, y_true, y_pred):
        """Weighted mean absolute percentage error."""
        et = y_true - y_pred
        _wmape = np.sum(np.abs(et)) * 100 / np.sum(np.abs(y_true))
        return _wmape

    def mase(self, y_true, y_pred):
        """Mean absolute scaled error."""
        # mean absolute error of one-step ahead
        # naive forecast for non-seasonal forecast
        mae_naive = pd.Series(y_true).diff().abs().mean()
        
        # mean absolute error of forecast
        _mae = self.mae(y_true,  y_pred)
        _mase = _mae / mae_naive
        return _mase

    def mae(self, y_true, y_pred):
        """Mean absolute error."""
        _mae = np.mean(np.abs(y_true - y_pred))
        return _mae

    def rmse(self, y_true, y_pred):
        """Root mean squared error."""
        _rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return _rmse

    def r_squared(self, y_true, y_pred):
        """r-squared (coefficient of determination)."""
        mse = np.mean((y_true - y_pred) ** 2)  # mean squared error
        var = np.mean((y_true - np.mean(y_true)) ** 2)  # sample variance
        _r_squared = 1 - mse / var
        return _r_squared
        
    def plot_timeseries(self, ts, color=None, marker=None, title=''):
        """Plot univariant time series data."""
        ts.plot(
            marker=marker,
            markersize=10,
            markerfacecolor='none',
            color=color,
            figsize=(18, 8),
        )
        plt.xlabel('Year', fontsize=20)
        plt.ylabel('Monthly sales', fontsize=20)
        plt.xlim('2014-12-01', '2021-10-01')
        plt.title(title, fontsize=20)
        plt.savefig('../img/ts.png')

    def plot_ts(self, ts, title='', nlags=None):
        """This function plots the original time series together rolling mean
        and standard deviations and its ACF and partial ACF.
        It also performs the Dickey-Fuller test
        """
        gridsize = (2, 2)
        ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2)
        ax2 = plt.subplot2grid(gridsize, (1, 0))
        ax3 = plt.subplot2grid(gridsize, (1, 1))

        # Rolling statistic
        rolling_mean = ts.rolling(window=24).mean()
        rolling_std = ts.rolling(window=24).std()

        # Plot original time series and rolling mean & std
        ts.plot(
            color='r',
            marker='o',
            markersize=10,
            markerfacecolor='none',
            ax=ax1,
            label='Original',
        )
        rolling_mean.plot(
            color='b',
            markersize=10,
            markerfacecolor='none',
            ax=ax1,
            label='Rolling mean',
        )
        rolling_std.plot(
            color='g',
            markersize=10,
            markerfacecolor='none',
            ax=ax1,
            label='Rolling Std',
        )
        ax1.set_xlabel('Time', fontsize=20)
        ax1.set_ylabel('Monthly sales', fontsize=20)
        ax1.set_title(title, fontsize=20)
        ax1.set_xlim('2014-12-01', '2021-10-01')
        ax1.legend(loc='best')

        # Plot ACF
        plot_acf(ts, lags=nlags, ax=ax2)
        ax2.set_xlabel('Lag', fontsize=20)
        ax2.set_ylabel('ACF', fontsize=20)
        ax2.set_xticks(np.arange(0, nlags, 12))
        ax2.set_title('Autocorrelation', fontsize=20)

        # Plot PACF
        plot_pacf(ts, lags=nlags, ax=ax3)
        ax3.set_xlabel('Lag', fontsize=20)
        ax3.set_ylabel('PACF', fontsize=20)
        ax3.set_xticks(np.arange(0, nlags, 12))
        ax3.set_title('Partial Autocorrelation', fontsize=20)

        # Perform Dickey-Fuller test
        adf_results = adfuller(ts.values)
        print('Test statistic:', adf_results[0])
        print('p-value:', adf_results[1])
        for key, value in adf_results[4].items():
            print('Critial Values (%s): %0.6f' % (key, value))

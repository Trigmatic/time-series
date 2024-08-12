################################################
# Supervised Regression Machine Learning
################################################

import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.model_selection import cross_validate


class RegressionModels:
    """A class for training supervised regression models."""

    def __init__(self):
        """Parameter initialization."""

    def custom_scorer(self):
        """Define custom scorer currently not 
        available in sklearn scoring list.

        Returns
        -------
        custom scoring metric
        """
        wmape_scoring = make_scorer(self.wmape, greater_is_better=False)
        return wmape_scoring

    def evaluate_metric_test(self, y_pred, y_true, model_nm=None):
        """Predictions on the test set.

        Parameters
        ----------
        y_pred: training set class labels
        y_true: test set class labels

        Returns
        -------
        Performance metrics on the test set
        """
        # Print results
        print(f'Test prediction results for {model_nm}')
        print('-' * 60)
        print(self.error_metrics(y_true, y_pred))
        print('-' * 60)

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
        _mae =  self.mae(y_true,  y_pred)
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

    def evaluate_metric_cv(
        self, model, X_train, y_train, cv_fold, model_nm=None
    ):
        """Cross-validation on the training set.

        Parameters
        ----------
        model: supervised classification model
        X_train: training feature matrix
        y_train: target variable
        cv_fold: number of cross-validation fold

        Returns
        -------
        Performance metrics on the cross-validation training set
        """
        scoring = {
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error',
            'wmape': self.custom_scorer(),
            'r2': 'r2',
        }
        cv_results = cross_validate(
            model, X_train, y_train, cv=cv_fold, scoring=scoring
        )
        mae = -cv_results['test_mae']
        rmse = -cv_results['test_rmse']
        wmape = -cv_results['test_wmape']
        r2 = cv_results['test_r2']

        errors = {
            'wMAPE' : np.round(wmape.mean(), 3),
            'MAE' : np.round(mae.mean(), 3),
            'RMSE' : np.round(rmse.mean(), 3),
            'R^2' : np.round(r2.mean(), 3),
        }
        print(f'5-fold cross-validation results for {str(model_nm)}')
        print('-' * 60)
        print(errors)
        print('-' * 60)

    def plot_wmape_svr(self, X_train, y_train, cv_fold):
        """Plot of cross-validation wMAPE and MAE for SVR.

        Parameters
        ----------
        X_train: training feature matrix
        y_train: target variable
        cv_fold: number of cross-validation fold

        Returns
        -------
        matplolib figure of wMAPE & MAE
        """
        C_list = [2**x for x in range(-2, 11, 2)]
        gamma_list = [2**x for x in range(-7, -1, 2)]
        mae_list = [
            pd.Series(0.0, index=range(len(C_list)))
            for _ in range(len(gamma_list))
        ]
        wmape_list = [
            pd.Series(0.0, index=range(len(C_list)))
            for _ in range(len(gamma_list))
        ]

        axes_labels = ['2^-2', '2^0', '2^2', '2^4', '2^6', '2^8', '2^10']
        gamma_labels = ['2^-7', '2^-5', '2^-3']
        plt.rcParams.update({'font.size': 15})
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        for i, val1 in enumerate(gamma_list):
            for j, val2 in enumerate(C_list):
                model = SVR(C=val2, gamma=val1, kernel='rbf')
                scoring = {
                    'mae': 'neg_mean_absolute_error',
                    'wmape': self.custom_scorer(),
                }
                cv_results = cross_validate(
                    model, X_train, y_train, cv=cv_fold, scoring=scoring
                )
                mae = -cv_results['test_mae']
                wmape = -cv_results['test_wmape']
                mae_list[i][j] = mae.mean()
                wmape_list[i][j] = wmape.mean()
            wmape_list[i].plot(
                label='gamma=' + str(gamma_labels[i]),
                marker='o',
                linestyle='-',
                ax=ax1,
            )
            mae_list[i].plot(
                label='gamma=' + str(gamma_labels[i]),
                marker='o',
                linestyle='-',
                ax=ax2,
            )

        ax1.set_xlabel('C', fontsize=15)
        ax1.set_ylabel('wMAPE', fontsize=15)
        ax1.set_title(
            '5-Fold Cross-Validation with RBF kernel SVR', fontsize=15
        )
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc='best')

        ax2.set_xlabel('C', fontsize=15)
        ax2.set_ylabel('MAE', fontsize=15)
        ax2.set_title(
            '5-Fold Cross-Validation with RBF kernel SVR', fontsize=15
        )
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc='best')
        plt.show()

    def diagnostic_plot(
        self, y_true, y_pred, marker=None, color=None, label=None
    ):
        """Diagnostics plot.

        Parameters
        ----------
        y_pred: predicted
        y_true: actual

        Returns
        -------
        Matplolib figure
        """
        # residuals & standardize residuals
        residual = y_true - y_pred
        rstandard = (residual - np.mean(residual)) / np.std(residual)

        # mean absolute percentage error
        mape = {}
        ape = np.abs(residual) * 100 / np.abs(y_true)
        for i in range(0, len(ape), 24):
            mape[i + 24] = np.mean(ape[: i + 24])

        # plot figures
        plt.subplot(221)
        plt.plot(
            range(1, len(rstandard) + 1),
            rstandard,
            color=color,
            marker=marker,
            markerfacecolor='none',
            label=label,
        )
        plt.grid(True)
        plt.xlabel('Forecast horizon (hours)')
        plt.ylabel('Standardized residuals')
        plt.title('Standardized residuals over time')
        plt.legend(loc='best')

        plt.subplot(222)
        plt.plot(
            mape.keys(), mape.values(), color=color, marker=marker, label=label
        )
        plt.grid(True)
        plt.xlabel('Forecast horizon (hours)')
        plt.ylabel('Mean absolute percentage error')
        plt.title('Mean absolute percentage error over time')
        plt.xticks(np.arange(24, 121, 24))
        labels = ['0%', '40%', '80%', '120%', '160%']
        plt.yticks(np.arange(0, 170, 40), labels)
        plt.legend(loc='best')

        plt.subplot(223)
        lb = acorr_ljungbox(residual, lags=49)[1]
        plt.plot(
            np.arange(1, len(lb) + 1),
            lb,
            color=color,
            marker=marker,
            markerfacecolor='none',
            linestyle='',
            label=label,
        )
        plt.axhline(y=0.05, linestyle='--', color='k')
        plt.ylabel('p-value')
        plt.xlabel('Lag')
        plt.xticks(np.arange(0, 50, 12))
        plt.ylim(-0.1, 1)
        plt.legend(loc='best')
        plt.title('p-values for Ljung-Box Test')

        plt.subplot(224)
        plt.acorr(
            residual, usevlines=True, maxlags=49, normed=True, color=color
        )
        plt.acorr(
            residual,
            usevlines=False,
            maxlags=49,
            normed=True,
            color=color,
            label=label,
        )
        plt.axhline(
            y=1.96 / np.sqrt(len(residual)),
            linestyle='--',
            linewidth=1.5,
            color='k',
        )
        plt.axhline(
            y=-1.96 / np.sqrt(len(residual)),
            linestyle='--',
            linewidth=1.5,
            color='k',
        )
        plt.xlim(0, 50)
        plt.ylim(-1, 1)
        plt.xticks(np.arange(0, 50, 12))
        plt.grid(True)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation of residuals')
        plt.legend(loc='best')
        plt.savefig('../images/residual.png')

    def diagnostic_plot_(self, y_pred, y_true, ylim=None):
        """
        Diagnostic plot.

        Parameters
        ----------
        y_pred: predicted labels
        y_true: true labels

        Returns
        -------
        Matplolib figure
        """
        # compute residual and metrics
        residual = y_true - y_pred
        r2 = np.round(self.r_squared(y_true, y_pred), 3)
        wmape = np.round(self.wmape(y_true, y_pred), 3)

        # plot figures
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        ax1.scatter(y_pred, residual, color='b')
        ax1.set_xlim([-0.1, 1.1])
        ax1.set_ylim(ylim)
        ax1.hlines(y=0, xmin=-0.1, xmax=1.1, lw=2, color='k')
        ax1.set_xlabel('Predicted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs. Predicted values')

        ax2.scatter(y_pred, y_true, color='b')
        ax2.plot([-0.08, 1.1], [-0.08, 1.1], color='k')
        ax2.set_xlim([-0.08, 1.1])
        ax2.set_ylim([-0.08, 1.1])
        ax2.text(
            0.3,
            0.1,
            r'$R^2 = {},~ wMAPE = {}$'.format(str(r2), str(wmape)),
            fontsize=20,
        )
        ax2.set_xlabel('Predicted values')
        ax2.set_ylabel('True values')
        ax2.set_title('True values vs. Predicted values')

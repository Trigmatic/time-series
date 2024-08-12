################################
# Transformation Pipeline
#################################

import warnings

warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler


class TransformationPipeline:
    """This class is used for transformation pipeline."""

    def __init__(self):
        """Define parameters."""

    def num_pipeline(self, X_train, X_test):
        """Transformation pipeline of data with only numerical variables.

        Parameters
        ----------
        X_train: training feature matrix
        X_test: test feature matrix

        Returns
        -------
        Transformation pipeline and transformed data in array
        """
        # create pipeline
        num_pipeline = Pipeline(
            [
                ('p_transform', PowerTransformer(standardize=False)),
                ('s_scaler', StandardScaler()),
            ]
        )

        # original numerical feature names
        feat_nm = list(X_train.select_dtypes('number'))

        # fit transform the training set and transform the test set
        X_train_scaled = num_pipeline.fit_transform(X_train)
        X_test_scaled = num_pipeline.transform(X_test)
        return X_train_scaled, X_test_scaled, feat_nm

    def cat_encoder(self, X_train, X_test):
        """Transformation pipeline of categorical variables.

        Parameters
        ----------
        X_train: training feature matrix
        X_test: test feature matrix

        Returns
        -------
        Transformation pipeline and transformed data in array
        """
        # instatiate class
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

        # fit transform the training set and only transform the test set
        X_train_scaled = one_hot_encoder.fit_transform(X_train)
        X_test_scaled = one_hot_encoder.transform(X_test)

        # feature names for output features
        feat_list = list(X_train.select_dtypes('O'))
        feat_nm = list(one_hot_encoder.get_feature_names_out(feat_list))
        return X_train_scaled.toarray(), X_test_scaled.toarray(), feat_nm

    def preprocessing(self, X_train, X_test):
        """Transformation pipeline of data with
        both numerical and categorical variables.

        Parameters
        ----------
        X_train: training feature matrix
        X_test: test feature matrix

        Returns
        -------
        Transformed data in array
        """

        # numerical transformation pipepline
        x_train = X_train.select_dtypes('number')
        x_test = X_test.select_dtypes('number')
        num_train, num_test, num_col = self.num_pipeline(x_train, x_test)

        # categorical transformation pipepline
        cat_train, cat_test, cat_col = self.cat_encoder(
            X_train.select_dtypes('O'), X_test.select_dtypes('O')
        )

        # transformed training and test sets
        X_train_scaled = np.concatenate((num_train, cat_train), axis=1)
        X_test_scaled = np.concatenate((num_test, cat_test), axis=1)

        # feature names
        feat_nm = num_col + cat_col
        return X_train_scaled, X_test_scaled, feat_nm

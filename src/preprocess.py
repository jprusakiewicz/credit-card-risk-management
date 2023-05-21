import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2


def get_feature_selector(config):
    # todo use python switch case mechanism https://docs.python.org/3/whatsnew/3.10.html#pep-634-structural-pattern-matching
    # return VarianceThreshold # todo add as option
    if config.type == "LinearSVC":
        return SelectFromModel(LinearSVC(**config.params))


def get_balancer(data: np.array, config) -> np.array:
    # use one of belows options to balance data (create switch case based on config)
    # python switch case mechanism https://docs.python.org/3/whatsnew/3.10.html#pep-634-structural-pattern-matching
    # 1. from sklearn.decomposition import PCA
    # 2. from imblearn.over_sampling import SMOTE
    # 3. from imblearn.under_sampling import TomekLinks
    # 4. imblearn.under_sampling import AllKNN
    # 5. SMOTE + Tomek
    # 6. unbalanced
    # raise error if string in config not matching any case
    pass


def get_scaler(config):
    return MinMaxScaler()
    # return StandardScaler() # todo add as option


def get_preprocessor(config):
    # numeric_features = config.numeric_features
    # categorical_features = config.categorical_features
    numeric_features = [1, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23]  # todo move to config
    categorical_features = [2, 3, 4, 6, 7, 8, 9, 10, 11]  # todo move to config
    # todo 0 is index column, if data will be read without index, all these must be -1

    numeric_transformer = Pipeline(
        # todo if list will be empty, (nothing in config) it will fail
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # todo choose in config
            ("scaler", get_scaler(config.scaler))  # todo choose in config
        ]
    )

    categorical_transformer = Pipeline(
        # todo if list will be empty, (nothing in config) it will fail
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),  # todo choose in config ()
            ("selector", SelectPercentile(chi2, percentile=50)),  # todo choose in config
        ]
    )
    preprocessor = ColumnTransformer(
        # todo if list will be empty, (nothing in config) it will fail
        transformers=[
            ("num", numeric_transformer, numeric_features),  # todo choose in config
            ("cat", categorical_transformer, categorical_features),  # todo choose in config
        ]
    )
    return preprocessor

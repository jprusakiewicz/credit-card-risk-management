import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, chi2


def preprocess(data: np.array, config) -> np.array:
    # todo
    return data


def rescale_data(data: np.array, config) -> np.array:
    # from sklearn.preprocessing import MinMaxScaler
    # todo will we need to scale x_test as well?
    pass


def get_feature_selector(config):
    # from sklearn.feature_selection import VarianceThreshold
    return SelectFromModel(LinearSVC(**config.LinearSVC))


def balance_data(data: np.array, config) -> np.array:
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


def get_preprocessor(config):
    # numeric_features = config.numeric_features
    # categorical_features = config.categorical_features
    numeric_features = [1, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23] # todo to config
    categorical_features = [2, 3, 4, 6, 7, 8, 9, 10, 11] # todo to config
    # todo 0 is index column, if data will be read without index, all these must be -1

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

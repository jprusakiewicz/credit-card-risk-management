import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


def preprocess(data: np.array, config) -> np.array:
    # todo
    return data


def rescale_data(data: np.array, config) -> np.array:
    # from sklearn.preprocessing import MinMaxScaler
    # todo will we need to scale x_test as well?
    pass


def feature_selection(x, y, config):
    # from sklearn.feature_selection import VarianceThreshold
    selector = SelectFromModel(LinearSVC(**config.LinearSVC)).fit(x, y)  # todo params from config
    return selector

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

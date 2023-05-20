import pandas as pd
import numpy as np


def read_data(path: str = "./data/credit_cards.xls") -> pd.DataFrame:
    return pd.read_excel(path, header=0, skiprows=[0])


def get_split_values(data: pd.DataFrame, target_column="default payment next month") -> (np.array, np.array):
    x = data.drop(target_column, axis=1).values
    y = data[target_column].values
    return x, y

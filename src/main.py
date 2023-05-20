import sys

sys.path.append('src')

from read_data import read_data, get_split_values
from train import run_training
from evaluate import evaluate_model

data = read_data()
x, y = get_split_values(data)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y,
                                                    random_state=420)  # todo test_size from config

model = run_training(x_train, y_train, config=...)
metrics = evaluate_model(model=model, x=x_test, y=y_test)
print(metrics)
# todo save artifacts (config, model, metrics, etc.) in one place

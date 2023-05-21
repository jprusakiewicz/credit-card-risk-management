import sys
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split


sys.path.append('src')

from read_data import read_data, get_split_values
from train import run_training
from evaluate import evaluate_model
from preprocess import feature_selection


config = OmegaConf.load('config/test_config.yaml')
data = read_data()
x, y = get_split_values(data)
x = feature_selection(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, **config.train_test_split)
model = run_training(x_train, y_train, config=config)
metrics = evaluate_model(model=model, x=x_test, y=y_test)

print(metrics)
# todo save artifacts (config, model, metrics, etc.) in one place

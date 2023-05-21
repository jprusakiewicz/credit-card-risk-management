import sys
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

sys.path.append('src')

from read_data import read_data, get_split_values
from train import build_model
from evaluate import evaluate_model
from preprocess import get_feature_selector, get_preprocessor, get_balancer


# todo kuba save artifacts (config, model, metrics, etc.) in one place
def run() -> dict:
    config = OmegaConf.load('config/test_config.yaml')
    data = read_data()
    x, y = get_split_values(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, **config.train_test_split)

    pipeline = make_pipeline(get_preprocessor(config.preprocessing),
                             get_feature_selector(config.feature_selection),
                             # todo balance data using get_balancer() (if wont work, use https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
                             build_model(config.model), verbose=2)
    pipeline.fit(x_train, y_train)

    metrics = evaluate_model(model=pipeline, x=x_test, y=y_test)
    return metrics


if __name__ == "__main__":
    metrics = run()
    print(metrics)

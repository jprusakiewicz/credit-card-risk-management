from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier


def build_model(config):
    match config.type:
        case "LogisticRegression":
            model = LogisticRegression
        case "DecisionTreeClassifier":
            model = DecisionTreeClassifier
        case "GaussianNB":
            model = GaussianNB
        case "MLPClassifier":
            model = MLPClassifier
        case "Perceptron":
            model = Perceptron
        case _:
            raise ValueError(f"model type {config.model.type} not supported")
    return model(**config.params)


def run_training(x, y, config):
    model = build_model(config.model)
    model.fit(x, y)
    # todo save model
    return model

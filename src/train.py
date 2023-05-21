from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


def build_model(config):
    match config.type:
        case "LogisticRegression":
            model = LogisticRegression
        case "RidgeClassifier":
            model = RidgeClassifier
        case "DecisionTree":
            model = DecisionTreeClassifier
        case "RandomForest":
            model = RandomForestClassifier
        case "GaussianNB":
            model = GaussianNB
        case "MLPClassifier":
            model = MLPClassifier
        case "KNN":
            model = KNeighborsClassifier
        case "GBoost":
            model = GradientBoostingClassifier
        case "AdaBoost":
            model = AdaBoostClassifier
        case "ExtraTrees":
            model = ExtraTreesClassifier
        case _:
            raise ValueError(f"model type {config.model.type} not supported")
    return model(**config.params)


def run_training(x, y, config):
    model = build_model(config.model)
    model.fit(x, y)
    return model

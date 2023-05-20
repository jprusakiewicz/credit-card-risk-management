from sklearn.linear_model import LogisticRegression


def build_model(config):
    return LogisticRegression()


def run_training(x, y, config):
    model = build_model(config)
    model.fit(x, y)
    # todo save model
    return model

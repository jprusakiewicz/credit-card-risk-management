from sklearn.metrics import f1_score, accuracy_score


def evaluate_model(model, x, y) -> dict:
    y_pred = model.predict(x)

    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)

    return {"f1": f1, "accuracy": acc}

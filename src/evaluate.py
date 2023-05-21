from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def evaluate_model(model, x, y) -> dict:
    y_pred = model.predict(x)

    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": acc}

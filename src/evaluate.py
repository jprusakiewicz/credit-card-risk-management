from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, x, y) -> dict:
    y_pred = model.predict(x)
    y_pred_proba = model.predict_proba(x)

    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)

    positive_probs = [prob[1] for prob in y_pred_proba]
    auc = roc_auc_score(y, positive_probs)
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": acc, "auc": auc}

# creating an evaluation function for our model experiments

# * Accuracy = Default metrics for classification problems
# * Prcision = Higher precision leads to less false positives
# * Recall = Higer recall leads to less false negatives
# * F1-score = Combination of precision and recall usually a good overall metric fro a classification model

# * confusion metrics = custom for images


from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_results(y_true, y_pred):
    """
    Calcualtes model accuracy , precision , recall and F1 score of binary classification model
    """

    model_accuracy = accuracy_score(y_true, y_pred) * 100

    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred)

    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision, "recall": model_recall, "f1": model_f1}

    return model_results


def process_the_model_output(tf, model, val_sentences, val_labels, calculate_results):
    evalres = model.evaluate(val_sentences, val_labels)
    print(f"Model Evaluation result {evalres}")

    model_pred_probs = model.predict(val_sentences)
    model_pred = tf.squeeze(tf.round(model_pred_probs))

    return calculate_results(y_true=val_labels, y_pred=model_pred)

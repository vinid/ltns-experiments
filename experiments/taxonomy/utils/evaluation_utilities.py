from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd


def compute_all_values(gold_standard, prediction_file, kb, use_kb=True, use_only_kb=False):
    """
    Compute performance measures
    :param gold_standard: csv to be loaded into a pd.DataFrame
    :param prediction_file: csv to be loaded into a pd.DataFrame
    :param kb:
    :param use_kb:
    :param use_only_kb:
    :return:
    """
    closed = pd.read_csv(gold_standard, names=["first", "second", "type"])
    testing = pd.read_csv(prediction_file, skiprows=[0], names=["first", "second", "type"])

    data = pd.merge(closed, testing, on=["first", "second"])

    zipped_data = []
    if not use_kb:
        for index, row in data.iterrows():
            if (row["first"], row["second"]) in kb:
                continue
            else:
                zipped_data.append((row["type_x"], row["type_y"]))
    else:
        zipped_data = zip(data["type_x"].values.tolist(), data["type_y"].values.tolist())

    if use_only_kb:
        zipped_data = []
        for index, row in data.iterrows():
            if (row["first"], row["second"]) not in kb:
                continue
            else:
                zipped_data.append((row["type_x"], row["type_y"]))
    
    zipped_data_list = list(zipped_data)

    mae = 0
    count = 0
    for a, b in zipped_data_list:
        mae = mae + abs(a - b)
        count += 1

    
    # mae = mae / float(len(zipped_data))
    mae = mae / float(count)
    

    y_actual = map(lambda x: x[0], zipped_data_list)
    y_hat = map(lambda x: int(lucky_round(x)), map(lambda x: x[1], zipped_data_list))

    y_actual = list(y_actual)
    y_hat = list(y_hat)

    tp, fp, tn, fn = perf_measure(y_actual, y_hat)
    return {"mae": mae, 
        "matthews": matthews_corrcoef(y_actual, y_hat), 
            "f1": f1_score(y_actual, y_hat),
            "precision": precision_score(y_actual, y_hat), "recall": recall_score(y_actual, y_hat),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy_score": accuracy_score(y_actual, y_hat)}

def compute_all_values_ml(gold_standard, prediction_file):
    """
    Compute performance measures
    :param gold_standard: csv to be loaded into a pd.DataFrame
    :param prediction_file: csv to be loaded into a pd.DataFrame
    :param kb:
    :param use_kb:
    :param use_only_kb:
    :return:
    """
    closed = pd.read_csv(gold_standard)
    testing = pd.read_csv(prediction_file, skiprows=[0], names=["first", "second", "type"])

    data = pd.merge(closed, testing, on=["first", "second"])


    zipped_data = zip(data["type_x"].values.tolist(), data["type_y"].values.tolist())

    mae = 0
    for a, b in zipped_data:
        mae = mae + abs(a - b)

    mae = mae / float(len(zipped_data))

    y_actual = map(lambda x: x[0], zipped_data)
    y_hat = map(lambda x: int(lucky_round(x)), map(lambda x: x[1], zipped_data))

    tp, fp, tn, fn = perf_measure(y_actual, y_hat)
    return {"mae": mae, 
            "matthews": matthews_corrcoef(y_actual, y_hat),
             "f1": f1_score(y_actual, y_hat),
            "precision": precision_score(y_actual, y_hat), "recall": recall_score(y_actual, y_hat),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy_score": accuracy_score(y_actual, y_hat)}


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(list(y_hat))):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)


def lucky_round(val, hold=0.5):
    if val >= hold:
        return 1
    else:
        return 0

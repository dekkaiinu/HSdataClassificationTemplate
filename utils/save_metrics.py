import csv
import numpy as np

def save_metrics(OA, AA_mean, Kappa, AA, path):
    AA = AA.tolist()
    # metrics = [np.round(OA, 4), np.round(AA_mean, 4), np.round(Kappa, 4), np.round(AA, 4)]
    metrics = {"OA": OA, "AA_mean": AA_mean, "Kappa": Kappa, "AA": AA}
    with open(path + "/metrics.csv", mode='w', newline='') as csvfile:
        fieldnames = ['OA', 'AAmean', 'Kappa', 'AA']
        writer = csv.writer(csvfile, delimiter=',')
        
        # Headerを書き込む
        formatted_fieldnames = [f"{name:>10}" for name in fieldnames]  # 各フィールド名を20文字の幅で右揃え
        writer.writerow(formatted_fieldnames)
        formatted_row = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:>10.4f}"
                formatted_row.append(formatted_value)
            elif isinstance(value, list):
                for i, element in enumerate(value):
                    if isinstance(element, float):
                        if i == 0:
                            formatted_element = f"        [{element:>.4f}"
                        elif i == len(value) - 1:
                            formatted_element = f" {element:>.4f}]"
                        else:
                            formatted_element = f" {element:>.4f}"
                        formatted_row.append(formatted_element)
        writer.writerow(formatted_row)
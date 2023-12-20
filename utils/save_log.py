import csv

def save_log(csv_file_path, log_data):
    with open(csv_file_path, mode='w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']
        writer = csv.writer(csvfile, delimiter=',')
        
        # Headerを書き込む
        formatted_fieldnames = [f"{name:>20}" for name in fieldnames]  # 各フィールド名を20文字の幅で右揃え
        writer.writerow(formatted_fieldnames)
        
        for row in log_data:
            formatted_row = []
            for key, value in row.items():
                if isinstance(value, int):
                    formatted_value = f"{value:>20}"  # intの場合、20文字の幅で右揃え
                elif isinstance(value, float):
                    formatted_value = f"{value:>20.4f}"  # floatの場合、20文字の幅で小数点以下4桁、右揃え
                else:
                    formatted_value = f"{value:>20}"  # その他の場合、20文字の幅で右揃え
                    
                formatted_row.append(formatted_value)
                
            writer.writerow(formatted_row)
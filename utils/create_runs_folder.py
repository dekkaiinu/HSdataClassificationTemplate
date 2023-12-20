import os
import datetime

def create_runs_folder():
    current_datetime = datetime.datetime.now()
    date_folder_name = "runs/" + current_datetime.strftime("%Y-%m-%d")
    time_folder_name = current_datetime.strftime("%H-%M")

    if not os.path.exists(date_folder_name):
        os.makedirs(date_folder_name)

    time_folder_path = os.path.join(date_folder_name, time_folder_name)
    if not os.path.exists(time_folder_path):
        os.makedirs(time_folder_path)
    if not os.path.exists(time_folder_path + "/train"):
        os.makedirs(time_folder_path + "/train")
    if not os.path.exists(time_folder_path + "/val"):
        os.makedirs(time_folder_path + "/val")
    if not os.path.exists(time_folder_path + "/test"):
        os.makedirs(time_folder_path + "/test")
    return time_folder_path
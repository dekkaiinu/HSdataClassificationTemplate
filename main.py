from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn as nn

# from dataset_loader import dataset_loader

from utils.create_runs_folder import *
from data.data_prepro import data_prepro
from data.torch_data_loader import torch_data_loader
from model_choice import *
from train import train
from test import test

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    runs_path = create_runs_folder()

    # データの用意（train, validation, test）numpy配列で(データ数, 151(次元数))の形にする
    # X_train, X_val, X_test, y_train, y_val, y_test = torch_data_loader()

    # データの前処理
    X_train, X_val, X_test = data_prepro(X_train, X_val, X_test, cfg.prepro)


    # pytorch学習用のデータセット作成
    label_train_loader, label_val_loader, label_test_loader = torch_data_loader(X_train, X_val, X_test, y_train, y_val, y_test, cfg.train_parameter)

    # モデル，学習用のハイパラを定義
    input_dim = X_train.shape[1]
    output_dim = cfg.dataload.class_num
    model = model_choice(input_dim, output_dim, cfg.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # criterion
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_parameter.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=, gamma=)

    # 学習
    model = train(model, cfg.train_parameter.epochs, criterion, optimizer, label_train_loader, label_val_loader, runs_path)

    # 評価
    test(model, label_test_loader, criterion, optimizer, runs_path)


if __name__ == "__main__":
    main()
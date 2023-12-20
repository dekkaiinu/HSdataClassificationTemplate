import torch
from collections import Counter
import numpy as np
from utils.output_metric import *
from utils.AvgrageMeter import *
from utils.accuracy import *
from utils.save_log import *
from utils.save_confusion_matrix import *
from utils.save_metrics import *
from sklearn.metrics import accuracy_score

def test(model, label_test_loader, criterion, optimizer, run_path):
    model.eval()
    tar_t, pre_t = test_epoch(model, label_test_loader, criterion, optimizer)
    OA, AA_mean, Kappa, AA = output_metric(tar_t, pre_t)

    cm_save(pre_t, tar_t, run_path + "/test/")
    save_metrics(OA, AA_mean, Kappa, AA, run_path + "/test/")

def test_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)
        
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre

def img_acc(model, X_test_list, y_test_list):
    y_pred_in_img = []
    for X_test, y_test in zip(X_test_list, y_test_list):
        X_test = pix_wise_std(X_test)
        X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
        X_test = X_test.cuda()
        
        preds = model(X_test)

        topk=(1,)
        maxk = max(topk)
        _, pred = preds.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = pred.data.cpu().numpy()
        pred = np.squeeze(pred, axis=0)

        pred_list = pred.tolist()
        count = Counter(pred_list)
        mode_pred = count.most_common(1)
        y_pred_in_img.append(mode_pred[0][0])
    return y_pred_in_img, y_test_list

def pix_wise_std(X):
    mean_vals = X.mean(axis=1, keepdims=True)
    std_vals = X.std(axis=1, keepdims=True)
    X_norm = (X - mean_vals) / std_vals
    return X_norm

    

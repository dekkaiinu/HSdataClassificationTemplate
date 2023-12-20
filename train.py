import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.output_metric import *
from utils.AvgrageMeter import *
from utils.accuracy import *
from utils.save_log import *
from utils.save_metrics import *
from utils.save_confusion_matrix import * 

def train(model, epochs, criterion, optimizer, label_train_loader, label_val_loader, run_path):
    print("**************************************************")
    print("start training")
    tic = time.time()
    log_data = []

    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        # train
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        
        # val
        model.eval()
        val_acc, val_obj, tar_v, pre_v = valid_epoch(model, label_val_loader, criterion, optimizer)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

        tqdm.write("Epoch [{}/{}], train_loss: {:.4f} train_acc: {:.4f}, val_loss: {:.4f} val_acc: {:.4f}"
                        .format(epoch+1, epochs, train_obj, train_acc, val_obj, val_acc))

        log_entry = {
            'Epoch': epoch + 1,
            'Train Loss': np.round(train_obj.data.cpu().numpy(), 4),
            'Train Accuracy': np.round(train_acc.data.cpu().numpy(), 4),
            'Validation Loss': np.round(val_obj.data.cpu().numpy(), 4),
            'Validation Accuracy': np.round(val_acc.data.cpu().numpy(), 4)
        }
        log_data.append(log_entry)

        save_log(run_path + '/training_log.csv', log_data)
        torch.save(model.state_dict(), run_path + "/weight.pt")
    
    plot_learning_curve(log_data, run_path, epochs)
    toc = time.time()

    cm_save(pre_t, tar_t, run_path + "/train/")
    save_metrics(OA1, AA_mean1, Kappa1, AA1, run_path + "/train/")
    cm_save(pre_v, tar_v, run_path + "/val/")
    save_metrics(OA2, AA_mean2, Kappa2, AA2, run_path + "/val/")

    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")
    print("Final result:")
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print(AA2)
    print("**************************************************")
    return model



def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in tqdm(enumerate(train_loader), desc="Epoch", leave=False, total=len(train_loader)):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre

def valid_epoch(model, valid_loader, criterion, optimizer):
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
        
    return top1.avg, objs.avg, tar, pre

def plot_learning_curve(log_data, run_path, epochs):
    train_loss = [entry['Train Loss'] for entry in log_data]
    val_loss = [entry['Validation Loss'] for entry in log_data]
    train_acc = [entry['Train Accuracy'] for entry in log_data]
    val_acc = [entry['Validation Accuracy'] for entry in log_data]

    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_acc, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(run_path + "/learning_curve.png")
    plt.show()
import torch
import torch.utils.data as Data

def torch_data_loader(X_train, X_val, X_test, y_train, y_val, y_test, train_parameter):
    print("**************************************************")
    print("hyperparams")
    print("epochs:{}, batch size:{}".format(train_parameter.epochs, train_parameter.batch_size))
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    Label_train = Data.TensorDataset(X_train, y_train)

    X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
    y_val = torch.from_numpy(y_val).type(torch.LongTensor)
    Label_val = Data.TensorDataset(X_val, y_val)

    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    Label_test = Data.TensorDataset(X_test, y_test)


    label_train_loader = Data.DataLoader(Label_train, batch_size=train_parameter.batch_size, shuffle=True)
    label_val_loader = Data.DataLoader(Label_val, batch_size=train_parameter.batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=train_parameter.batch_size, shuffle=True)

    return label_train_loader, label_val_loader, label_test_loader
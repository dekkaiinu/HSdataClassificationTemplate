from models.MLP_BatchNorm import MLP_BatchNotm
from models.MLP_InstanceNorm import MLP_InstanceNorm
from models.MCE_ST import MCE_ST
from models.CNN1d_BatchNorm import CNN1d_BatchNorm
from models.CNN1d_LayerNorm import CNN1d_layerNorm

def model_choice(input_dim, output_dim, cfg_model):
    print("model:{}".format(cfg_model.name))
    if cfg_model.name == "MLP_BatchNotm":
        model = MLP_BatchNotm(input_dim, output_dim, cfg_model.dropout_prob)
    elif cfg_model.name == "MLP_InstanceNorm":
        model = MLP_InstanceNorm(input_dim, output_dim, cfg_model.dropout_prob)
    elif cfg_model.name == "MCE_ST":
        model = MCE_ST(output_dim, input_dim, cfg_model.emb_size, cfg_model.depth, cfg_model.heads)
    elif cfg_model.name == "CNN1d_BatchNorm":
        model = CNN1d_BatchNorm(input_dim, output_dim, cfg_model.kernel_size, cfg_model.pool_size)
    elif cfg_model.name == "CNN1d_LayerNorm":
        model = CNN1d_layerNorm(input_dim, output_dim, cfg_model.kernel_size, cfg_model.pool_size, cfg_model.dropout_prob)
    else:
        print("error")

    return model
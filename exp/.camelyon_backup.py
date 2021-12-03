import os
import sys
from tensorboardX import SummaryWriter
import torch
sys.path.append('../')
from models.model import DenseNetDann
from core.train import train_dann
from utils.utils import get_data_loader, init_model, init_random_seed

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_test_loader

class Config(object):
    # params for path
    dataset_root = os.path.expanduser(os.path.join('../raw_datasets'))
    model_root = os.path.expanduser(os.path.join('../results/saved_models_camelyon', 'pytorch-DANN'))

    # params for datasets and data loader
    batch_size = 64

    # params for source dataset
    src_dataset = "mnist"
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')
    class_num_src = 31

    # params for target dataset
    tgt_dataset = "mnistm"
    tgt_model_trained = True
    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')
    
    finetune_flag = False
    lr_adjust_flag = 'simple'
    src_only_flag = False

    # params for pretrain
    num_epochs_src = 100
    log_step_src = 10
    save_step_src = 50
    eval_step_src = 20

    # params for training dann
    gpu_id = '0'

    ## for digit
    num_epochs = 5
    log_step = 1
    save_step = 1
    eval_step = 1

    ## for office
    # num_epochs = 1000
    # log_step = 10  # iters
    # save_step = 500
    # eval_step = 5  # epochs

    manual_seed = 8888
    alpha = 0

    # params for optimizing models
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-2

params = Config()

logger = SummaryWriter(params.model_root)

# init random seed
init_random_seed(params.manual_seed)

# init device
device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")

# load dataset
dataset = get_dataset(dataset='camelyon17', download=True)
train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))

src_data_loader = get_train_loader('standard', train_data, batch_size=32)

test_data =  dataset.get_subset('test', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))

t1, t2 = torch.utils.data.random_split(dataset, [42527, 42527], generator=torch.Generator().manual_seed(132))

tgt_data_loader = get_train_loader('standard', t1, batch_size=32)
tgt_data_loader_eval = get_eval_loader('standard', t2, batch_size=32)


# load dann model
dann = init_model(net=MNISTmodel(model_kernel_sizes), restore=None)
print(params.batch_size)
# train dann model
print("Training dann model")
if not (dann.restored and params.dann_restore):
    dann = train_dann(dann, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval, device, logger)

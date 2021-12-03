import os
import random
import sys
from tensorboardX import SummaryWriter
import torch
sys.path.append('../')
from models.model import DenseNetDann
from core.train import train_dann
from core.test import test
from utils.utils import get_data_loader, init_model, init_random_seed
import torchvision.transforms as transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
num_runs = 5
class Config(object):
    # params for path
    dataset_root = os.path.expanduser(os.path.join('../raw_datasets'))
    model_root = os.path.expanduser(os.path.join('../results/saved_models_camelyon', 'pytorch-DANN'))

    # params for datasets and data loader
    batch_size = 32
    src_dataset = "camelyon_train"
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt') 
    
    tgt_dataset = "camelyon_target"
    tgt_dataset_trained = True
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
    log_step = 300
    save_step = 1
    eval_step = 1

    ## for office
    # num_epochs = 1000
    # log_step = 10  # iters
    # save_step = 500
    # eval_step = 5  # epochs

    manual_seed = None
    alpha = 0

    # params for optimizing models
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-2

params = Config()

logger = SummaryWriter(params.model_root)

# init random seed

# init device
device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")

# load dataset
dataset = get_dataset(dataset='camelyon17', download=True)
train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((96,96)), transforms.ToTensor()]))

src_data_loader = get_train_loader('standard', train_data, batch_size=32, num_workers=8)

val_data = dataset.get_subset('id_val', transform=transforms.Compose([transforms.Resize((96,96)), transforms.ToTensor()]))
src_eval_data_loader = get_eval_loader('standard', val_data, batch_size=32, num_workers=8)


test_data =  dataset.get_subset('test', transform=transforms.Compose([transforms.Resize((96,96)), transforms.ToTensor()]))

size = int(len(test_data)* 4/5)


t1, t2 = torch.utils.data.random_split(test_data, [size, len(test_data)-size], generator=torch.Generator().manual_seed(132))
t1.collate = None
t2.collate = None
tgt_data_loader = get_train_loader('standard', t1, batch_size=32, num_workers=8)
tgt_data_loader_eval = get_eval_loader('standard', t2, batch_size=32, num_workers=8)
seeds = [2087, 2104, 3004]
for i in range(len(seeds)):
    # load dann model
    seed = seeds[i]
    init_random_seed(seed)
    
    params.dataset_root = os.path.expanduser(os.path.join('../raw_datasets'))
    params.model_root = os.path.expanduser(os.path.join('../results/saved_models_camelyon_' + str(seed), 'pytorch-DANN'))
    for i in range(1,6): 
        params.dann_restore = os.path.join(params.model_root, params.src_dataset + '-' + params.tgt_dataset + '-dann-' + str(i)+'.pt')

        dann = init_model(net=DenseNetDann(32, (6, 12, 24, 16), 64, num_classes=2), restore=params.dann_restore)
        # train dann model
        print("Testing Model with seed " + str(i))
        test(dann,src_eval_data_loader, device, False) 

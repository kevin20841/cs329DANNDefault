import os
import random
import sys
from tensorboardX import SummaryWriter
import torch
sys.path.append('../')
from models.model import DenseNetDann
from core.train import train_dann
from utils.utils import get_data_loader, init_model, init_random_seed
import torchvision.transforms as transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
num_runs = 10
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



# init random seed

for i in range(num_runs):
    # init device
    device = torch.device("cuda:" + "0"  if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = get_dataset(dataset='camelyon17', download=True)
    train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((96,96)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))

    src_data_loader = get_train_loader('standard', train_data, batch_size=32, num_workers=8)

    val_data = dataset.get_subset('id_val', transform=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5))]))
    src_eval_data_loader = get_eval_loader('standard', val_data, batch_size=32, num_workers=8)


    test_data =  dataset.get_subset('test', transform=transforms.Compose([transforms.Resize((96,96)), transforms.ToTensor()]))

    size = int(len(test_data)* 4/5)


    t1, t2 = torch.utils.data.random_split(test_data, [size, len(test_data)-size], generator=torch.Generator().manual_seed(132))
    t1.collate = None
    t2.collate = None
    tgt_data_loader = get_train_loader('standard', t1, batch_size=32, num_workers=8)
    tgt_data_loader_eval = get_eval_loader('standard', t2, batch_size=32, num_workers=8)

    # load dann model
    
    params = Config()
    
    logger = SummaryWriter(params.model_root)
    seed = random.randint(1, 10000)
    init_random_seed(seed)
    params.dataset_root = os.path.expanduser(os.path.join('../raw_datasets'))
    params.model_root = os.path.expanduser(os.path.join('../results/saved_models_camelyon_' + str(seed), 'pytorch-DANN'))
    
    dann = init_model(net=DenseNetDann(32, (6, 12, 24, 16), 64, num_classes=2), restore=None)
    # train dann model
    print("Training dann model with seed ", seed) 
    dann = train_dann(dann, params, src_data_loader, src_eval_data_loader, tgt_data_loader, tgt_data_loader_eval, device, logger)
    del dann

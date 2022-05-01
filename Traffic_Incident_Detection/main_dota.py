from __future__ import print_function

import torch.nn as nn

from cfg import parser
from core.utils import *
from core.region_loss import RegionLoss
from core.model import YOWO, get_fine_tuning_parameters
from core.optimization import *

################# CHECKPOINTING ######################
######################################################

import wandb

args  = parser.parse_args()
cfg   = parser.load_config(args)

# logging and checkpoint directories
run_name = cfg.BACKUP_DIR.split('/')[-1]
wandb_id_file_path = pathlib.Path("./_wandb_runid_" + run_name + ".txt")
wandb.login()
config = init_or_resume_wandb_run(wandb_id_file_path, entity_name="vector_cv" , project_name="Dota_Report", run_name=run_name)

####### Load configuration arguments
# ---------------------------------------------------------------
config = wandb.config
config.update(cfg, allow_val_change=True)

####### Check backup directory, create if necessary
# ---------------------------------------------------------------
if not os.path.exists(cfg.BACKUP_DIR):
    os.makedirs(cfg.BACKUP_DIR)


####### Create model
# ---------------------------------------------------------------
model = YOWO(cfg)
model = model.cuda()

# Multi- or single gpu
model = nn.DataParallel(model) # in multi-gpu case
model = model.cuda() # in single-gpu case
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging('Total number of trainable parameters: {}'.format(pytorch_total_params))


####### Create optimizer
# ---------------------------------------------------------------
parameters = get_fine_tuning_parameters(model, cfg)
optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
best_score   = 0 # initialize best score
# optimizer = optim.SGD(parameters, lr=cfg.TRAIN.LEARNING_RATE/batch_size, momentum=cfg.SOLVER.MOMENTUM, dampening=0, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

seed = int(time.time())
batch = 0

####### Load resume path if necessary
# ---------------------------------------------------------------
if cfg.TRAIN.RESUME_PATH:
    print("===================================================================")
    print('loading checkpoint {}'.format(cfg.TRAIN.RESUME_PATH))
    checkpoint = torch.load(cfg.TRAIN.RESUME_PATH)
    batch = checkpoint['batch']
    print("àn > this is batch", batch)
    if batch == 0:
        cfg.TRAIN.BEGIN_EPOCH = checkpoint['epoch'] + 1
    else:
        cfg.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
    best_score = checkpoint['score']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    seed = checkpoint['seed']
    
    print("Loaded model score: ", checkpoint['score'])
    print("===================================================================")
    del checkpoint


torch.manual_seed(seed)
use_cuda = True
    

print('looking for file...')
continue_file_path = '%s/%s_checkpoint.pth' % (cfg.BACKUP_DIR, 'yowo_' + cfg.TRAIN.DATASET + '_' + str(cfg.DATA.NUM_FRAMES) + 'f')
print(continue_file_path)
if os.path.exists(continue_file_path):
    print("===================================================================")
    print('resuming from checkpoint {}'.format(continue_file_path))
    checkpoint = torch.load(continue_file_path)
    cfg.TRAIN.BEGIN_EPOCH = checkpoint['epoch'] + 1
    best_score = checkpoint['score']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded model fscore: ", checkpoint['score'])
    print("===================================================================")
else:
    print('No checkpoint found, training from scratch!')


####### Create backup directory if necessary
# ---------------------------------------------------------------
if not os.path.exists(cfg.BACKUP_DIR):
    os.mkdir(cfg.BACKUP_DIR)

from dataset_factory.dota import DoTA
import yaml
from argparse import Namespace
with open("cfg/dota_config.yaml", 'r') as f:
    dl_args = yaml.load(f)

dl_args = Namespace(**dl_args)

# set data paths for local:
if not os.path.exists(dl_args.root):
    print('did not find data! -------------')
    sys.exit()



train_dataset = DoTA(dl_args, phase="train", n_frames=16, combined_bbox=False)
test_dataset = DoTA(dl_args, phase="test", n_frames=16, combined_bbox=False)

train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                            num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=True, pin_memory=True)
test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                            num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)
loss_module   = RegionLoss(cfg).cuda()

train = getattr(sys.modules[__name__], 'train')
test  = getattr(sys.modules[__name__], 'test_dota')

    
def checkpointer(seed, epoch, batch, model, optimizer, score, is_best, cfg):
    state = {
        'seed': seed,
        'batch': batch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'score': score
    }
    save_checkpoint(state, is_best, cfg.BACKUP_DIR, cfg.TRAIN.DATASET, cfg.DATA.NUM_FRAMES)
    return state   

score = 0
is_best = False

####### Training and Testing Schedule
# ---------------------------------------------------------------
if cfg.TRAIN.EVALUATE:
    logging('evaluating ...')
    test(cfg, 999, model, test_loader)
else:
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH + 1):
        # Adjust learning rate
        lr_new = adjust_learning_rate(optimizer, epoch, cfg)
        wandb.log({'lr': lr_new}) 
        # Train and test model
        logging('training at epoch %d, lr %f' % (epoch, lr_new))
        train(cfg, epoch, model, train_loader, loss_module, optimizer,
              seed=seed, score=score, is_best=is_best, checkpointer=checkpointer)
        if epoch % 5 == 0:
            logging('testing at epoch %d' % (epoch))
            score = test(cfg, epoch, model, test_loader)
            wandb.log({'epoch': epoch, 'fscore': score})
            is_best = score > best_score
            if is_best:
                print("New best score is achieved: ", score)
                print("Previous score was: ", best_score)
                best_score = score

        checkpointer(seed, epoch, 0, model, optimizer, score, is_best, cfg)
        logging('Weights are saved to backup directory: %s' % (cfg.BACKUP_DIR))

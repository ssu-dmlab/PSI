import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from LFM import MFWithBias

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# Load pretrained LFM
if world.config['use_lfm'] == 1:
    print("Loading pretrained LFM...")
    lfm_ckpt_path = "LFM_checkpoints/amazon-book_False_dim64.pt"
    lfm_ckpt = torch.load(lfm_ckpt_path, map_location=world.device)
    
    lfm_model = MFWithBias(
        n_users=lfm_ckpt['n_users'],
        n_items=lfm_ckpt['n_items'],
        dim=lfm_ckpt['dim']
    ).to(world.device)
    lfm_model.load_state_dict(lfm_ckpt['state_dict'])
    lfm_model.eval()

    # 모든 user-item 쌍에 대해 rating 캐시 생성
    n_users = lfm_ckpt['n_users']
    n_items = lfm_ckpt['n_items']
    with torch.no_grad():
        # user/item 인덱스 생성
        user_ids = torch.arange(n_users, device=world.device)
        item_ids = torch.arange(n_items, device=world.device)
        # user-item meshgrid
        user_grid, item_grid = torch.meshgrid(user_ids, item_ids, indexing='ij')
        # (n_users, n_items) → flatten
        lfm_ratings = lfm_model(user_grid.flatten(), item_grid.flatten())
        lfm_ratings = lfm_ratings.view(n_users, n_items).cpu()
        
        # interaction 있는 아이템은 매우 낮은 값으로 설정
        min_value = torch.finfo(torch.float32).min
        for u, pos_items in enumerate(dataset.allPos):
            lfm_ratings[u, pos_items] = min_value
else:
    lfm_model = None
    lfm_ratings = None

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        with utils.timer(name="Train"):
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w, lfm_ratings=lfm_ratings)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}-{utils.timer.dict()}')
        utils.timer.zero()
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
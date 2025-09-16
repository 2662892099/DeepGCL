import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
# from model_pcl import ZHC
# from MMPD_DTA1 import ZHC
# from MMPD_DTA1 import ZHC
# from MMPD_DTA3 import ZHC
# from MMPD_DTA4 import ZHC
from MMPD_DTA5 import ZHC,graphcl,GNN #完整架构模型
# from MMPD_DTA5_noProtein import ZHC,graphcl,GNN
# from MMPD_DTA5_noPD import ZHC,graphcl,GNN
#from MMPD_DTA5_noLigand import ZHC,graphcl,GNN
import metrics
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
from dataset_two_graph import TestbedDataset
# from dataset1 import TestbedDataset
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import   CosineAnnealingLR
import torch.nn.functional as F


def train(model, device, train_loader, optimizer,loss_fn):
    # scheduler = CosineAnnealingLR(optimizer, T_max=T_max)  # T_max 是余弦周期的长度
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in tqdm(enumerate(train_loader), disable=False, total=len(train_loader)):
        # print(data)
        data = data.to(device)
        optimizer.zero_grad()

       


        with autocast():
            
            output,view_loss,ligand_loss = model(data)
            
            loss = loss_fn(output, data.y_s.view(-1, 1).float().to(device))
            
            
            total_loss = loss + 0.5*view_loss*0 + 0.5*ligand_loss
           
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    # scheduler.step()



def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            data = data.to(device)
            y = data.y_s
            
            y_hat,view_loss,ligand_loss = model(data)
            
            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            
            test_loss += 0.5*view_loss.item()*0
            test_loss += 0.5*ligand_loss.item()
            
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    
    test_loss /= len(test_loader.dataset)

   
    
    evaluation = {
        
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation







seed = 42 ##random



# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.enable =True
# torch.backends.cudnn.benchmark = True



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(seed)

scaler = GradScaler()


device = torch.device("cuda")


model = ZHC().to(device)

loss_fn = nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(),lr=0.0008)
optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=1e-6)#默认0.001



# torch.optim.SGD

data_loaders = {phase_name:
                DataLoader(TestbedDataset(root='data', dataset=phase_name),
                           batch_size= 64  ,
                           pin_memory=True,
                           # num_workers=8,
                           shuffle=True,follow_batch=['x_s','x_t'])
            # for phase_name in ['train_without_inter','val_without_inter','test2016_without_inter','test2013_without_inter','BDB2020_without_inter']}
            for phase_name in ['train','val','test2016','test2013','BDBbind']}
time_now = datetime.now()
path = Path(f'result/MMDTA_{datetime.now().strftime("%Y%m%d%H%M%S")}_xiaorong_no_PD') # 这次PD图部分的数据用GIN共享编码器进行特征提取
# path = Path(f'/home/zhc/桌面/pythonzhc/complex_graph/result/edge_xiao')
path = Path(f'result/MMDTA_{datetime.now().strftime("%Y%m%d%H%M%S")}_bestmodeltest')
writer = SummaryWriter(path)
# NUM_EPOCHS = 140
NUM_EPOCHS = 350

# save_best_epoch=130
save_best_epoch= 230
best_val_loss = 100000000
best_epoch = -1
patience = 10 #设定耐心值
patience_count = 0 # 初始化耐心计数器
# scheduler = CosineAnnealingLR(optimizer,T_max=NUM_EPOCHS//2)

start = datetime.now()
print('start at ', start)

for epoch in range(1,NUM_EPOCHS+1):

    train(model, device, data_loaders['train'], optimizer, loss_fn)

    performances = []

    for _p in ['train','val','test2016','test2013','BDBbind']:
        performance = test(model, data_loaders[_p], loss_fn, device, False)
        performances.append(performance)
        print("epoch=",epoch,performance)
        for i in performance:
            writer.add_scalar(f'{_p} {i}', performance[i], global_step=epoch)
        if _p == 'val' and epoch >= save_best_epoch and performance['loss'] < best_val_loss:
            best_val_loss = performance['loss']
            best_epoch = epoch
            torch.save(model.state_dict(), path / 'best_model.pt')
        # 在训练到一定 epoch 之后，每次训练结束后保存模型和指标文件
        if epoch >= save_best_epoch and _p=='BDBbind':
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # print(current_time)

            # 保存模型
            model_filename = f'{current_time}_{epoch}_best_model.pt'
            model_path = path / model_filename
            torch.save(model.state_dict(), model_path)
            # print(f"Model saved at {model_path}")

            # 保存指标文件
            result_filename = f'{current_time}_{epoch}_result.txt'
            result_path = path / result_filename

            with open(result_path, 'w') as f:
                f.write(f'Model saved at epoch NO.{epoch}\n')
                for performance in performances:
                    for k, v in performance.items():
                        f.write(f'{k}: {v}\n')
                        # print(f'{k}: {v}\n')
                    f.write('\n')


model.load_state_dict(torch.load(path / 'best_model.pt'))
with open(path / 'result.txt', 'w') as f:
    f.write(f'best model found at epoch NO.{best_epoch}\n')
    for _p in ['train','val','test2016','test2013','BDBbind']:
        performance = test(model, data_loaders[_p], loss_fn, device, True)
        f.write(f'{_p}:\n')
        print(f'{_p}:')
        for k, v in performance.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')
        f.write('\n')
        print()

print('train finished')

end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
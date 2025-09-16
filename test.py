import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from model_pcl import ZHC
# from MMPD_DTA5 import ZHC
# from MMPD_DTA5 import ZHC
# from MMPD_DTA5_noProtein import ZHC
from MMPD_DTA5_zhenglizhushi import ZHC
import metrics
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
from dataset_two_graph import TestbedDataset
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import pandas as pd


# def train(model, device, train_loader, optimizer,loss_fn):
#     print('Training on {} samples...'.format(len(train_loader.dataset)))
#     model.train()
#     for batch_idx, data in tqdm(enumerate(train_loader), disable=False, total=len(train_loader)):
#         # print(data)
#         data = data.to(device)
#         optimizer.zero_grad()
#
#         # output = model(data)
#         # loss = loss_fn(output, data.y_s.view(-1, 1).to(device))
#         # loss.backward()
#         # optimizer.step()
#         # # scheduler.step()
#
#
#         with autocast():
#             output = model(data)
#             loss = loss_fn(output, data.y_s.view(-1, 1).float().to(device))
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            data = data.to(device)
            y = data.y_s
            # y_hat,z1,z2,x1,x2 = model(data)
            # y_hat,z1,z2 = model(data)
            y_hat, view_loss, ligand_loss,initial_features,multimodal_features,final_features = model(data)
            # print(y,y_hat)
            # y_hat = model(data)
            # print(y,y_hat)
            # y_hat,gcl_loss = model(data)
            # test_loss += 0.4*compute_loss(z1,z2).item()
            # test_loss += 0.3*compute_loss(x1,x2).item()
            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            # test_loss += 0.5*loss_function(y_hat.view(-1), y.view(-1)).item()
            # test_loss += 0.5*loss_function(y_hat.view(-1), y.view(-1)).item() + 0.5*view_loss.item()
            # test_loss += 0.6*loss_function(y_hat.view(-1),y.view(-1)).item()
            # test_loss += 0.5*loss_function(y_hat.view(-1), y.view(-1)).item()
            # test_loss += 0.5*model.complex_graph.loss_cl(z1,z2).item()
            test_loss += 0.5 * view_loss.item()
            test_loss += 0.5 * ligand_loss.item()
            # test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
            # print(outputs,targets)

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    # print("比较",targets,outputs)
    df = pd.DataFrame({"True":targets,"predicted":outputs})
    df.to_csv("results_"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv",index=False)
    # gcl_loss /= len(test_loader.dataset)
    # mse_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)

    # if np.isnan(outputs).any() or np.isnan(targets).any():
    #     outputs = np.nan_to_num(outputs)
    #     targets = np.nan_to_num(targets)

    evaluation = {
        # 'gcl_loss':gcl_loss,
        # 'mse_loss':mse_loss,
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation
#
# def test(model: nn.Module, test_loader, loss_function, device, show):
#     model.eval()
#     test_loss = 0
#     outputs = []
#     targets = []
#     with torch.no_grad():
#         for batch_idx, data in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
#             data = data.to(device)
#             y = data.y_s
#             y_hat = model(data)
#             test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
#             outputs.append(y_hat.cpu().numpy().reshape(-1))
#             targets.append(y.cpu().numpy().reshape(-1))
#
#     targets = np.concatenate(targets).reshape(-1)
#     outputs = np.concatenate(outputs).reshape(-1)
#
#     test_loss /= len(test_loader.dataset)
#
#     evaluation = {
#         'loss': test_loss,
#         'c_index': metrics.c_index(targets, outputs),
#         'RMSE': metrics.RMSE(targets, outputs),
#         'MAE': metrics.MAE(targets, outputs),
#         'SD': metrics.SD(targets, outputs),
#         'CORR': metrics.CORR(targets, outputs),
#     }
#
#     return evaluation







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
optimizer = torch.optim.AdamW(model.parameters())

data_loaders = {phase_name:
                DataLoader(TestbedDataset(root='data', dataset=phase_name),
                           batch_size= 64  ,
                           pin_memory=True,
                           # num_workers=8,
                           shuffle=True,follow_batch=['x_s','x_t'])
            for phase_name in ['train','val','test2016','test2013','BDBbind']}



# model.load_state_dict(torch.load('/media/zhc/新加卷1/pythonzhc/complex_graph/result/1/best/best_model.pt'))
# model.load_state_dict(torch.load('/home/zhc/sss/complex_graph/result/best/best_model.pt'))
# model.load_state_dict(torch.load('/home/zhc/sss/complex_graph1/result/MMDTA_20250112135502_消融_图对比数据增强共享编码器提取PD图，transformer用于蛋白质序列，添加了配体数据的图对比提取特征/best_model.pt'))
# model.load_state_dict(torch.load('/home/zhc/sss/complex_graph1/result/MMDTA_20250110200018_消融_图对比数据增强共享编码器提取PD图，transformer用于蛋白质序列，添加了配体数据的图对比提取特征/best_model.pt'))
# model.load_state_dict(torch.load('/home/zhc/sss/complex_graph1/result/MMDTA_20250309100952_消融_图对比数据增强共享编码器提取PD图，transformer用于蛋白质序列，添加了配体数据的图对比提取特征 测试多次训练取平均1/best_model.pt'))
# model.load_state_dict(torch.load('/home/zhc/sss/complex_graph1/result/best/best_model1.pt'))
model.load_state_dict(torch.load('/home/zhc/sss/complex_graph1/result/best/best_model1.pt'))



for _p in ['train','val','test2016','test2013','BDBbind']:
    print(_p)
    performance = test(model, data_loaders[_p], loss_fn, device, False)
    # with open('test.txt','w') as f :
    #     f.write(f'{_p}:\n')
    #     # print(f'{_p}:')
    #     for k, v in performance.items():
    #         f.write(f'{k}: {v}\n')
    #         print(f'{k}: {v}\n')
    #     f.write('\n')
    print(performance)

# with open('test.txt, 'a') as f:
#     for _p in ['train','val','test2016','test2013']:
#         performance = test(model, data_loaders[_p], loss_fn, device, True)
#         f.write(f'{_p}:\n')
#         print(f'{_p}:')
#         for k, v in performance.items():
#             f.write(f'{k}: {v}\n')
#             print(f'{k}: {v}\n')
#         f.write('\n')
#         print()


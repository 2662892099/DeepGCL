import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU,Dropout
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool as gmp,GATConv,global_mean_pool,GCNConv,SAGEConv,GATv2Conv,MessagePassing
import numpy as np
from torch_geometric.utils import dropout_edge,add_self_loops
# from torch_geometric.utils import dropout_adj
from mamba_ssm import Mamba2
from mamba_ssm import Mamba
from mamba_ssm.models.mixer_seq_simple import MixerModel
# import selective_scan_cuda
import random
from collections import defaultdict
from mambaDemo import ResidualGroup

# d_model=60          #the number of expected features in the input
d_model= 60
dim_feedforward = 256          #the dimension of the feedforward network model
n_heads = 2         #the number of heads in the multiheadattention models
n_layers=4          # the number of TransformerEncoderLayers in each block
class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src):
        output = src
        for mod in self.layers:
            output,attn = mod(output)
        if self.norm is not None:
            output = self.norm(output)
        return output,attn
class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first']
    def __init__(self, d_model, nhead, dim_feedforward=dim_feedforward, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=batch_first)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)# **factory_kwargs 允许传递额外的关键字参数
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src):
        src2,attn = self.self_attn(src, src, src)
        src = src + self.dropout1(src2) #进行了一次残差操作
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attn[:,0,:]
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)
class GIN(torch.nn.Module):
    '''
    4-layer GCN model class.
    '''

    def __init__(self, c_feature=108,MLP_dim=96):
        super(GIN, self).__init__()
        nn1 = Sequential(Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(MLP_dim)
        nn2 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(MLP_dim)
        nn3 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(MLP_dim)
        nn4 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(MLP_dim)
        nn5 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(MLP_dim)

        self.lin = Linear(MLP_dim, 120)
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        # x = self.conv4(x, edge_index)
        # x = x.relu()
        # x = self.conv5(x, edge_index)

        x = global_add_pool(x, batch)

        x = F.dropout(x, p=0.1)
        x = self.lin(x)

        return x
class GraphSAGE(torch.nn.Module):
    def __init__(self,c_feature=108,MLP_dim=98):
        super().__init__()
        self.conv1 = SAGEConv(c_feature,MLP_dim,aggr='mean')
        self.conv2 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv3 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv4 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv5 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv6 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.conv7 = SAGEConv(MLP_dim, MLP_dim, aggr='mean')
        self.lin = Linear(MLP_dim,120)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        x = global_add_pool(x, batch)

        x = self.lin(x)
        return x


class GCNNet1(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,c_feature=108,MLP_dim=98,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):

        super(GCNNet1, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(c_feature, c_feature)
        self.conv2 = GCNConv(c_feature, c_feature*2)
        self.conv3 = GCNConv(c_feature*2, c_feature * 4)
        # self.conv3 = GCNConv(c_feature*2, c_feature * 2)
        self.fc_g1 = torch.nn.Linear(c_feature*4, 512)
        self.fc_g2 = torch.nn.Linear(512, output_dim)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, x,edge_index,batch):
        # get graph input
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        # target = data.target

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling
        return x

class GCNNet2(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,c_feature=108,MLP_dim=98,num_features_xd=78, num_features_xt=25, output_dim=120, dropout=0.2):

        super(GCNNet2, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(c_feature, c_feature)
        self.conv2 = GCNConv(c_feature, c_feature*2)
        self.conv3 = GCNConv(c_feature*2, c_feature * 4)
        # self.conv3 = GCNConv(c_feature*2, c_feature * 2)
        self.fc_g1 = torch.nn.Linear(c_feature*4, 512)
        self.fc_g2 = torch.nn.Linear(512, output_dim)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, x,edge_index,batch):
        

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling
        #
        # # flatten
        # x = self.relu(self.fc_g1(x))
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        
        return x







class GraphContrastiveLearning(nn.Module):
    def __init__(self, c_feature=108, proj_hidden_dim=256, tau=0.1,out_dim=120,
                 drop_edge_rate_1=0.5,
                 drop_edge_rate_2=0.5,
                 drop_feature_rate=0.5
                 ):
        """
        对比学习模块，包含投影头和损失计算
        feature_dim: 输入特征的维度
        proj_hidden_dim: 投影头隐藏层的维度
        tau: 温度参数
        """
        super(GraphContrastiveLearning, self).__init__()
        self.tau = tau
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate = drop_feature_rate
        self.dropout_edge1 = dropout_edge
        self.dropout_edge2 = dropout_edge
        self.gcc1 = GCNNet1()
        # self.gcc1 = GraphConvNet()
        self.gcc2 = GCNNet1()
       
        self.projection_head = nn.Sequential(
            nn.Linear(c_feature*4, proj_hidden_dim),
            
        )


    def forward(self,x,edge_index,batch):
        """
        前向传播
        x1, x2: 两个视图的图编码特征（来自GNN）
        """
        # 投影到新的特征空间
        # z1 = self.projection_head(x1)
        # z2 = self.projection_head(x2)

        edge_index_1 = self.dropout_edge1(edge_index, p=self.drop_edge_rate_1)[0]
        edge_index_2 = self.dropout_edge2(edge_index, p=self.drop_edge_rate_2)[0]

        x1 = self.drop_feature(x,self.drop_feature_rate)
        x2 = self.drop_feature(x,self.drop_feature_rate)

        h1 = self.gcc1(x1,edge_index_1,batch)
        h2 = self.gcc2(x,edge_index_2,batch)
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        x = torch.cat([z1,z2],dim=1)  #两个120维度
        # loss = self.compute_loss(z1,z2)

        # loss.backward()
        return z1,z2


    def compute_loss(self, z1, z2):
        """
        计算对比损失
        z1, z2: 两个图视图的投影特征
        """
        # 温度缩放后的相似性计算
        batch_size, _ = z1.size()
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 相似度矩阵计算
        sim_matrix = torch.mm(z1, z2.t()) / self.tau
        sim_matrix = torch.exp(sim_matrix)

        # 获取正样本对
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        # 计算对比损失
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()

        return loss


    
    
    def drop_feature(self, x, drop_prob):
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0
        return x

    def drop_nodes(x,edge_index,batch, drop_prob=0.2):
        """随机丢弃节点"""
        num_nodes = x.size(0)
        mask = torch.rand(num_nodes) > drop_prob  # 随机生成掩码
        x = x[mask]
        edge_index = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
        return x,edge_index,batch


    def permute_edges(x,edge_index,batch, permute_prob=0.2):
        """随机扰动边连接"""
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) > permute_prob  # 随机生成掩码
        edge_index = edge_index[:, mask]
        return x,edge_index,batch

    def mask_features(x,edge_index,batch, mask_prob=0.2):
        """随机掩蔽节点特征"""
        mask = torch.rand(x.size()) > mask_prob
        x = x * mask.float()
        return x,edge_index,batch

    def augment_data(x,edge_index,batch, aug_type='node_drop', aug_prob=0.2):
        """根据选择的增强类型对图数据进行增强"""
        # if aug_type == 'node_drop':
        #     return drop_nodes(data, aug_prob)
        # elif aug_type == 'edge_permute':
        #     return permute_edges(data, aug_prob)
        # elif aug_type == 'feature_mask':
        #     return mask_features(data, aug_prob)
        # else:
        #     return data  # 默认不进行增强
        



class SharedModule(nn.Module):
    def __init__(self, input_dim=108, hidden_dim=98, output_dim=120, num_heads=2, dropout=0.3):
        super(SharedModule, self).__init__()
        


        self.input_dim = input_dim #108
        self.hidden_dim = hidden_dim #256
        self.output_dim = output_dim #200
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        
        self.linear2 = Linear(self.hidden_dim,self.hidden_dim)  # 256->256
        self.linear3 = Linear(self.hidden_dim*self.num_heads,self.output_dim) #512->200
        self.linear4 = Linear(self.hidden_dim*self.num_heads,self.hidden_dim*self.num_heads)
        self.linear5 = Linear(self.hidden_dim*self.num_heads*self.num_heads,self.hidden_dim*self.num_heads*self.num_heads)
        self.linear6 = Linear(self.hidden_dim*self.num_heads*self.num_heads,self.output_dim)
        # self.dropout = nn.Dropout(dropout)
        self.relu = nn.RReLU()

        # self.gcn1 = GCNConv(self.input_dim, self.hidden_dim, dropout=self.dropout)
        self.gcn1 = GCNConv(self.input_dim, self.hidden_dim)
        # self.nn1 = Sequential(self.linear2, self.relu, self.dropout,
        #                       self.linear2, self.relu, self.dropout,
        #                       self.linear2)
        self.nn1 = Sequential(self.linear2, self.relu,
                              self.linear2, self.relu, 
                              self.linear2)
        

        
        self.gcn2 = GCNConv(self.hidden_dim,512)
        
        self.nn2 = Sequential(self.linear4,self.relu,
                              self.linear4,self.relu,
                              self.linear4) 
        
        
        self.gcn3 = GCNConv(512,1024)

        self.gcn4 = GCNConv(1024,512)
        self.gcn5 = GCNConv(512,256)
        
        self.out = self.output_dim//2
        
        self.gat1 = GATConv(self.hidden_dim,self.out,heads=self.num_heads,concat=True,dropout=0.2)
        
        self.nn3 = Sequential(self.linear5,self.relu,
                              self.linear5,self.relu,
                              self.linear6)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim*self.num_heads)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim*self.num_heads*self.num_heads)
        self.bn = nn.BatchNorm1d(self.output_dim)

        self.ln = nn.LayerNorm(self.output_dim)

        self.lin = nn.Linear(self.hidden_dim,self.output_dim)


       

        self.gin1 = GINConv(nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)))
        self.gin2 = GINConv(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)))
        self.gin3 = GINConv(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)))
        self.gin4 = GINConv(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)))
        self.gin5 = GINConv(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)))

    def forward(self, x, edge_index, batch):

        """

        :param x:
        :param edge_index:
        :param batch:
        :return:
        """
        '''
            x = self.gcn1(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            # print("gcn1",x.shape)
            x = self.nn1(x)
            # print("nn1",x.shape)
            x = self.gat1(x, edge_index)
            # print("gat1",x.shape)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.nn2(x)
            x = self.gat2(x,edge_index)
            # print("shape",x.shape)
            x = self.relu(x)
            x = self.dropout(x)
            x = global_add_pool(x, batch)
            h = self.nn3(x)
        '''

        """
        
        # 第一个 GCN 块
        x = self.gcn1(x, edge_index)  # [22835,256]
        # x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # print("gcn1",x.shape)
        # x = self.nn1(x)
        x = self.dropout(x)

        
        # print("nn1",x.shape)
        # x = self.gat1(x, edge_index)

        # 第二个 GCN 块
        x = self.gcn2(x,edge_index)  # [22835,512]
        # x = self.bn2(x)
        # print("gcn2",x.shape)
        x = self.relu(x)
        # x = self.dropout(x)
        # x = self.nn2(x)
        x = self.dropout(x)  # [22835,512]


        #第三个GCN
        x = self.gcn3(x,edge_index)  # [22835,1024]
        # x = self.bn3(x)
        # print("gcn3",x.shape)
        x = self.relu(x)
        x = self.dropout(x) # [22835,1024]
        # print("gcn3",x.shape)
        
        #第四个GCN
        x = self.gcn4(x,edge_index)  # [22835,512]
        # x = self.bn4(x)
        # print("gcn4",x.shape)
        x = self.relu(x)
        x = self.dropout(x) # [22835,512]
        # print("gcn4",x.shape)

        #第五个GCN
        x = self.gcn5(x,edge_index)  # [22835,256]
        # x = self.bn5(x)
        # print("gcn5",x.shape)
        x = self.relu(x)
        x = self.dropout(x) # [22835,256]
        # print("gcn5",x.shape)

        """

        x = self.gin1(x,edge_index)
        x = self.relu(x)
        x = self.gin2(x,edge_index)
        x = self.relu(x)
        x = self.gin3(x,edge_index)
        x = self.relu(x)
        x = self.gin4(x,edge_index)
        x = self.relu(x)
        x = self.gin5(x,edge_index)
        x = self.relu(x)

        x = self.lin(x)



        """
        # GAT 块
        x = self.gat1(x, edge_index)  # [22835,120]
        # x = self.bn3(x)
        # print("gat1",x.shape)
        x = self.relu(x)
        x = self.dropout(x)
        """
        # print("After augmentation - x shape:", x.shape)  # 应该显示 [num_nodes, num_features]
        # print("After augmentation - batch shape:", batch.shape)  # 应该显示 [num_nodes]

        # 全局池化和最终变换
        x = global_add_pool(x, batch) #这里x的维度转变为了[64，120]
       
        h = self.ln(x)
        
        return h

class graphcl(nn.Module):

    def __init__(self,c_features = 108,hidden_dim = 256,proj_dim = 300,out_dim = 120,
                 drop_feature_rate=0.2,
                 drop_node_prob=0.2,
                 permute_edge_prob=0.2,
                 mask_feature_prob=0.2,
                 subgraph_prob=0.2,drop_percent=0.2,
                 heads = 2,mode='both'):
        super(graphcl, self).__init__()
        # self.gnn = GNN(num_layer=3,emb_dim=120,JK="last",drop_ratio=0.1,gnn_type="gcn",out_channels=120)
        self.drop_percent = drop_percent
        # 视图1的参数
        self.view1_params = {
            # 'drop_feature_rate': 0,
            'drop_node_prob': self.drop_percent,    # 随机丢弃节点
            'permute_edge_prob': 0,   # 随机扰动边连接
            'mask_feature_prob': 0,   # 随机掩蔽节点特征
            'subgraph_prob': 0      # 子图增强
        }
        # 视图2的参数
        self.view2_params = {
            # 'drop_feature_rate': 0,
            'drop_node_prob': 0,      # 随机丢弃节点
            'permute_edge_prob': 0, # 随机扰动边连接
            'mask_feature_prob': self.drop_percent, # 随机掩蔽节点特征
            'subgraph_prob': 0        # 子图增强
        }

        # 原始数据的视图 不进行数据增强
        self.raw_params = {
            'drop_node_prob': 0,    # 随机丢弃节点
            'permute_edge_prob': 0,   # 随机扰动边连接
            'mask_feature_prob': 0,   # 随机掩蔽节点特征
            'subgraph_prob': 0      # 子图增强
        }

        # 视图3的参数
        self.view3_params = {
            'drop_node_prob': 0,    # 随机丢弃节点
            'permute_edge_prob': self.drop_percent,   # 随机扰动边连接
            'mask_feature_prob': 0,   # 随机掩蔽节点特征
            'subgraph_prob': 0      # 子图增强
        }
        
        #定义一个词典 用来调用视图选择  
        self.view_dict = {
            'first':self.view1_params,
            'second':self.view2_params,
            'raw':self.raw_params,
            'third':self.view3_params
        }
        self.c_features = c_features
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # self.drop_edge_rate_1 = drop_edge_rate_1
        # self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature = drop_feature_rate
        self.drop_node_prob = drop_node_prob
        self.permute_edge_prob = permute_edge_prob
        self.mask_feature_prob  = mask_feature_prob
        self.subgraph_prob = subgraph_prob
        self.heads = heads
        self.mode = mode
        self.linear1 = Linear(c_features,hidden_dim)
        self.linear2 = Linear(hidden_dim,hidden_dim)
        self.linear3 = Linear(hidden_dim,out_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.RReLU()

       
        self.pool = global_mean_pool
        # self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.projection_head = nn.Sequential(nn.Linear(self.out_dim,256),
                                             self.relu,
                                             self.dropout,
                                             nn.Linear(256,self.out_dim)
                                             )

        self.shareModule = SharedModule(input_dim=self.c_features,hidden_dim=self.hidden_dim,output_dim=self.out_dim,num_heads=self.heads,dropout=0.2)
    def forward(self,x,edge_index,batch,view='first'):

        # x,edge_index,batch = apply_augmentation(x=x,edge_index=edge_index,batch=batch)
        # params = self.view1_params if view == 'first' else self.view2_params
        params = self.view_dict[view]
        # print("x shape前",x.shape)
        # print("edge_index shape前",edge_index.shape)
        # print("batch shape前",batch.shape)
        x, edge_index, batch = self.apply_augmentation(
                    x, edge_index, batch,
                    #drop_feature_rate=params['drop_feature_rate'],
                    drop_node_prob=params['drop_node_prob'],
                    permute_edge_prob=params['permute_edge_prob'],
                    mask_feature_prob=params['mask_feature_prob'],
                    subgraph_prob=params['subgraph_prob']
                )
        
        h = self.shareModule(x,edge_index,batch)
        

        z = self.projection_head(h)
        
        return z
    
    def apply_augmentation(self, x, edge_index, batch, **params):
        """
        应用数据增强。
        
        参数:
        - x: 输入特征 [num_nodes, num_features]
        - edge_index: 边索引 [2, num_edges]
        - batch: 批处理索引
        
        返回:
        - x: 增强后的特征 [num_nodes, num_features]
        - edge_index: 增强后的边索引 [2, num_edges]
        - batch: 批处理索引
        """
        if params.get('drop_node_prob', 0) > 0:
            x, edge_index,batch = self.aug_drop_node(x, edge_index,batch, params['drop_node_prob'])

        if params.get('drop_feature_rate', 0) > 0:
            x,edge_index,batch = self.aug_random_mask(x,edge_index,batch, params['drop_feature_rate'])
        # if params.get('mask_feature_prob', 0) > 0:
        #     x,edge_index,batch = self.aug_random_mask(x,edge_index,batch, params['mask_feature_prob'])
            
        if params.get('permute_edge_prob', 0) > 0:
            x,edge_index,batch = self.aug_random_edge(x,edge_index,batch, params['permute_edge_prob'])
            
        
            
        if params.get('subgraph_prob', 0) > 0:
            x, edge_index,batch = self.aug_subgraph(x, edge_index,batch, params['subgraph_prob'])
            
        return x, edge_index, batch
    
    def aug_random_mask(self, x,edge_index,batch, drop_percent=0.2):
        """
        随机掩蔽节点特征。
        
        参数:
        - x: [num_nodes, num_features]
        """
        node_num, feat_dim = x.shape
        mask_num = int(feat_dim * drop_percent)
        
        # 对每个节点独立进行特征掩码
        for i in range(node_num):
            mask_idx = random.sample(range(feat_dim), mask_num)
            x[i, mask_idx] = 0
            
        return x,edge_index,batch

    def aug_random_edge(self,x,edge_index,batch, drop_percent=0.2):
        """
        随机扰动边连接。
        
        参数:
        - edge_index: 边索引张量 [2, num_edges]
        - drop_percent: 边扰动的百分比
        
        返回:
        - new_edge_index: 扰动后的边索引张量
        """
        device = edge_index.device
        num_edges = edge_index.size(1)
        
        # 计算需要扰动的边的数量
        num_perturb = int(num_edges * drop_percent)
        
        # 随机选择要删除的边
        perm = torch.randperm(num_edges, device=device)
        preserved_edges = perm[num_perturb:]
        
        # 保留选定的边
        new_edge_index = edge_index[:, preserved_edges]
        
        # 获取节点数量（通过边索引中的最大值）
        num_nodes = edge_index.max().item() + 1
        
        # 随机添加新边
        for _ in range(num_perturb):
            # 随机选择两个不同的节点
            while True:
                src = random.randint(0, num_nodes-1)
                dst = random.randint(0, num_nodes-1)
                if src != dst:  # 确保不添加自环
                    break
            
            # 添加新边（无向图需要添加两个方向）
            new_edge = torch.tensor([[src, dst], [dst, src]], 
                                  device=device, dtype=edge_index.dtype)
            new_edge_index = torch.cat([new_edge_index, new_edge], dim=1)
        
        return x,new_edge_index,batch

    def aug_drop_node(self, x, edge_index,batch,drop_percent=0.2):
        """
        随机丢弃节点。
        
        参数:
        - x: [num_nodes, num_features]
        - edge_index: 边索引 [2, num_edges]
        - batch: 批处理索引 [num_nodes]
        - drop_percent: 丢弃节点的百分比
        """
        device = x.device
        node_num, feat_dim = x.shape # 节点数量和特征维度 []
        drop_num = int(node_num * drop_percent)
        
        # 随机选择要保留的节点
        keep_nodes = sorted(random.sample(range(node_num), node_num - drop_num)) # 随机抽取需要保留的节点 然后排序
        keep_nodes = torch.tensor(keep_nodes, device=device) # 将保留的节点转换为tensor
        
        # 更新节点特征
        x = x[keep_nodes] # 保留节点
        
        # 更新边索引
        mask = (edge_index[0].unsqueeze(1) == keep_nodes).any(1) & \
               (edge_index[1].unsqueeze(1) == keep_nodes).any(1)
        new_edge_index = edge_index[:, mask]

        # print("mask",mask)
        # print("new_edge_index,重映射节点索引之前",new_edge_index)
        
        # 重映射节点索引
        node_idx = torch.full((node_num,), -1, device=device) # 创建一个全为-1的tensor 用来映射节点
        node_idx[keep_nodes] = torch.arange(len(keep_nodes), device=device)
        new_edge_index = node_idx[new_edge_index]
        # print("new_edge_index,重映射节点索引之后",new_edge_index)


        # 更新 batch 张量
        batch = batch[keep_nodes]
        # print("aug_drop_node batch shape",batch.shape)
        
        return x, new_edge_index,batch

    def aug_subgraph(self, x, edge_index,batch, drop_percent=0.2):
        """
        随机生成子图。
        
        参数:
        - x: [num_nodes, num_features]
        """
        device = x.device
        node_num, feat_dim = x.shape
        keep_num = int(node_num * (1 - drop_percent))

        # 随机选择中心节点
        center_idx = random.randint(0, node_num - 1)
        sampled_nodes = {center_idx}
        frontier = {center_idx}
        
        # BFS采样子图
        edge_list = edge_index.t().cpu().numpy()
        adj_lists = defaultdict(list)
        for src, dst in edge_list:
            adj_lists[src].append(dst)
            adj_lists[dst].append(src)
            
        while len(sampled_nodes) < keep_num and frontier:
            new_frontier = set()
            for node in frontier:
                neighbors = adj_lists[node]
                for neighbor in neighbors:
                    if neighbor not in sampled_nodes and len(sampled_nodes) < keep_num:
                        sampled_nodes.add(neighbor)
                        new_frontier.add(neighbor)
            frontier = new_frontier
        #如果没有采样到节点，返回原始数据    
        if not sampled_nodes:
            return x, edge_index, batch
        # 更新特征矩阵
        sampled_nodes = sorted(list(sampled_nodes))
        sampled_nodes = torch.tensor(sampled_nodes, device=device)
        x = x[sampled_nodes]
        print("x shape",x.shape)
        
        # 更新边索引
        mask = (edge_index[0].unsqueeze(1) == sampled_nodes).any(1) & \
               (edge_index[1].unsqueeze(1) == sampled_nodes).any(1)
        new_edge_index = edge_index[:, mask]
        
        # 重映射节点索引
        node_idx = torch.full((node_num,), -1, device=device)
        node_idx[sampled_nodes] = torch.arange(len(sampled_nodes), device=device)
        new_edge_index = node_idx[new_edge_index]

        #更新batch
        new_batch = batch[sampled_nodes]

        if len(new_batch) < len(batch):
            padding = torch.full((len(batch) - len(new_batch),), batch.max(), device=device)
            new_batch = torch.cat([new_batch, padding])

        

        return x, new_edge_index,new_batch

    
    
    #总对比损失
    def multi_view_contrastive_loss(self,h,h_aug1,h_aug2,h_aug3,temperature=0.1):

        """
        计算多视图对比学习损失。

        参数:
        - views: 包含多个视图的嵌入向量列表
        - temperature: 温度参数

        返回:
        - 多视图对比学习损失值
        """
        #     batch_size, _ = raw.size()

        #     # 计算嵌入向量的范数 归一化
        #     raw = F.normalize(raw,dim=1)
        #     x1 = F.normalize(x1,dim=1)
        #     x2 = F.normalize(x2,dim=1)
        #     x3 = F.normalize(x3,dim=1)

        #     #初始化总损失
        #     total_loss = 0

        #    #各个视图之间的损失对比  先改进原始视图与增强视图之间的相似性计算


        # 正例对比损失
        loss_pos = (
            self.loss_cl(h, h_aug1,temperature) +
            self.loss_cl(h, h_aug2,temperature) +
            self.loss_cl(h, h_aug3,temperature)
        )

        # 增强视图之间的对比损失
        loss_aug = (
            self.loss_cl(h_aug1, h_aug2, temperature) +
            self.loss_cl(h_aug1, h_aug3, temperature) +
            self.loss_cl(h_aug2, h_aug3, temperature)
        )

        # 总损失
        alpha, beta = 0.8, 0.2
        total_loss = alpha * loss_pos + beta * loss_aug
        return total_loss


    # 对比学习损失 多视图的对比损失 
    # 对比学习改进
    def loss_cl(self, x1, x2,T=0.1):
        """
        计算对比学习损失。

        参数:
        - x1: 第一组嵌入向量
        - x2: 第二组嵌入向量

        返回:
        - 对比学习损失值
        """
        # 温度参数
        # T = 0.1
        
        # 获取批次大小
        batch_size, _ = x1.size()
        
        # 计算每个向量的范数 归一化
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        # x1 = F.normalize(x1,dim=1)
        # x2 = F.normalize(x2,dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        # sim_matrix = torch.mm(x1,x2.t())/T
        
        # 应用温度缩放
        sim_matrix = torch.exp(sim_matrix / T)
        
        # 提取正样本对的相似度
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        
        # 计算对比损失
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()
        
        return loss






def apply_augmentation(x,edge_index,batch,drop_feature_prob=0.5,drop_node_prob=0.5,permute_edge_prob=0.5,mask_feature_prob=0.5):

    # self.drop_node_prob = drop_node_prob
    # self.drop_feature_prob = drop_feature_prob
    # self.permute_edge_prob = permute_edge_prob
    # self.mask_feature_prob = mask_feature_prob

    """随机丢失节点特征"""
    # if self.drop_feature_prob > 0:
    if drop_feature_prob > 0:
        x = drop_feature(x=x, drop_prob=drop_feature_prob)
    #
    """随机丢弃节点"""
    # if self.drop_node_prob > 0:
    if drop_node_prob > 0:
        x, edge_index, batch = drop_nodes(x=x, edge_index=edge_index, batch=batch, drop_prob=drop_node_prob) 
    #
    """随机扰动边连接"""
    if permute_edge_prob > 0:
        x, edge_index, batch = permute_edges(x, edge_index, batch, permute_edge_prob)
    #
    """随机掩蔽节点特征"""
    if mask_feature_prob > 0:
        x, edge_index, batch = mask_features(x, edge_index,batch,mask_feature_prob)

    return x, edge_index, batch

def drop_feature(x, drop_prob):
    """随机丢失节点特征"""
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def drop_nodes(x, edge_index, batch, drop_prob=0.5):
    """随机丢弃节点"""

    """
    num_nodes = x.size(0)
    # print("num_nodes=",num_nodes)
    mask = torch.rand(num_nodes,device='cuda') > drop_prob  # 随机生成掩码
    # print("mask device",mask.device)
    # print("edge_index device",edge_index.device)
    # mask = mask.to(edge_index.device)
    # batch = batch.to(edge_index.device)
    x = x[mask]
    # print("x device",x.device)
    edge_index = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
    # print("batch device",batch.device)
    """
    # 随机选择保留的节点
    keep_mask = torch.rand(x.size(0),device="cuda") > drop_prob
    # 保留的节点索引
    keep_idx = torch.nonzero(keep_mask).squeeze()
    # 节点重映射
    mapping = torch.full((x.size(0),), -1, dtype=torch.long,device="cuda")
    mapping[keep_idx] = torch.arange(keep_idx.size(0),device="cuda")
    # 更新边索引
    row, col = edge_index
    mask = keep_mask[row] & keep_mask[col]
    edge_index = torch.stack([mapping[row[mask]], mapping[col[mask]]], dim=0)
    # 保留特征
    x = x[keep_mask]

    return x, edge_index, batch

def permute_edges(x, edge_index, batch, permute_prob=0.2):
    """随机扰动边连接"""
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges,device="cuda") > permute_prob  # 随机生成掩码
    edge_index = edge_index[:, mask]
    return x,edge_index,batch

def mask_features(x, edge_index, batch, mask_prob=0.2):
    """随机掩蔽节点特征"""
    mask = torch.rand(x.size(),device="cuda") > mask_prob
    # mask = torch.rand_like(x,dtype=torch.float32) > mask_prob
    x = x * mask.float()
    return x, edge_index, batch






num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3
class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3
class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)

class ZHC(torch.nn.Module):
    def __init__(self, MLP_dim=96, dropout=0.1,c_feature=108):
        super(ZHC, self).__init__()

        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.src_emb = nn.Embedding(26, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(26, d_model), freeze=True)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # GIN model for extracting ligand features
        nn1 = Sequential(Linear(c_feature, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(MLP_dim)
        nn2 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(MLP_dim)
        nn3 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(MLP_dim)
        nn4 = Sequential(Linear(MLP_dim, MLP_dim), ReLU(), Linear(MLP_dim, MLP_dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(MLP_dim)

        self.fc1_c = Linear(MLP_dim, 120)
        self.poc_fc = Linear(120, 60)

        self.fc1 = nn.Linear(120+120+120, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)        # n_output = 1 for regression task

        #消融 gcn 对比 图对比模型
        self.gcn = GCNNet2()

        # FNN
        self.classifier = nn.Sequential(
            
            nn.Linear(1020, 2048), # 120+120+120+120+120+120+120+120+60
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(512,32),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        
        self.complex_graph = graphcl(
                                        # c_features = 108,hidden_dim = 256,proj_dim = 300,out_dim = 200,
                                        c_features = 108,hidden_dim = 256,proj_dim = 256,out_dim = 120,
                                        drop_percent=0.1,
                                        # drop_feature_rate=0,
                                        # drop_node_prob=0.2, #随机丢弃节点
                                        # permute_edge_prob=0, #随机扰动边连接
                                        # mask_feature_prob=0, #随机掩蔽节点特征    
                                        # subgraph_prob=0.2, #子图增强    
                                        heads = 2,mode='both'
        )

        self.pt = GraphSAGE()

        self.mamba = Mamba(
            d_model=256,
            d_state=128,  #只能设置256 不能变
            d_conv=4,
            expand=2,
        )

        self.mamba2 = Mamba2(
            d_model=256,
            d_state=128,
            d_conv=4,
            expand=2,
        
        )
        

        self.mixer_model = MixerModel(
            d_intermediate=256,
            d_model=256,
            n_layer=3,
            vocab_size=26,
            rms_norm=True
        )
        self.nn = Sequential(Linear(d_model,256),nn.Dropout(0.2),
                             Linear(256,256,nn.Dropout(0.2)),
                             Linear(256,d_model))
        

        self.bn4 = nn.BatchNorm1d(200)
        self.bn5 = nn.BatchNorm1d(128)
        self.shared_encoder = SharedModule(output_dim=120)

        self.mambaDemo = ResidualGroup(dim=60, input_resolution=(100, 10), depth=6, d_state=16, mlp_ratio=4.,drop_path=0.)

    def forward(self, data):
        # print(data)
        x_ligand, edge_index_ligand , batch_ligand = data.x_t, data.edge_index_t,data.x_t_batch
        x_complex, edge_index_complex, batch_complex = data.x_s, data.edge_index_s, data.x_s_batch #[26686, 108]
        
        protein = data.protein
        

        z1 = self.complex_graph(x_complex,edge_index_complex,batch_complex,view='first') # 视图随机节点丢失
        z2 = self.complex_graph(x_complex,edge_index_complex,batch_complex,view='second') # 随机扰动边连接
        z3 = self.complex_graph(x_complex,edge_index_complex,batch_complex,view='third') # 随机掩蔽节点特征
        raw = self.complex_graph(x_complex,edge_index_complex,batch_complex,view='raw') # 子图增强

        l1 = self.complex_graph(x_ligand,edge_index_ligand,batch_ligand,view='first')
        l2 = self.complex_graph(x_ligand,edge_index_ligand,batch_ligand,view='second')
        l3 = self.complex_graph(x_ligand,edge_index_ligand,batch_ligand,view='third')
        l_raw = self.complex_graph(x_ligand,edge_index_ligand,batch_ligand,view='raw')
        
        


        protein = self.src_emb(protein) + self.pos_emb(protein)  # [64,1000,60] --> 目前的d_model 更改为了 256
        

        embedded_xt, _ = self.transformer_encoder(protein)
        protein = embedded_xt[:, 0, :]
        
        """
        
        protein = self.mixer_model(protein)
        protein = self.mamba(protein)
        protein = self.dropout(protein)
        protein = self.relu(protein)
        protein = self.mamba(protein)
        """
        
        x = torch.cat([raw,z1,z2,z3,l_raw,l1,l2,l3,protein],dim=1)

        x = self.classifier(x)
        # print(x,x.shape)

        # gcl_loss = self.complex_graph.loss_cl(z1,z2)
        loss = self.complex_graph.multi_view_contrastive_loss(raw,z1,z2,z3)
        ligand_loss = self.complex_graph.multi_view_contrastive_loss(l_raw,l1,l2,l3)

        return x,loss,ligand_loss
        # return x
    
    # 将有向图的边索引转换为无向图的边索引
    def to_undirected(edge_index):
        # 反转边的方向
        reversed_edges = edge_index[[1, 0], :]
        # print("reversed_edges",reversed_edges)
        
        # 将原始边和反转边合并
        undirected_edge_index = torch.cat([edge_index, reversed_edges], dim=1)
        # print("undirected_edge_index",undirected_edge_index)
        # 去除重复的边，并排序
        undirected_edge_index = torch.unique(undirected_edge_index, dim=1, sorted=True)
        # print("undirected_edge_index",undirected_edge_index)
        # 对每一对边进行排序，确保 (u, v) 和 (v, u) 只保留一个
        sorted_edges = torch.sort(undirected_edge_index, dim=0)[0]
        # print("sorted_edges",sorted_edges)
        # 去除重复的边
        unique_edges = torch.unique(sorted_edges, dim=1)
        
        return unique_edges


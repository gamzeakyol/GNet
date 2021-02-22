
from comet_ml import Experiment

import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch_scatter import scatter_add

import torch_geometric as tg
import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GraphConv, TopKPooling, Set2Set, GCNConv, global_mean_pool #GNNExplainer
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import remove_isolated_nodes, to_dense_adj, negative_sampling, remove_self_loops, add_self_loops, dropout
#from torch_geometric.datasets import TUDataset
#from torch_geometric.datasets import Planetoid
from torch_geometric.data import Batch, DataLoader
import torch_geometric.transforms as T
#from torch_geometric.nn.conv.gcn_conv import gcn_norm

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from pathlib import Path
from pathlib import PurePath
from xml.dom import minidom
from numpy import argmax

from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

import time
from datetime import datetime


import Utils.utils as util
from Utils import SECParser
import Utils.config as conf

from Utils import MANIAC as m

from train_test_split_edges import train_split_edges, train_test_split_edges


# Load MANIAC config
cfg = conf.get_maniac_cfg()

# Folder with MANIAC keyframes
_FOLDER = os.getcwd() + "/MANIAC/"


#Logs on Comet ML
LOG_COMMET = False

'''
if LOG_COMMET:
    experiment = Experiment(api_key='H4xiOoxfcTYN8DvanvkI7QJeR', project_name="graph-gmz2", workspace="ardai", auto_metric_logging=True, 
                            log_code=True)
    
    experiment.set_name("graphnet_VGAE_3layer".format(datetime.now().strftime("%d/%m/%Y - %H:%M:%S")))
    experiment.add_tag("graphnet_VGAE_3layer")
    experiment.set_code()
else:
    experiment = Experiment(api_key='')
'''

SUMMARY_WRITER = True

if SUMMARY_WRITER:
    writer = SummaryWriter("./runs1/maniac_graphconv_3layer_kernel_128_action_node_edge_trial3")


'''

    This is used to create new datasets.
    
    Example:
    train_set = MANIAC_DS( "FOLDER_TO_MANIAC_GRAPHML")
    
    All settings is set inside config.py
    except save new data and create new dataset

'''

# Settings for creating MANIAC dataset
_SAVE_RAW_DATA      = False
_CREATE_DATASET     = False

EPS = 1e-15

if _CREATE_DATASET:
    train_set = m.MANIAC_DS(_FOLDER + "training/")
    val_set = m.MANIAC_DS(_FOLDER + "validation/")
    test_set = m.MANIAC_DS(_FOLDER + "test/")


# Save datasets into _FOLDER + "raw/maniac_training_xw.pt"
# This is needed for PyTorch Geometric and DataLoaders
if _SAVE_RAW_DATA:
    with open(os.path.join(_FOLDER + "raw/maniac_training_" + str(cfg.time_window) + "w.pt"), 'wb') as f:
                torch.save(train_set, f)

    with open(os.path.join(_FOLDER + "raw/maniac_validation_" + str(cfg.time_window) + "w.pt"), 'wb') as df:
                torch.save(val_set, df)

    with open(os.path.join(_FOLDER + "raw/maniac_test_" + str(cfg.time_window) + "w.pt"), 'wb') as df:
                torch.save(test_set, df)


'''

    This is used to create load MANIAC dataset into DataLoader.
    
    Example)
    To load processed or create preprocessed data into list.
    train_dataset = m.ManiacIMDS(_FOLDER, "train")
    
    Creates a DataLoader of the loaded dataset.
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

'''

# Loading pre-processed or creates new processed pt files into FOLDER/processed/

train_dataset = m.ManiacIMDS(_FOLDER, "train") #normally, these lines should run
print("dataset is constructed")
test_ds = m.ManiacIMDS(_FOLDER, "test")
valid_ds = m.ManiacIMDS(_FOLDER, "valid")

#dataset = torch.utils.data.ConcatDataset([train_dataset, valid_ds]) #concatenating datasets to increase amount of data


#####################PRINT################################

print("Total graphs:\t {}\n=========".format(len(train_dataset)+len(test_ds)+len(valid_ds)))
print("Training: \t {}".format(len(train_dataset)))
print("Test: \t\t {}".format(len(test_ds)))
print("Validation: \t {}\n=========".format(len(valid_ds)))
#####################PRINT################################

# Create data loaders from dataset.
# https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.DataLoader
#from torch_geometric.data import DataLoader


# Get maximum node for graph reconstruction
#max_num_nodes_train = max([len(i.x) for i in train_dataset])
#max_num_nodes_valid = max([len(i.x) for i in valid_ds])
#max_num_nodes_test = max([len(i.x) for i in test_ds])
#max_num_nodes = max(max_num_nodes_test, max_num_nodes_train, max_num_nodes_valid)

#####################PRINT################################
#print("Max number of nodes found:", max_num_nodes)
#####################PRINT################################


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True) 
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, drop_last=True) 
valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=True, drop_last=True) 


#####################PRINT################################
print("Total batchs:\t {}\n=========".format(len(train_loader)+len(test_loader)+len(valid_loader)))
print("Training: \t {}".format(len(train_loader)))
print("Test: \t\t {}".format(len(test_loader)))
print("Validation: \t {}\n=========".format(len(valid_loader)))
#####################PRINT################################


#writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

#dataset = Planetoid("/tmp/citeseer", "Citeseer", T.NormalizeFeatures())
#data = dataset[0]


node_class_list  = ['Apple', 'Arm', 'Ball', 'Banana', 'Body', 'Bowl', 'Box', 'Bread',
                   'Carrot', 'Chopper', 'Cucumber', 'Cup', 'Hand',
                   'Knife', 'Liquid', 'Pepper', 'Plate', 'Sausage', 'Slice', 'Spoon', 'null']   #cfg.objects, cfg._objects


'''
    Model definition
    Link prediction (Encoder + Decoder)
'''
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.do = nn.Dropout(p=0.3)
        
        """
        self.conv1 = pyg_nn.GCNConv(len(cfg.objects), 16 * out_channels, cached=False)
        self.conv2 = pyg_nn.GCNConv(16 * out_channels, 8 * out_channels, cached=False)
        #self.conv3 = pyg_nn.GCNConv(32 * out_channels, 16 * out_channels, cached=False)
        #self.conv4 = pyg_nn.GCNConv(16 * out_channels, 8 * out_channels, cached=False)
        #self.conv5 = pyg_nn.GCNConv(8 * out_channels, 4 * out_channels, cached=False)

        self.conv_mu = pyg_nn.GCNConv(8 * out_channels, out_channels, cached=False)
        self.conv_logstd = pyg_nn.GCNConv(8 * out_channels, out_channels, cached=False)
        """
        

        """
        Kernel isimlendirme -> 8 * out_channels -> out_channels: kernel128
                            -> 4 * out_channels -> out_channels: kernel64
        """
        
        self.conv1 = pyg_nn.GraphConv(len(cfg.objects), 16 * out_channels, aggr="mean")
        self.conv2 = pyg_nn.GraphConv(16 * out_channels, 8 * out_channels, aggr="mean")
        #self.conv3 = pyg_nn.GraphConv(16 * out_channels, 8 * out_channels, aggr="mean")
        #self.conv4 = pyg_nn.GraphConv(128 * out_channels, 64 * out_channels, aggr="mean")
        #self.conv5 = pyg_nn.GraphConv(64 * out_channels, 32 * out_channels, aggr="mean")
        #self.conv6 = pyg_nn.GraphConv(32 * out_channels, 16 * out_channels, aggr="mean")
        #self.conv7 = pyg_nn.GraphConv(16 * out_channels, 8 * out_channels, aggr="mean")
        #self.conv8 = pyg_nn.GraphConv(8 * out_channels, 4 * out_channels, aggr="mean")


        self.conv_mu = pyg_nn.GraphConv(8 * out_channels, out_channels, aggr="mean")
        self.conv_logstd = pyg_nn.GraphConv(8 * out_channels, out_channels, aggr="mean")
        
       

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()
        #edge_index, data.edge_weight = GCNConv.norm(data.edge_index, data.num_nodes, data.edge_weight)

        x = self.conv2(x, edge_index).relu()
        #data.edge_index, data.edge_weight = GCNConv.norm(data.edge_index, data.num_nodes, data.edge_weight)

        #x = self.conv3(x, edge_index).relu()
        #data.edge_index, data.edge_weight = GCNConv.norm(data.edge_index, data.num_nodes, data.edge_weight)
        
        '''
        x = self.conv4(x, edge_index).relu()
        #data.edge_index, data.edge_weight = GCNConv.norm(data.edge_index, data.num_nodes, data.edge_weight)

        x = self.conv5(x, edge_index).relu()

        x = self.conv6(x, edge_index).relu()

        x = self.conv7(x, edge_index).relu()

        x = self.conv8(x, edge_index).relu()

        '''

        #x = self.do(x)

        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class  Decoder(torch.nn.Module):
    def __init__(self, out_channels, sigmoid=True):
        super(Decoder, self).__init__()

        #self.do = nn.Dropout(p=0.5)
        """
        self.linear = nn.Linear(1, channels)

        self.conv1 = pyg_nn.GCNConv(channels, 4 * out_channels, cached=False) #channels
        self.conv2 = pyg_nn.GCNConv(4 * out_channels, 8 * out_channels, cached=False)
        self.conv3 = pyg_nn.GCNConv(8 * out_channels, 16 * out_channels, cached=False)
        self.conv4 = pyg_nn.GCNConv(16 * out_channels, 32 * out_channels, cached=False)
        self.conv5 = pyg_nn.GCNConv(32 * out_channels, 64 * out_channels, cached=False)

        self.conv6 = pyg_nn.GCNConv(64 * out_channels, out_channels, cached=False)
        """
        """
        self.conv1 = pyg_nn.GraphConv(channels, 4 * out_channels, aggr="mean")
        self.conv2 = pyg_nn.GraphConv(4 * out_channels, 8 * out_channels, aggr="mean")
        self.conv3 = pyg_nn.GraphConv(8 * out_channels, 16 * out_channels, aggr="mean")
        self.conv4 = pyg_nn.GraphConv(16 * out_channels, 32 * out_channels, aggr="mean")
        self.conv5 = pyg_nn.GraphConv(32 * out_channels, 64 * out_channels, aggr="mean")

        self.conv6 = pyg_nn.GraphConv(64 * out_channels, out_channels, aggr="mean")
        """



       
    def forward(self, value, edge_index, sigmoid=True):

        #value = (value[edge_index[0]] * value[edge_index[1]]).sum(dim=1)

        #value = self.linear(value)
        '''
        value = self.conv1(value, edge_index).relu()
        value = self.conv2(value, edge_index).relu()
        value = self.conv3(value, edge_index).relu()
        value = self.conv4(value, edge_index).relu()
        value = self.conv5(value, edge_index).relu()

        value = self.conv6(value, edge_index).relu()
        '''
        #print("value1", value.shape, value[edge_index[0]].shape, value[edge_index[1]].shape)
        #value = value.sum(dim=1)
        value = (value[edge_index[0]] * value[edge_index[1]]).sum(dim=1)
        print("value", value, value.shape)

        #print("value", value, value.shape, value[edge_index[0]].shape, value[edge_index[1]].shape)
        #print("edge_index", edge_index, edge_index.shape)

        return torch.sigmoid(value) if sigmoid else value

        #adj = torch.matmul(value, value.t())
        #return torch.sigmoid(adj) if sigmoid else adj





'''
    Node classification
'''
class Predictor(torch.nn.Module):
    def __init__(self, channels): #dataset
        super(Predictor, self).__init__()
        #self.conv11 = GCNConv(21, 16) #dataset.num_features, args.hidden
        #self.conv21 = GCNConv(16, len(cfg.objects)) #args.hidden, dataset.num_classes

        #self.conv11 = nn.Conv1d(21, 16, 1, stride=2)
        #self.conv21 = nn.Conv1d(16, len(cfg.objects), 1, stride=2)
        self.lin1 = nn.Linear(channels, channels*5)
        self.lin2 = nn.Linear(channels*5, channels)
    
    '''
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    '''

    def forward(self, data):
        #x, edge_index = data.x, data.edge_index
        #x = F.relu(self.conv11(data))
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = F.relu(self.conv21(x))

        data = self.lin1(data)
        data = self.lin2(data)

        return F.log_softmax(data, dim=1)
        #return data


'''
    Action classification
'''
class ActionPredictor(torch.nn.Module):
    def __init__(self, channels, node_num): #dataset
        super(ActionPredictor, self).__init__()

        #self.lstm = torch.nn.LSTM(node_num, channels, 4, dropout=0)
        #self.conv = nn.Conv1d(channels, node_num, 1)
        #self.conv2 = nn.Conv1d(29, 29, 1)

        #self.lstm = nn.LSTM(channels, channels*5, 1, dropout=0, batch_first=True)
        #self.conv1 = nn.Conv1d(channels, channels*5, 1) 
        self.lin1 = nn.Linear(channels, channels*5)
        self.lin2 = nn.Linear(channels*5, len(cfg.action_map))

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, data, batch):

        #data = data.unsqueeze(dim=2)
        #print("data", data, data.shape)
        
        data = global_mean_pool(data, batch)

        #data = self.lstm(data)
        data = self.lin1(data).relu()
        #data = self.conv1(data)
        data = self.lin2(data)

        return self.logsoftmax(data)
        #return data

'''
    Full network
'''
class graphNet(torch.nn.Module):
    def __init__(self):
        super(graphNet, self).__init__()
        #self.decoder = Decoder(channels, len(cfg.objects))
        #channels = 21->42
        self.model = pyg_nn.VGAE(Encoder(len(cfg.objects), 21), Decoder(21, len(cfg.objects))).to(dev) #decoder = Decoder(len(cfg.objects))).to(dev)
        #self.model = pyg_nn.ARGVA(Encoder(21, channels)).to(dev)

        self.predictor = Predictor(21).to(dev)

        
    def forward(self, x, train_flag):
                
        batch = x.batch
        x = x.to_data_list()
        node_len = x[0].x.shape[0]

        self.actionpredictor = ActionPredictor(21, node_len).to(dev)

           
        if train_flag == True:
            #x[0].edge_index, x[0].edge_attr = dropout.dropout_adj(x[0].edge_index, p=0.3, force_undirected=True, training=True)

            x = train_split_edges(x[0]) #x = self.model.split_edges(x[0], val_ratio=0.0, test_ratio=0.0) #, test_ratio=0.0, train_ratio=1.0)       #self.model.split_edges
            #x = train_test_split_edges(x[0], val_ratio=0.0, test_ratio=0.0)
            #y = to_dense_adj(x.train_pos_edge_index)
            #y_neg = to_dense_adj(x.train_neg_edge_index)

        else:
            #x[0].edge_index, x[0].edge_attr = dropout.dropout_adj(x[0].edge_index, p=0.3, force_undirected=True, training=False)

            x = train_split_edges(x[0]) #x = self.model.split_edges(x[0], test_ratio=1.0) # modified function
                
        x = Batch.from_data_list([x])

        
        if train_flag == True: 
            z = self.model.encode(x.x, x.train_pos_edge_index)         # Encoder input

            p_z = self.predictor(z)                                    # Prediction input

            a_z = self.actionpredictor(z, batch)

            #q_z = self.model.decoder(z, x.train_pos_edge_index)       # Decoder input: Modelin test fonksiyonunda decoder zaten kullanılıyor.

            return p_z, a_z, z, x.train_pos_edge_index, x.train_neg_edge_index #x.train_neg_adj_mask #, x.test_pos_edge_index, x.test_neg_edge_index
            #return p_z, a_z, z, x.train_pos_edge_index, x.train_neg_adj_mask, x.test_pos_edge_index, x.test_neg_edge_index

        else:
            z = self.model.encode(x.x, x.train_pos_edge_index)           #x.train_pos_edge_index # Encoder input

            p_z = self.predictor(z)                                    # Prediction input

            a_z = self.actionpredictor(z, batch)

            #q_z = self.model.decoder(z, x.test_pos_edge_index)         # Decoder input: Modelin test fonksiyonunda decoder zaten kullanılıyor.

            return p_z, a_z, z, x.train_pos_edge_index, x.train_neg_edge_index      #z, x.train_pos_edge_index, x.train_neg_edge_index
            #return p_z, a_z, z, x.test_pos_edge_index, x.test_neg_edge_index


    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """

        #print("pos_edge_index", pos_edge_index, pos_edge_index.shape)

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)


        pos_pred = self.model.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.model.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        """
        if y.shape[0] == 0 or pred.shape[0] == 0:
            #print("y, pred", y.shape , pred.shape)
            return 1.0
        """

        return average_precision_score(y, pred) #multi_class = 'ovr' or 'ovo' roc_auc_score(y, pred, multi_class='ovo')


    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.model.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        

        return pos_loss + neg_loss

   


channels = 21
#dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

print('CUDA availability:', torch.cuda.is_available())


total_model = graphNet().to(dev)

print("**********************")
print(total_model)
print("**********************")

LR= 0.000001
DATA_AUG = None  #"GCN norm" or "Remove isolated nodes" or None 
optimizer = torch.optim.Adam(total_model.parameters(), lr=LR) #default lr=0.001
nllloss = nn.NLLLoss()
crossentropy = torch.nn.CrossEntropyLoss()
LAYER_NUM = "3"
MODEL_NAME = "VGAE" #GAE, VGAE, or ARGVA

#if SUMMARY_WRITER:
#   writer.add_hparams({'lr': LR, 'dataset': "MANIAC", 'layer_num': 3, 'kernel size': 64, 'dropout_encoder': False, 'dropout_decoder': False, 'conv_type': 'Graph Conv', 'action_pred_layer': 1}, {})
isolated_nodes = T.RemoveIsolatedNodes()
#gcnnorm = T.GCNNorm()

'''
if LOG_COMMET:
    experiment.log_parameter('lr', LR)
    experiment.log_parameter('optimizer', optimizer)
    experiment.log_parameter('loss', nllloss)
    experiment.log_parameter('data_augmentation', DATA_AUG)
    experiment.log_parameter('num of encoder layers', LAYER_NUM)
    experiment.log_parameter('model type', MODEL_NAME)
'''

# GNN explainer
#explainer = GNNExplainer(total_model, epochs=200)
#node_idx = 21

for epoch in range(0, 200):

    action_score_trains = []
    action_score_vals = []
    action_score_tests = []

    tr_correct = 0
    correct_action_test = 0
    correct_action_val = 0

    tr_counter = 0.0
    val_counter = 0.0
    test_counter = 0.0

    #Training loop
    total_model.train()
    for graph in train_loader: #for batch_idx, (graph) in enumerate(train_loader):

            batch_train_idx = 0

            #graph.train_mask = graph.val_mask = graph.test_mask = graph.y = None

            print("graph", graph, graph.x, graph.edge_index, graph.x.argmax(dim=1))

            #graph = isolated_nodes(graph)
            #graph = gcnnorm(graph)


            target_list = graph.x

            train_flag = True

            optimizer.zero_grad()

            p_z, a_z, q_z, train_pos_edge_index, train_neg_edge_index = total_model(graph, train_flag)
            
            #print("graph", graph, train_pos_edge_index, train_pos_edge_index.shape, train_neg_edge_index, train_neg_edge_index.shape)

            """
            train_neg_edge_list = []
            for i in range(len(train_neg_adj_mask[0])):
                for j in range(len(train_neg_adj_mask[1])):
                        
                        if train_neg_adj_mask[i][j] == True:
                            train_neg_edge_list.append([i,j])
            
            

            train_neg_edge_index = torch.tensor(train_neg_edge_list, dtype=torch.long)

            train_neg_edge_index = torch.transpose(train_neg_edge_index, 0, 1)
            """

	        # Tüm elemanların hepsinin CPU veya GPU'da olmasına dikkat et.
            #Data(edge_attr=[32, 3], test_neg_edge_index=[2, 1], test_pos_edge_index=[2, 1], train_neg_adj_mask=[15, 15], train_pos_edge_index=[2, 20], val_neg_edge_index=[2, 0], val_pos_edge_index=[2, 0], x=[15, 21], y=[8])
            
            
            #Compute auc and ap values for the link prediction
            link_train_ap = total_model.test(q_z, train_pos_edge_index, train_neg_edge_index)
            
            
            #Compute acc score for the node classification
            pred = p_z.argmax(dim=1)

            gt = target_list.argmax(dim=1)
            
            train_correct = torch.eq(pred, gt)

            counts_correct = (train_correct == True).sum(dim=0)

            node_score_train = int(torch.sum(counts_correct)) / train_correct.shape[0]            

            
            #Compute action classification
            print("Graph", graph, graph.edge_index, graph.x, np.argmax(graph.x, axis=1))

            #a_z = torch.sum(a_z, dim=0)

            #a_z = a_z / len(graph.batch)

            #pred_action = a_z.argmax(dim=0) #actions are listed for every node. so a_z[0] is sufficient for classification of a graph

            #gt_action = graph.y.argmax(dim=0)

            pred = int(a_z.argmax(dim=1))  # Use the class with highest probability.

            tr_correct += int((pred == int(graph.y.argmax(dim=0))))  # Check against ground-truth labels.

            print("pred graph y", a_z, pred, graph.y, int(graph.y.argmax(dim=0)))

            tr_counter += 1
            

            #print("correct", pred, int(graph.y.argmax(dim=0)), correct)


            #correct_actions = torch.eq(pred_action, gt_action)

            #counts_correct_actions = (correct_actions == True).sum(dim=0)


            #print("count actions", counts_correct_actions, correct_actions)

            #action_score_train = int(torch.sum(counts_correct_actions)) / 1 #correct_actions.shape

            #print("action score train", action_score_trains)
            
            #action_score_trains.append(action_score_train)

            #print("action score train", action_score_trains)
            


            """
            # Total score
            avg_score_train = (node_score_train + link_train_ap) / 2

            print("Avg score train", avg_score_train)
            """


            #Compute node_loss, link_loss +net_loss
            
            linkloss = total_model.model.recon_loss(q_z, train_pos_edge_index)
            #actionloss = nllloss(a_z.unsqueeze(0), gt_action.unsqueeze(0))  #actionloss = nllloss(az, graph.y)
            

            target_list = torch.argmax(target_list, dim=1)
            nodeloss = nllloss(p_z, target_list)


            action_class = torch.tensor(graph.y.argmax(dim=0)).unsqueeze(0)

            print("action_class", a_z, action_class,  len(train_loader.dataset))

            actionloss = nllloss(a_z, action_class)

            kl_loss = total_model.model.kl_loss()

            net_loss = actionloss + linkloss + (1 * nodeloss) #+ kl_loss #+ linkloss + (1 * nodeloss)

            net_loss.backward()
            optimizer.step()

            #step=epoch*len(train_loader)+batch_train_idx

            '''
            #if batch_idx % step == 0: 
            if LOG_COMMET:
                experiment.log_metric('Net_loss_train', net_loss.item(), epoch)
                experiment.log_metric('Node_loss_train', nodeloss.item(), epoch)
                experiment.log_metric('Link_loss_train', linkloss.item(), epoch)
                experiment.log_metric('Link_AP_train', link_train_ap, epoch)
                experiment.log_metric('Node_acc_train', node_score_train, epoch)
                experiment.log_metric('Avg_score_train', avg_score_train, epoch)
                #experiment.log_metric('AUC_train', train_auc, epoch)
            '''
            #batch_train_idx += 1

            if SUMMARY_WRITER:
                writer.add_scalar("Net loss/train", net_loss.item(), epoch)
                writer.add_scalar("Node loss/train", nodeloss.item(), epoch)
                writer.add_scalar("Link loss/train", linkloss.item(), epoch)
                writer.add_scalar("Action loss/train", actionloss.item(), epoch)
                writer.add_scalar("Link AP/train", link_train_ap, epoch)
                writer.add_scalar("Node score/train", node_score_train, epoch)
                #writer.add_scalar("AVG score/train", avg_score_train, epoch)

            print("Epoch: {:3d}, train_loss: {:.4f}\n".format(epoch, net_loss))

            #Visualization of the results
            #node_feat_mask, edge_mask = explainer.explain_node(node_idx, graph.x, graph.edge_index)
            #ax, G = explainer.visualize_subgraph(node_idx, graph.edge_index, edge_mask, y=graph.y)
            #plt.savefig("graph.jpeg")

    
    #pred = out.argmax(dim=1)  # Use the class with highest probability.
    #correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    
    avg_action_train = tr_correct / tr_counter

    #avg_action_train = sum(action_score_trains) / len(action_score_trains)

    print("Avg action score", avg_action_train)

    if SUMMARY_WRITER:
        writer.add_scalar("Avg action score/train", avg_action_train, epoch)

    #action_score_trains.clear()
    
    

    #Validation loop
    total_model.eval()
    for graph in valid_loader: #for batch_idx, (graph) in enumerate(valid_loader):

        with torch.no_grad():

            train = False

            #graph = isolated_nodes(graph)
            #graph = gcnnorm(graph)

            target_list = graph.x

            p_z, a_z, q_z, val_pos_edge_index, val_neg_edge_index = total_model(graph, train)

            
            val_link_ap = total_model.test(q_z, val_pos_edge_index, val_neg_edge_index)

            #Compute acc score for the node classification
            pred = p_z.argmax(dim=1)

            gt = target_list.argmax(dim=1)
            
            val_correct = torch.eq(pred, gt)

            counts_val_correct = (val_correct == True).sum(dim=0)

            val_node_score = int(torch.sum(counts_val_correct)) / val_correct.shape[0]

            
            #Compute action classification

            val_pred = int(a_z.argmax(dim=1))  # Use the class with highest probability.

            correct_action_val += int((val_pred == int(graph.y.argmax(dim=0))))  # Check against ground-truth labels.

            val_counter += 1

            """
            a_z = torch.sum(a_z, dim=0)

            a_z = a_z / len(graph.batch)

            pred_val_action = a_z.argmax(dim=0)

            gt_val_action = graph.y.argmax(dim=0)

            correct_val_actions = torch.eq(pred_val_action, gt_val_action)

            counts_correct_val_actions = (correct_val_actions == True).sum(dim=0)

            action_score_val = int(torch.sum(counts_correct_val_actions)) / 1 #correct_actions.shape

            action_score_vals.append(action_score_val)
            

            pred_val = int(a_z.argmax(dim=1))  # Use the class with highest probability.

            #print("correct", pred, int(graph.y.argmax(dim=0)), tr_correct)

            correct_action_val += int(pred_val == int(graph.y.argmax(dim=0))) 
            """

            """
            # Total score
            avg_score_val = (val_node_score + val_link_ap) / 2

            print("Avg score val", avg_score_val)
            """

            
            #Compute node_loss, link_loss +net_loss
            val_linkloss = total_model.model.recon_loss(q_z, val_pos_edge_index)
            #action_val_loss = nllloss(a_z.unsqueeze(0), gt_val_action.unsqueeze(0))
            
            target_list = torch.argmax(target_list, dim=1)
            val_nodeloss = nllloss(p_z, target_list)


            action_class_val = torch.tensor(graph.y.argmax(dim=0)).unsqueeze(0)

            action_val_loss = nllloss(a_z, action_class_val)

            kl_loss = total_model.model.kl_loss()

            val_netloss = action_val_loss + (1 * val_nodeloss) + val_linkloss #+ kl_loss 

            if SUMMARY_WRITER:
                writer.add_scalar("Net loss/val", val_netloss.item(), epoch)
                writer.add_scalar("Node loss/val", val_nodeloss.item(), epoch)
                writer.add_scalar("Link loss/val", val_linkloss.item(), epoch)
                writer.add_scalar("Action loss/val", action_val_loss.item(), epoch)
                writer.add_scalar("Link AP/val", val_link_ap, epoch)
                writer.add_scalar("Node score/val", val_node_score, epoch)
                #writer.add_scalar("AVG score/val", avg_score_val, epoch)

            '''
            if LOG_COMMET:
                experiment.log_metric('Node_loss_val', val_nodeloss.item(), epoch)
                experiment.log_metric('Link_loss_val', val_linkloss.item(), epoch)
                experiment.log_metric('Net_loss_val', val_netloss.item(), epoch)
                experiment.log_metric('Node_acc_val', val_node_score, epoch)
                experiment.log_metric('Link_AP_val', val_link_ap, epoch)
                experiment.log_metric('Avg_score_val', avg_score_val, epoch)
                #experiment.log_metric('AUC_val', auc, epoch)
            '''

            #print('Epoch: {:3d}, Val NET loss: {:.4f}, Val node loss: {:.4f}, Val link loss: {:.4f}, AP: {:.4f}'.format(epoch, val_nodeloss, val_linkloss, val_netloss, val_link_ap))
            print('Epoch: {:3d}, Val NET loss: {:.4f}'.format(epoch, val_netloss))

    
    avg_action_val = correct_action_val / val_counter
 
    #avg_action_val = sum(action_score_vals) / len(action_score_vals)

    print("Avg action score val", avg_action_val)

    if SUMMARY_WRITER:
        writer.add_scalar("Avg action score/val", avg_action_val, epoch)

    #action_score_vals.clear()
    
    


    # Test loop
    total_model.eval()
    for graph in test_loader: #for batch_idx, (graph) in enumerate(test_loader):

        with torch.no_grad():

            #graph.train_mask = graph.val_mask = graph.test_mask = graph.y = None

            #graph = isolated_nodes(graph)
            #graph = gcnnorm(graph)

            target_list = graph.x

            train_fl = False

            p_z, a_z, q_z, test_pos_edge_index, test_neg_edge_index = total_model(graph, train_fl)


	        # Tüm elemanların hepsinin CPU veya GPU'da olmasına dikkat et.
            #Data(edge_attr=[32, 3], test_neg_edge_index=[2, 1], test_pos_edge_index=[2, 1], train_neg_adj_mask=[15, 15], train_pos_edge_index=[2, 20], val_neg_edge_index=[2, 0], val_pos_edge_index=[2, 0], x=[15, 21], y=[8])

            
            #Compute test link auc and ap values
            link_test_ap = total_model.test(q_z, test_pos_edge_index, test_neg_edge_index)


            
            #Compute acc score for the node classification
            pred = p_z.argmax(dim=1)

            gt = target_list.argmax(dim=1)
            
            test_correct = torch.eq(pred, gt)

            counts_test_correct = (test_correct == True).sum(dim=0)

            node_score_test = int(torch.sum(counts_test_correct)) / test_correct.shape[0]
            

            
            #Compute action classification
            """
            a_z = torch.sum(a_z, dim=0)

            a_z = a_z / len(graph.batch)

            pred_test_action = a_z.argmax(dim=0)

            gt_test_action = graph.y.argmax(dim=0)

            correct_test_actions = torch.eq(pred_test_action, gt_test_action)

            counts_correct_test_actions = (correct_test_actions == True).sum(dim=0)

            action_score_test = int(torch.sum(counts_correct_test_actions)) / 1 #correct_actions.shape

            action_score_tests.append(action_score_test)
            """
            
            

            
            pred_test = int(a_z.argmax(dim=1))  # Use the class with highest probability.

            #print("correct", pred, int(graph.y.argmax(dim=0)), tr_correct)

            correct_action_test += int((pred_test == int(graph.y.argmax(dim=0)))) 

            test_counter += 1
            
            """
            # Total score
            avg_score_test = (node_score_test + link_test_ap) / 2

            print("Avg score test", avg_score_test)
            """

            #Compute node_loss, link_loss +net_loss
            
            linklosstest = total_model.model.recon_loss(q_z, test_pos_edge_index)
            #action_test_loss = nllloss(a_z.unsqueeze(0), gt_val_action.unsqueeze(0))
            

            target_list = torch.argmax(target_list, dim=1)
            nodelosstest = nllloss(p_z, target_list)

            
            action_class_test = torch.tensor(graph.y.argmax(dim=0)).unsqueeze(0)
            action_test_loss = nllloss(a_z, action_class_test)
            

            kl_loss = total_model.model.kl_loss()

            net_losstest = action_test_loss + (1 * nodelosstest) + linklosstest # + kl_loss

            if SUMMARY_WRITER:
                writer.add_scalar("Net loss/test", net_losstest.item(), epoch)
                writer.add_scalar("Node loss/test", nodelosstest.item(), epoch)
                writer.add_scalar("Link loss/test", linklosstest.item(), epoch)
                writer.add_scalar("Action loss/test", action_test_loss.item(), epoch)
                writer.add_scalar("Link AP/test", link_test_ap, epoch)
                writer.add_scalar("Node score/test", node_score_test, epoch)
                #writer.add_scalar("AVG score/test", avg_score_test, epoch)

            '''
            #if batch_idx % step == 0: 
            if LOG_COMMET:
                experiment.log_metric('Net_loss_test', net_losstest.item(), epoch)
                experiment.log_metric('Node_loss_test', nodelosstest.item(), epoch)
                experiment.log_metric('Link_loss_test', linklosstest.item(), epoch)
                experiment.log_metric('Link_AP_test', link_test_ap, epoch)
                experiment.log_metric('Node_acc_test', node_score_test, epoch)
                experiment.log_metric('Avg_score_test', avg_score_test, epoch)
                #experiment.log_metric('AUC_train', train_auc, epoch)
            '''
            #batch_idx += 1

            print("Epoch: {:3d}, test_loss: {:.4f}\n".format(epoch, net_losstest))


    #avg_action_test = sum(action_score_tests) / len(action_score_tests)

    avg_action_test = correct_action_test / test_counter

    print("Avg action score test", avg_action_test)

    if SUMMARY_WRITER:
        writer.add_scalar("Avg action score/test", avg_action_test, epoch)

    #action_score_tests.clear()

    

        
'''
if LOG_COMMET:
    experiment.end()  
'''

writer.flush()
writer.close()

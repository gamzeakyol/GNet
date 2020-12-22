
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
from torch_geometric.nn import GraphConv, TopKPooling, Set2Set, GCNConv, GNNExplainer
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import remove_isolated_nodes
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

from train_test_split_edges import train_test_split_edges


# Load MANIAC config
cfg = conf.get_maniac_cfg()

# Folder with MANIAC keyframes
_FOLDER = os.getcwd() + "/MANIAC/"


#Logs on Comet ML
LOG_COMMET = False
    
if LOG_COMMET:
    experiment = Experiment(api_key='H4xiOoxfcTYN8DvanvkI7QJeR', project_name="graph-gmz2", workspace="ardai", auto_metric_logging=True, 
                            log_code=True)
    
    experiment.set_name("graphnet_VGAE_4layer_sym".format(datetime.now().strftime("%d/%m/%Y - %H:%M:%S")))
    experiment.add_tag("graphnet_VGAE_4layer_sym")
    experiment.set_code()
else:
    experiment = Experiment(api_key='')


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
train_dataset = m.ManiacIMDS(_FOLDER, "train")
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
max_num_nodes_train = max([len(i.x) for i in train_dataset])
max_num_nodes_valid = max([len(i.x) for i in valid_ds])
max_num_nodes_test = max([len(i.x) for i in test_ds])
max_num_nodes = max(max_num_nodes_test, max_num_nodes_train, max_num_nodes_valid)

#####################PRINT################################
print("Max number of nodes found:", max_num_nodes)
#####################PRINT################################


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True) #train_dataset
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=True, drop_last=True)


#####################PRINT################################
print("Total batchs:\t {}\n=========".format(len(train_loader)+len(test_loader)+len(valid_loader)))
print("Training: \t {}".format(len(train_loader)))
print("Test: \t\t {}".format(len(test_loader)))
print("Validation: \t {}\n=========".format(len(valid_loader)))
#####################PRINT################################


writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

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

        #self.do = nn.Dropout(p=0.5)

        self.conv1 = pyg_nn.GCNConv(len(cfg.objects), 16 * out_channels, cached=False)
        self.conv2 = pyg_nn.GCNConv(16 * out_channels, 8 * out_channels, cached=False)
        self.conv3 = pyg_nn.GCNConv(8 * out_channels, 4 * out_channels, cached=False)
        self.conv4 = pyg_nn.GCNConv(4 * out_channels, 2 * out_channels, cached=False)


        self.conv_mu = pyg_nn.GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)


    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()
        #edge_index, data.edge_weight = GCNConv.norm(data.edge_index, data.num_nodes, data.edge_weight)

        x = self.conv2(x, edge_index).relu()
        #data.edge_index, data.edge_weight = GCNConv.norm(data.edge_index, data.num_nodes, data.edge_weight)

        x = self.conv3(x, edge_index).relu()
        #data.edge_index, data.edge_weight = GCNConv.norm(data.edge_index, data.num_nodes, data.edge_weight)

        x = self.conv4(x, edge_index).relu()
        #data.edge_index, data.edge_weight = GCNConv.norm(data.edge_index, data.num_nodes, data.edge_weight)

        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class  Decoder(torch.nn.Module):
    def __init__(self, out_channels, sigmoid=True):
        super(Decoder, self).__init__()

        #self.do = nn.Dropout(p=0.5)

        self.conv1 = pyg_nn.GCNConv(channels, 2 * out_channels, cached=False)
        self.conv2 = pyg_nn.GCNConv(2 * out_channels, 4 * out_channels, cached=False)
        self.conv3 = pyg_nn.GCNConv(4 * out_channels, 8 * out_channels, cached=False)
        self.conv4 = pyg_nn.GCNConv(8 * out_channels, 16 * out_channels, cached=False)

        self.conv5 = pyg_nn.GCNConv(16 * out_channels, out_channels, cached=False)


       
    def forward(self, value, edge_index, sigmoid=True):
        
        #value = (x[edge_index[0]] * x[edge_index[1]])#.sum(dim=1)

        value = self.conv1(value, edge_index).relu()
        value = self.conv2(value, edge_index).relu()
        value = self.conv3(value, edge_index).relu()
        value = self.conv4(value, edge_index).relu()
        value = self.conv5(value, edge_index).relu()

        value = value.sum(dim=1)

        return torch.sigmoid(value) if sigmoid else value




'''
    Node classification
'''
class Predictor(torch.nn.Module):
    def __init__(self): #dataset
        super(Predictor, self).__init__()
        #self.conv11 = GCNConv(21, 16) #dataset.num_features, args.hidden
        #self.conv21 = GCNConv(16, len(cfg.objects)) #args.hidden, dataset.num_classes

        #self.conv11 = nn.Conv1d(21, 16, 1, stride=2)
        #self.conv21 = nn.Conv1d(16, len(cfg.objects), 1, stride=2)
    
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

        return F.log_softmax(data, dim=1)
        #return data


'''
    Full network
'''
class graphNet(torch.nn.Module):
    def __init__(self):
        super(graphNet, self).__init__()
        #self.decoder = Decoder(channels, len(cfg.objects))

        self.model = pyg_nn.VGAE(Encoder(len(cfg.objects), channels)).to(dev) #decoder = Decoder(len(cfg.objects))).to(dev)
        #self.model = pyg_nn.ARGVA(Encoder(21, channels)).to(dev)

        self.predictor = Predictor().to(dev)
        
    
    def forward(self, x, train_flag):
                

        x = x.to_data_list()

        #print(cfg._relations)
           
        if train_flag == True:
            x = self.model.split_edges(x[0], test_ratio=0.0)       
        else:
            x = train_test_split_edges(x[0], test_ratio=1.0)            # modified function
                
        x = Batch.from_data_list([x])

        
        if train_flag == True: 
            z = self.model.encode(x.x, x.train_pos_edge_index)          # Encoder input

            p_z = self.predictor(z)                                     # Prediction input

            #q_z = self.model.decoder(z, x.train_pos_edge_index)        # Decoder input: Modelin test fonksiyonunda decoder zaten kullanılıyor.

            return p_z, z, x.train_pos_edge_index, x.train_neg_adj_mask, x.test_pos_edge_index, x.test_neg_edge_index
        else:
            z = self.model.encode(x.x, x.test_pos_edge_index)           # Encoder input

            p_z = self.predictor(z)                                     # Prediction input

            #q_z = self.model.decoder(z, x.test_pos_edge_index)         # Decoder input: Modelin test fonksiyonunda decoder zaten kullanılıyor.

            return p_z, z, x.test_pos_edge_index, x.test_neg_edge_index


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
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.model.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.model.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        print("y and pred", y.shape, pred.shape)

        return roc_auc_score(y, pred, multi_class='ovo'), average_precision_score(y, pred) #multi_class = 'ovr' or 'ovo'



channels = 21
#dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

print('CUDA availability:', torch.cuda.is_available())


total_model = graphNet().to(dev)

print("**********************")
print(total_model)
print("**********************")

LR=0.000001
DATA_AUG = None  #"GCN norm" or "Remove isolated nodes" or None 
optimizer = torch.optim.Adam(total_model.parameters(), lr=LR) #default lr=0.001
nllloss = nn.NLLLoss()
LAYER_NUM = "4-4"
MODEL_NAME = "VGAE" #GAE, VGAE, or ARGVA

isolated_nodes = T.RemoveIsolatedNodes()
#gcnnorm = T.GCNNorm()

if LOG_COMMET:
    experiment.log_parameter('lr', LR)
    experiment.log_parameter('optimizer', optimizer)
    experiment.log_parameter('loss', nllloss)
    experiment.log_parameter('data_augmentation', DATA_AUG)
    experiment.log_parameter('num of encoder layers', LAYER_NUM)
    experiment.log_parameter('model type', MODEL_NAME)



explainer = GNNExplainer(total_model, epochs=200)
node_idx = 21

for epoch in range(0, 200):

    #Training loop
    total_model.train()
    for batch_idx, (graph) in enumerate(train_loader):

            batch_train_idx = 0

            graph.train_mask = graph.val_mask = graph.test_mask = graph.y = None

            #graph = isolated_nodes(graph)
            #graph = gcnnorm(graph)

            target_list = graph.x

            train_flag = True

            optimizer.zero_grad()

            p_z, q_z, train_pos_edge_index, train_neg_adj_mask, test_pos_edge_index, test_neg_edge_index = total_model(graph, train_flag)


            train_neg_edge_list = []
            for i in range(len(train_neg_adj_mask[0])):
                for j in range(len(train_neg_adj_mask[1])):
                        
                        if train_neg_adj_mask[i][j] == True:
                            train_neg_edge_list.append([i,j])
            

            train_neg_edge_index = torch.tensor(train_neg_edge_list, dtype=torch.long)

            train_neg_edge_index = torch.transpose(train_neg_edge_index, 0, 1)

	        # Tüm elemanların hepsinin CPU veya GPU'da olmasına dikkat et.
            #Data(edge_attr=[32, 3], test_neg_edge_index=[2, 1], test_pos_edge_index=[2, 1], train_neg_adj_mask=[15, 15], train_pos_edge_index=[2, 20], val_neg_edge_index=[2, 0], val_pos_edge_index=[2, 0], x=[15, 21], y=[8])
            

            #Compute auc and ap values for the link prediction
            _, link_train_ap = total_model.test(q_z, train_pos_edge_index, train_neg_edge_index) #Custom test function, modified for multi_class

            #Compute acc score for the node classification
            pred = p_z.argmax(dim=1)

            gt = target_list.argmax(dim=1)
            
            train_correct = torch.eq(pred, gt)

            counts_correct = (train_correct == True).sum(dim=0)

            node_score_train = int(torch.sum(counts_correct)) / train_correct.shape[0]

            #node_score_train = int(torch.sum(counts_correct)) / list(graph.batch.shape)[0]  # Derive ratio of correct predictions.


            #Compute node_loss, link_loss +net_loss
            target_list = torch.argmax(target_list, dim=1)
            linkloss = total_model.model.recon_loss(q_z, train_pos_edge_index)
            nodeloss = nllloss(p_z, target_list)

            net_loss = linkloss + nodeloss

            net_loss.backward()
            optimizer.step()

            #step=epoch*len(train_loader)+batch_train_idx


            #if batch_idx % step == 0: 
            if LOG_COMMET:
                experiment.log_metric('Net_loss_train', net_loss.item(), epoch)
                experiment.log_metric('Node_loss_train', nodeloss.item(), epoch)
                experiment.log_metric('Link_loss_train', linkloss.item(), epoch)
                experiment.log_metric('Link_AP_train', link_train_ap, epoch)
                experiment.log_metric('Node_acc_train', node_score_train, epoch)
                #experiment.log_metric('AUC_train', train_auc, epoch)

            #batch_train_idx += 1

            print("Epoch: {:3d}, train_loss: {:.4f}\n".format(epoch, net_loss))

            #Visualization of the results
            node_feat_mask, edge_mask = explainer.explain_node(node_idx, graph.x, graph.edge_index)
            ax, G = explainer.visualize_subgraph(node_idx, graph.edge_index, edge_mask, y=graph.y)
            plt.savefig("graph.jpeg")

            exit()


    #Validation loop
    total_model.eval()
    for batch_idx, (graph) in enumerate(valid_loader):

        with torch.no_grad():

            train = False

            #graph = isolated_nodes(graph)
            #graph = gcnnorm(graph)

            target_list = graph.x

            p_z, q_z, test_pos_edge_index, test_neg_edge_index = total_model(graph, train)

            _, val_link_ap = total_model.model.test(q_z, test_pos_edge_index, test_neg_edge_index)


            #Compute acc score for the node classification
            pred = p_z.argmax(dim=1)

            gt = target_list.argmax(dim=1)
            
            val_correct = torch.eq(pred, gt)

            counts_val_correct = (val_correct == True).sum(dim=0)

            val_node_score = int(torch.sum(counts_val_correct)) / val_correct.shape[0]


            #Compute node_loss, link_loss +net_loss
            target_list = torch.argmax(target_list, dim=1)
            val_linkloss = total_model.model.recon_loss(q_z, test_pos_edge_index)
            val_nodeloss = nllloss(p_z, target_list)

            val_netloss = val_linkloss + val_nodeloss

            if LOG_COMMET:
                experiment.log_metric('Node_loss_val', val_nodeloss.item(), epoch)
                experiment.log_metric('Link_loss_val', val_linkloss.item(), epoch)
                experiment.log_metric('Net_loss_val', val_netloss.item(), epoch)
                experiment.log_metric('Node_acc_val', val_node_score, epoch)
                experiment.log_metric('Link_AP_val', val_link_ap, epoch)
                #experiment.log_metric('AUC_val', auc, epoch)
                                        
            print('Epoch: {:3d}, Val NET loss: {:.4f}, Val node loss: {:.4f}, Val link loss: {:.4f}, AP: {:.4f}'.format(epoch, val_nodeloss, val_linkloss, val_netloss, val_link_ap))



    # Test loop
    total_model.eval()
    for batch_idx, (graph) in enumerate(test_loader):

        with torch.no_grad():

            graph.train_mask = graph.val_mask = graph.test_mask = graph.y = None

            #graph = isolated_nodes(graph)
            #graph = gcnnorm(graph)

            target_list = graph.x

            train_fl = False

            p_z, q_z, test_pos_edge_index, test_neg_edge_index = total_model(graph, train_fl)


	        # Tüm elemanların hepsinin CPU veya GPU'da olmasına dikkat et.
            #Data(edge_attr=[32, 3], test_neg_edge_index=[2, 1], test_pos_edge_index=[2, 1], train_neg_adj_mask=[15, 15], train_pos_edge_index=[2, 20], val_neg_edge_index=[2, 0], val_pos_edge_index=[2, 0], x=[15, 21], y=[8])


            #Compute test link auc and ap values
            _, link_test_ap = total_model.model.test(q_z, test_pos_edge_index, test_neg_edge_index)

            
            #Compute acc score for the node classification
            pred = p_z.argmax(dim=1)

            gt = target_list.argmax(dim=1)
            
            test_correct = torch.eq(pred, gt)

            counts_test_correct = (val_correct == True).sum(dim=0)

            node_score_test = int(torch.sum(counts_test_correct)) / test_correct.shape[0]


            #Compute node_loss, link_loss +net_loss
            target_list = torch.argmax(target_list, dim=1)
            linklosstest = total_model.model.recon_loss(q_z, test_pos_edge_index)
            nodelosstest = nllloss(p_z, target_list)

            net_losstest = linklosstest + nodelosstest

            #if batch_idx % step == 0: 
            if LOG_COMMET:
                experiment.log_metric('Net_loss_test', net_losstest.item(), epoch)
                experiment.log_metric('Node_loss_test', nodelosstest.item(), epoch)
                experiment.log_metric('Link_loss_test', linklosstest.item(), epoch)
                experiment.log_metric('Link_AP_test', link_test_ap, epoch)
                experiment.log_metric('Node_acc_test', node_score_test, epoch)
                #experiment.log_metric('AUC_train', train_auc, epoch)

            #batch_idx += 1

            print("Epoch: {:3d}, test_loss: {:.4f}\n".format(epoch, net_losstest))

        



if LOG_COMMET:
    experiment.end()  

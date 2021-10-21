
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
from torch_geometric.nn import GraphConv, TopKPooling, Set2Set, GCNConv, global_mean_pool
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Batch, DataLoader
import torch_geometric.transforms as T

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from statistics import mean

from pathlib import Path
from pathlib import PurePath
from xml.dom import minidom
from numpy import argmax

import seaborn as sn
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

import time
from datetime import datetime

import Utils.utils as util
from Utils import SECParser
import Utils.config as conf

from Utils import MANIAC as m

from train_test_split_edges import train_split_edges


# Load MANIAC config
cfg = conf.get_maniac_cfg()

# Folder with MANIAC keyframes
_FOLDER = os.getcwd() + "/MANIAC/"


SUMMARY_WRITER = True 

if SUMMARY_WRITER:
    writer = SummaryWriter("./maniac_experiment")


'''

    This is used to create new datasets.
    
    Example:
    train_set = MANIAC_DS( "FOLDER_TO_MANIAC_GRAPHML")
    
    All settings is set inside config.py
    except save new data and create new dataset

'''

# Settings for creating MANIAC dataset
_SAVE_RAW_DATA      = True
_CREATE_DATASET     = True

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
train_dataset = m.ManiacIMDS(_FOLDER, "train") #"train")
test_ds = m.ManiacIMDS(_FOLDER, "test") #"test")
valid_ds = m.ManiacIMDS(_FOLDER, "valid") #"valid")

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


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=True, drop_last=True)


#####################PRINT################################
print("Total batches:\t {}\n=========".format(len(train_loader)+len(test_loader)+len(valid_loader)))
print("Training: \t {}".format(len(train_loader)))
print("Test: \t\t {}".format(len(test_loader)))
print("Validation: \t {}\n=========".format(len(valid_loader)))
#####################PRINT################################



'''
    Model definition (Encoder + Decoder)
'''
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.do = nn.Dropout(p=0.5)

        """
        MANIAC default: kernel256
        Kernel isimlendirme -> 8 * out_channels -> out_channels: kernel128
                            -> 4 * out_channels -> out_channels: kernel64
                            -> 16 * out_channels -> out_channels: kernel256
                            -> 32 * out_channels -> out_channels: kernel512
                            -> 64 * out_channels -> out_channels: kernel1024
                            -> 128 * out_channels -> out_channels: kernel2048
        """
        
        self.conv1 = pyg_nn.GraphConv(len(cfg.objects), 16 * out_channels, aggr="mean")
        self.conv2 = pyg_nn.GraphConv(16 * out_channels, 8 * out_channels, aggr="mean")
        #self.conv3 = pyg_nn.GraphConv(32 * out_channels, 16 * out_channels, aggr="mean")
        #self.conv4 = pyg_nn.GraphConv(128 * out_channels, 64 * out_channels, aggr="mean")
        #self.conv5 = pyg_nn.GraphConv(64 * out_channels, 32 * out_channels, aggr="mean")
        #self.conv6 = pyg_nn.GraphConv(32 * out_channels, 16 * out_channels, aggr="mean")
        #self.conv7 = pyg_nn.GraphConv(16 * out_channels, 8 * out_channels, aggr="mean")
        #self.conv8 = pyg_nn.GraphConv(8 * out_channels, 4 * out_channels, aggr="mean")


        self.conv_mu = pyg_nn.GraphConv(8 * out_channels, out_channels, aggr="mean")
        self.conv_logstd = pyg_nn.GraphConv(8 * out_channels, out_channels, aggr="mean")
       

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()

        x = self.conv2(x, edge_index).relu()

        #x = self.conv3(x, edge_index).relu()
        
        
        #x = self.conv4(x, edge_index).relu()

        #x = self.conv5(x, edge_index).relu()

        #x = self.conv6(x, edge_index).relu()

        #x = self.conv7(x, edge_index).relu()

        #x = self.conv8(x, edge_index).relu()

        #x = self.do(x)
        

        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class  Decoder(torch.nn.Module):
    def __init__(self, out_channels, sigmoid=True):
        super(Decoder, self).__init__()

       
    def forward(self, value, edge_index, sigmoid=True):

        adj = torch.matmul(value, value.t())

        return torch.sigmoid(adj) if sigmoid else adj




'''
    Action recognition branch
'''
class ActionRecognizer(torch.nn.Module):
    def __init__(self, channels): #dataset
        super(ActionRecognizer, self).__init__()

        self.lin2 = nn.Linear(channels, len(cfg.action_map))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, data, batch):

        data = global_mean_pool(data, batch)
        data = self.lin2(data)
        return self.logsoftmax(data)


'''
    Action prediction branch
'''
class ActionPredictor(torch.nn.Module):
    def __init__(self, channels): #dataset
        super(ActionPredictor, self).__init__()

        self.conv1 = pyg_nn.GraphConv(42, 8, aggr="add") #default: aggr="add"
        self.set2set = Set2Set(8, 12, 12) #initial:num_layers=8, iterations=8
        self.lin1 = nn.Linear(16, 16)
        self.lin2 = nn.Linear(16, len(cfg.action_map))


    def forward(self, data, edge_index, batch):

        data = self.conv1(data, edge_index)
        data = self.set2set(data, batch)
        data = self.lin1(data).relu()
        data = F.dropout(data, p=0.5) #training=self.training) #default: p=0.5
        data = self.lin2(data)
        return F.log_softmax(data, dim=-1)




'''
    Full network - GNet
'''
class GNet(torch.nn.Module):
    def __init__(self):
        super(GNet, self).__init__()

        out_channels = 42

        self.model = pyg_nn.VGAE(Encoder(len(cfg.objects), out_channels), decoder=Decoder(out_channels, len(cfg.objects))).to(dev)

        self.actionrecognizer = ActionRecognizer(out_channels).to(dev)

        self.actionpredictor = ActionPredictor(out_channels).to(dev)

        
    def forward(self, x):
                
        batch = x.batch

        x = x.to_data_list()
        node_len = x[0].x.shape[0]

        x = train_split_edges(x[0])

        x = Batch.from_data_list([x])

        
        z = self.model.encode(x.x, x.train_pos_edge_index)          # Encoder input

        a_z = self.actionrecognizer(z, batch)

        ap_z = self.actionpredictor(z, x.train_pos_edge_index, batch)

        return a_z, ap_z, z, x.train_pos_edge_index, x.train_neg_edge_index



'''
    Model parameters
'''
channels = 21
#dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')

print('CUDA availability:', torch.cuda.is_available())

total_model = GNet().to(dev)

print("**********************")
print(total_model)
print("**********************")

LR=0.000001
optimizer = torch.optim.Adam(total_model.parameters(), lr=LR) #default lr=0.001
nllloss = nn.NLLLoss(reduction='sum')
crossent = nn.MSELoss()
bceloss = nn.BCELoss()
LAYER_NUM = "3"
MODEL_NAME = "VGAE"

if SUMMARY_WRITER:
    writer.add_hparams({'lr': LR, 'dataset': "MANIAC", 'layer_num': 3, 'kernel size': 64, 'dropout_encoder': False, 'dropout_decoder': False, 'conv_type': 'Graph Conv', 'action_pred_layer': 1}, {})



'''
    Training loop
'''
for epoch in range(0, 200):

    action_score_trains = []
    action_score_vals = []
    action_score_tests = []

    tr_counter = 0.0
    val_counter = 0.0
    test_counter = 0.0

    tr_correct = 0
    val_correct = 0
    test_correct = 0


    tr_pred_counter = 0.0
    val_pred_counter = 0.0
    test_pred_counter = 0.0


    tr_pred_correct = 0
    val_pred_correct = 0
    test_pred_correct = 0


    avg_score_tr_list = []
    avg_score_val_list = []
    avg_score_test_list = []

    avg_score_pred_tr_list = []
    avg_score_pred_val_list = []
    avg_score_pred_test_list = []

    action_predictions = []
    action_ground_truths = []

    action_predictions_pred = []
    action_ground_truths_pred = []


    #Training loop
    total_model.train()
    for batch_idx, (graph) in enumerate(train_loader):

            target_list = graph.x

            optimizer.zero_grad()

            a_z, ap_z, q_z, train_pos_edge_index, train_neg_edge_index = total_model(graph)


            #Compute action recognition
            pred_action = int(a_z.argmax(dim=1))                            # Use the class with highest probability.
                     
            tr_correct += int((pred_action == int(graph.y.argmax(dim=0))))  # Check against ground-truth labels.

            tr_counter += 1
            
            
            #Compute action prediction
            ap_z = torch.reshape(ap_z, (-1,8))
            ap_z_last = ap_z[-1]

            pred2_action = int(ap_z_last.argmax(dim=0))  # Use the class with highest probability.

            tr_pred_correct += int((pred2_action == int(graph.y.argmax(dim=0))))  # Check against ground-truth labels. 

            tr_pred_counter += 1
            


            #Compute loss
            target_list = torch.argmax(target_list, dim=1)
            #target_list = target_list / 21.0

            gt_action = graph.y.argmax(dim=0)

            actionloss = nllloss(a_z, gt_action.unsqueeze(0)) 
            action_p_loss = nllloss(ap_z[-1].unsqueeze(0), gt_action.unsqueeze(0)) 

            net_loss = action_p_loss + actionloss #+ linkloss_nodeloss + linkloss + actionloss + action_p_loss

            net_loss.backward()
            optimizer.step()


            if SUMMARY_WRITER:
                writer.add_scalar("Net loss/train", net_loss.item(), epoch)
                writer.add_scalar("Action loss/train", actionloss.item(), epoch)
                writer.add_scalar("Action pred loss/train", action_p_loss.item(), epoch)

            print("Epoch: {:3d}, train_loss: {:.4f}\n".format(epoch, net_loss))

    
    avg_action_train = tr_correct / tr_counter
    avg_action_pred_train = tr_pred_correct / tr_pred_counter

    avg_score_tr_list.append(avg_action_train)
    avg_score_pred_tr_list.append(avg_action_pred_train)


    if SUMMARY_WRITER:
        writer.add_scalar("Avg action score/train", avg_action_train, epoch)
        writer.add_scalar("Avg action prediction score/train", avg_action_pred_train, epoch)


       

    #Validation loop
    total_model.eval()
    for batch_idx, (graph) in enumerate(valid_loader):

        with torch.no_grad():

            target_list = graph.x

            a_z_val, ap_z_val, q_z, test_pos_edge_index, test_neg_edge_index = total_model(graph)

            
            #Compute action recognition
            pred_val = int(a_z_val.argmax(dim=1))  # Use the class with highest probability.

            val_correct += int((pred_val == int(graph.y.argmax(dim=0))))  # Check against ground-truth labels.

            val_counter += 1
            
            
            #Compute action prediction
            ap_z_val = torch.reshape(ap_z_val, (-1,8))
            ap_z_val_last = ap_z_val[-1]

            pred2_action_val = int(ap_z_val_last.argmax(dim=0))  # Use the class with highest probability.

            val_pred_correct += int((pred2_action_val == int(graph.y.argmax(dim=0))))  # Check against ground-truth labels. 

            val_pred_counter += 1
            

            #Compute loss
            target_list = torch.argmax(target_list, dim=1)
            #target_list = target_list / 21.0

            gt_action_val = graph.y.argmax(dim=0)
            action_val_loss = nllloss(a_z_val, gt_action_val.unsqueeze(0)) 
            action_pred_val_loss = nllloss(ap_z_val[-1].unsqueeze(0), gt_action_val.unsqueeze(0)) 

            val_netloss = action_pred_val_loss + action_val_loss

            if SUMMARY_WRITER:
                writer.add_scalar("Net loss/val", val_netloss.item(), epoch)
                writer.add_scalar("Action loss/val", action_val_loss.item(), epoch)
                writer.add_scalar("Action pred loss/val", action_pred_val_loss.item(), epoch)

            print('Epoch: {:3d}, Val NET loss: {:.4f}'.format(epoch, val_netloss)) # Val link loss: {:.4f}, val_linkloss_nodelos))

    
    avg_action_val = val_correct / val_counter
    avg_action_pred_val = val_pred_correct / val_pred_counter

    avg_score_val_list.append(avg_action_val)
    avg_score_pred_val_list.append(avg_action_pred_val)

    if SUMMARY_WRITER:
        writer.add_scalar("Avg action score/val", avg_action_val, epoch)
        writer.add_scalar("Avg action prediction score/val", avg_action_pred_val, epoch)
    


    # Test loop
    total_model.eval()
    for batch_idx, (graph) in enumerate(test_loader):

        with torch.no_grad():

            target_list = graph.x

            a_z_test, ap_z_test, q_z, test_pos_edge_index, test_neg_edge_index = total_model(graph)

            
            #Compute action recognition
            pred_test = int(a_z_test.argmax(dim=1))  # Use the class with highest probability.

            test_correct += int((pred_test == int(graph.y.argmax(dim=0))))  # Check against ground-truth labels.

            action_predictions.append(pred_test)
            action_ground_truths.append(int(graph.y.argmax(dim=0)))

            test_counter += 1
            
    
            #Compute action prediction
            ap_z_test = torch.reshape(ap_z_test, (-1,8))
            ap_z_test_last = ap_z_test[-1]

            pred2_action_test = int(ap_z_test_last.argmax(dim=0))  # Use the class with highest probability.

                     
            test_pred_correct += int((pred2_action_test == int(graph.y.argmax(dim=0))))  # Check against ground-truth labels. 

            action_predictions_pred.append(pred2_action_test)
            action_ground_truths_pred.append(int(graph.y.argmax(dim=0)))

            test_pred_counter += 1
            

            #Compute loss
            target_list = torch.argmax(target_list, dim=1)
            #target_list = target_list / 21.0

            gt_action_test = graph.y.argmax(dim=0)
            action_test_loss = nllloss(a_z_test, gt_action_test.unsqueeze(0))
            action_pred_test_loss = nllloss(ap_z_test[-1].unsqueeze(0), gt_action_test.unsqueeze(0))

            net_losstest =  action_pred_test_loss + action_test_loss

            if SUMMARY_WRITER:
                writer.add_scalar("Net loss/test", net_losstest.item(), epoch)
                writer.add_scalar("Action loss/test", action_test_loss.item(), epoch)
                writer.add_scalar("Action pred loss/test", action_pred_test_loss.item(), epoch)
               

            print("Epoch: {:3d}, test_loss: {:.4f}\n".format(epoch, net_losstest))

    
    avg_action_test = test_correct / test_counter
    avg_action_pred_test = test_pred_correct / test_pred_counter

    avg_score_test_list.append(avg_action_test)
    avg_score_pred_test_list.append(avg_action_pred_test)

    if SUMMARY_WRITER:
        writer.add_scalar("Avg action score/test", avg_action_test, epoch)
        writer.add_scalar("Avg action prediction score/test", avg_action_pred_test, epoch)


writer.flush()
writer.close()

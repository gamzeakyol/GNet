# Adapted from the Github link: https://github.com/dawidejdeholm/dj_graph


from IPython.core.debugger import set_trace

import Utils.config as conf
import glob
import re
import os
import json
import networkx as nx
import numpy as np
import pandas as pd
from .utils import *

from torch_scatter import scatter_add
# Load BAC config
cfg = conf.get_bac_cfg()

rel2idx = {"contact": 0, "above": 1, "below": 2, "left of": 3, "right of": 4, "behind of": 5, 
           "in front of": 6, "inside": 7, "surround": 8, "moving together": 9, "halting together": 10, 
           "fixed moving together": 11, "getting close": 12, "moving apart": 13, "stable": 14, "temporal": 15}

# # binary edge, count
# counter_dict = {}

# #single edge count
# single_dict = {} 

# object_dict = {}

def binary_encode(lst, N):
    a = np.zeros(N, dtype=np.float32)
    a[lst] = 1
    return a


def get_object_list(path):
    # ['DREHER', 'bimacs_derived_data', 'subject_1', 'task_5_k_cereals', 'take_3', 'spatial_relations', 'frame_0.json']
    # ['E:', 'pyg_dev', 'DREHER', 'bimacs_derived_data', 'subject_1', 'task_1_k_cooking', 'take_0', 'spatial_relations', 'frame_0.json']
    i = 0
    for w in path:
        if w == "bimacs_derived_data":
            path[i] = "bimacs_derived_data_3D"
        if w == "spatial_relations":
            path[i] = "3d_objects"
        i += 1

    #final_path = os.getcwd() + '/bimacs_derived_data_3D/' + path[2] + '/' + path[3] + '/' + path[4] + '/3d_objects/' + path[6]
    final_path = '/'.join(path)
    #print("get_object_list: ", final_path)
    with open(final_path) as f:
        data = json.load(f)

    temp_dict = {}
    cnt = 0

    for obj in data:
        temp_dict[cnt] = obj['class_name']
        cnt += 1

    return temp_dict


# Takes a single JSON file and outputs a graph
def json_to_graph(input_path, target):
    graph = nx.DiGraph()

    node_list_path = os.path.normpath(input_path).split(os.sep)
    node_list = get_object_list(node_list_path)
    
    # Load JSON Object to list
    #print("json_to_graph ", input_path)
    with open(input_path) as f:
          data = json.load(f)

    for index, name in node_list.items():
        graph.add_node(index, x=cfg._objects[cfg.objects.index(name)])   
        
#         # Object stats counter
#         if name in object_dict:
#             object_dict[name] += 1
#         else:
#             object_dict[name] = 1
    
    edge_data_ = []
    # Populate the graph with nodes and edges
    for obj in data:
        relation_name = obj['relation_name']
        edge_data_.append((obj['object_index'], obj['subject_index'], rel2idx[relation_name]))
        
#         # Counter
#         if relation_name in single_dict:
#             single_dict[relation_name] += 1
#         else:
#             single_dict[relation_name] = 1
        
    df = pd.DataFrame(edge_data_, columns=["o1", "o2", "r"])
    gdf = df.groupby(['o1','o2'])    
    for pair, group in gdf:    
        bin_vec = binary_encode(group["r"].values, 16)    
        graph.add_edge(pair[0], pair[1],edge_attr=bin_vec)

#         k = ''.join(map(str,bin_vec.astype(int)))

#         if k in counter_dict:
#             counter_dict[k] += 1
#         else:
#             counter_dict[k] = 1


    # If the ground truth contain null value set the action to undefined
    if(target is None):
        return -1
        #graph.graph['features'] = _actions[action_map.index('undefined')]
    else:
        graph.graph["features"] = cfg._actions[cfg.action_map.index(cfg.original_index[target])]

    if len(graph.nodes()) == 0:
        print("------- NO NODES")
        return -1

    if len(graph.edges()) == 0:
        print("------- NO EDGES")
        return -1

    return graph


def get_target_action(cnt, ground_truth):
    # Find target by comparing the frame count with ground truth
    for index, item in enumerate(ground_truth['right_hand']):
        if(index % 2 == 0 and index != 0):
            if(cnt <= item):
                return ground_truth['right_hand'][index-1]
    return 'Not found'


def take_to_graph_list(path):
    graph_list = []
    # Get ground truth path
    gt_name = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')][0]
    gt_path = path + gt_name

    # Extract all json files in spatial_relations and sort them    
    json_files = [pos_json for pos_json in os.listdir(path + 'spatial_relations') if pos_json.endswith('.json')]
    json_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Load ground truth
    with open(gt_path) as f:
          ground_truth = json.load(f)

    for file in json_files:
        frame_cnt = int(re.search(r'\d+', file).group())
        graphs = json_to_graph(path + 'spatial_relations/' + file, get_target_action(frame_cnt, ground_truth))

        if graphs != -1:
            graph_list.append(graphs)
        else:
            print("file:", file, "frame_cnt:", frame_cnt, "gt:", ground_truth)

    return graph_list


def generate_graphs(MAIN_PATH=None, _seperate=True, _settings="full"):

    if _settings.lower() == "training":
        _allowed_subjects = [1] #---> all dataset, now using a subset 
        _allowed_take = [1] #[1,2,3,4,5,6,7,8,9]
    elif _settings.lower() == "test":
        _allowed_subjects = [1]
        _allowed_take = [2] #[1,2,3,4,5,6,7,8,9]
    elif _settings.lower() == "validation":
        _allowed_subjects = [2] #[2,3,4,5,6]
        _allowed_take = [2]
    else:
        print("[BAD DATASET ERROR] You can only chose between: training, test, validation.", _settings)
        return 0

    if _seperate:
        print("Creating graphs to dict.")
        all_data = {}
    else:
        print("Appending graphs to array.")
        all_data = []

    # Iterate subjects
    for dic in list(glob.glob(MAIN_PATH+'/*/')):
        # Iterate tasks
        for sub_dic in list(glob.glob(dic+'/*/')):
            # Iterate takes
            for sub_sub_dic in list(glob.glob(sub_dic+'/*/')):
                # Get subject number
                sub = int(re.search(r'\d+', dic).group())

                # If the subject is allowed
                if sub in _allowed_subjects:
                    # Get task number
                    task = int(re.search(r'\d+', sub_dic[len(dic):]).group())
                    # Get take number
                    take = int(re.search(r'\d+', sub_sub_dic[len(sub_dic):]).group())
                    if take in _allowed_take:
                        name = "take_" + str(sub) + "_" + str(task) + "_" + str(take)                        
                        if _seperate:
                            all_data[name] = take_to_graph_list(sub_sub_dic)
                        else:
                            all_data += take_to_graph_list(sub_sub_dic)
    return all_data


# def full_test(loader, model, max_num_nodes, device):
#     model.eval()
#     ap_predict_list = np.array([])
#     ap_gt_list = np.array([])

#     reconstruct_predict_list = []
#     reconstruct_gt_list = []
#     top3_list = []
#     num_nodes_list = []

#     for data in loader:
#         data = data.to(device)
#         batch_size = data.batch[-1].item() + 1

#         y_hat, logvar, mu, _, y_ap = model(data)
#         pred = y_ap.max(1)[1]

#         # Creating targets
#         target_adj = to_dense_adj_max_node(data.edge_index, data.x, data.edge_attr, data.batch, max_num_nodes).cuda()
#         target = data.y.view(_bs, -1)
#         y_ap_true = target.argmax(axis=1)

#         prediction = y_hat.view(_bs, -1, max_num_nodes)
#         ap_predict_list = np.append(ap_predict_list, pred.cpu().detach().numpy())
#         ap_gt_list = np.append(ap_gt_list, y_ap_true.cpu().detach().numpy())

#         batch_size = data.batch[-1].item() + 1

#         one = data.batch.new_ones(data.batch.size(0))
#         num_nodes = scatter_add(one, data.batch, dim=0, dim_size=batch_size)
#         num_nodes_list.append(num_nodes.cpu().detach().numpy())


#         # Get top 3 predictions
#         indices = torch.topk(y_ap, 3)
#         for i in range(len(indices[0])):
#             batch_top_list = []
#             for top in indices[0][i].tolist():
#                 batch_top_list.append(action_map[y_ap[i].tolist().index(top)])

#             top3_list.append(batch_top_list)

#         gr_pred = prediction.cpu().detach().numpy()
#         gr_gt = target_adj.cpu().detach().numpy()

#         reconstruct_predict_list.append(gr_pred)
#         reconstruct_gt_list.append(gr_gt)

#     return reconstruct_predict_list, reconstruct_gt_list, ap_predict_list, ap_gt_list, top3_list, num_nodes_list

def full_test(loader, model, max_num_nodes, device):
    model.eval()
    ap_predict_list = np.array([])
    ap_gt_list = np.array([])

    reconstruct_predict_list = []
    reconstruct_gt_list = []
    top3_list = []
    num_nodes_list = []

    for data in loader:
        data = data.to(device)
        batch_size = data.batch[-1].item() + 1

        logvar, mu, _, y_ap = model(data)
        pred = y_ap.max(1)[1]

        # Creating targets
        target = data.y.view(cfg.batch_size, -1) # reshape target
        y_ap_true = target.argmax(axis=1)

        ap_predict_list = np.append(ap_predict_list, pred.cpu().detach().numpy())
        ap_gt_list = np.append(ap_gt_list, y_ap_true.cpu().detach().numpy())

        batch_size = data.batch[-1].item() + 1

        # one = data.batch.new_ones(data.batch.size(0))
        # num_nodes = scatter_add(one, data.batch, dim=0, dim_size=batch_size)
        # num_nodes_list.append(num_nodes.cpu().detach().numpy())


        # Get top 3 predictions
        indices = torch.topk(y_ap, 3)
        for i in range(len(indices[0])):
            batch_top_list = []
            for top in indices[0][i].tolist():
                batch_top_list.append(cfg.action_map[y_ap[i].tolist().index(top)])

            top3_list.append(batch_top_list)


    return ap_predict_list, ap_gt_list, top3_list

# !/usr/local/bin/python
from IPython.core.debugger import set_trace

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from pathlib import PurePath
import networkx as nx
import numpy as np

from torch_scatter import scatter_add
import torch

"""
    Creates ont hot encoding of a list.

"""
def one_hot_string(map):
    values = np.array(map)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded


def get_action_from_encoding(enc):
    idx = _actions.index(enc)
    return action_map[idx], idx


####################### GRAPH #######################

"""
    Creates the temporal concatenate between graphs.

    concatenateTemporal is the main function. Rest is help functions.
"""
def createTemporalDict(in_dict):
    temporal_concat = {}

    for key in in_dict:
        temporal_concat[key] = {}
        for seq in in_dict[key]:
                    temporal_concat[key][seq] = [concatenateTemporal(in_dict[key][seq])]
    return temporal_concat


def nodesMatch(n1, n_list):
    for i, node in enumerate(n_list, start=0):
        if node[1] == n1:
            return True, i

    return False, 0


def calculateTemporalEdges(full_gr, nodes, index_list, _relations, spatial_map):
    temporal_rel = _relations[spatial_map.index("temporal")]
    old_nodes = []
    node_cnt = index_list[-1]


    for index in index_list:
        old_nodes.append(full_gr.nodes[index])

    for index in index_list:
        match, ind = nodesMatch(full_gr.nodes[index], nodes)
        if match:
            full_gr.add_edge(index, node_cnt, edge_attr=temporal_rel)
            node_cnt += 1

def concatenateTemporal(graphs, _relations, spatial_map):

    print("graphs", graphs, _relations, spatial_map)

    graph_nx = nx.DiGraph()
    graph_nx.graph["features"] = graphs[0].graph['features']
    node_cnt = 0
    node_index_list = []

    print("graphnx", graph_nx)

    for i, graph in enumerate(graphs, start=0):

        print("graph in concatenateTemporal", graph)

        if len(node_index_list) > 0:
            calculateTemporalEdges(graph_nx, graph.nodes(data=True), node_index_list, _relations, spatial_map)

        node_index_list = []

        for node in graph.nodes(data=True):
            graph_nx.add_node(node_cnt, x=node[1]['x'])
            node_index_list.append(node_cnt)
            node_cnt += 1

        for edge in graph.edges():
            graph_nx.add_edge(node_index_list[edge[0]], node_index_list[edge[1]], edge_attr=graph.get_edge_data(edge[0], edge[1])['edge_attr'])

    empty_list = []
    for node in graph_nx.nodes(data=True):
        if node[1] == {}:
            empty_list.append(node[0])

    for node in empty_list:
        graph_nx.remove_node(node)

    print("graph_nx", graph_nx)
    return graph_nx


'''
Method to find all xml files.

Input: String of folder
'''
def find_xmls(path):
    all_xmls = []
    for filename in Path(path).rglob('*.xml'):
        all_xmls.append(filename)
    return all_xmls

'''
Method to split xml into action and s

Input: Array with strings of xml 
'''
def parse_xml_info(xml):
    if isinstance(xml, Path):
        ss = str(PurePath(xml))
        if ss[-3:] != 'xml':
            raise ValueError("Must be XML! recieved {}".format(ss))
        
        path_split = xml.parts
        split = path_split[-2].split('_')
        action = "".join(split[:-1]).lower()
        seq = split[-1]
        
        return action, int(seq)
    else:
        raise ValueError("Must be XML! recieved {}".format(xml))
        
        


'''
    Creates a dense adjacency matrix from pg data.
    With normalization
'''
def to_dense_adj_max_node_norm(edge_index, x, edge_attr, batch=None, max_node=None):
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    
    batch_size = batch[-1].item() + 1
    
    if max_node is None:
        max_num_nodes = num_nodes.max().item()
    else:
        max_num_nodes = max_node
    
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_add(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    
    size = [batch_size, max_num_nodes, max_num_nodes]
    
    size = size
    dtype = torch.float
    
    adj = torch.zeros(size, dtype=dtype, device=edge_index.device)

    edge_index_0 = batch[edge_index[0]].view(1, -1)
    edge_index_1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    edge_index_2 = edge_index[1] - cum_nodes[batch][edge_index[1]]
    
    # Normalize the edges on the length of pre-defined spatial objects.
    _ea = []
    for ea in edge_attr:
        _ea.append((ea[0].item()+1)/len(spatial_map))
    
    adj[edge_index_0, edge_index_1, edge_index_2] = torch.FloatTensor(_ea).cuda()

    # Normalize objects in the diagonal
    objects_sum = x.argmax(dim=1).type(dtype)
    objects_sum = objects_sum/len(_objects)

    object_size = [batch_size, num_nodes.max().item()]
    object_mat = torch.zeros(object_size, dtype=dtype, device=edge_index.device)
    
    obj_offset = 0

    # Creates the adjacency matrix of size [max_node*max_node].
    for b in range(batch_size):
        temp = torch.zeros(num_nodes.max().item(), dtype=dtype, device=edge_index.device)
        _obj = objects_sum[obj_offset:(obj_offset+num_nodes[b])].type(dtype)
        obj_offset += num_nodes[b]
        dd = torch.cat((_obj, temp), dim=0)
        dd = dd[:num_nodes.max().item()]
        object_mat[b] = dd
            
    adj.as_strided(object_mat.size(), [adj.stride(0), adj.size(2) + 1]).copy_(object_mat)
    
    # Error check to see that the diagonal is not zero.
    for i in range(len(adj)):
        if torch.diag(adj[i]).sum().item() == 0:
            print("ERROR! ZERO DIAGONAL!", i)

    return adj


'''
    Creates a dense adjacency matrix from pg data.
    Without normalization
'''
def to_dense_adj_max_node(edge_index, x, edge_attr, batch=None, max_node=None):
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    
    batch_size = batch[-1].item() + 1
    
    if max_node is None:
        max_num_nodes = num_nodes.max().item()
    else:
        max_num_nodes = max_node
    
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_add(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    
    size = [batch_size, max_num_nodes, max_num_nodes]
    
    size = size
    dtype = torch.float
    
    adj = torch.zeros(size, dtype=dtype, device=edge_index.device)

    edge_index_0 = batch[edge_index[0]].view(1, -1)
    edge_index_1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    edge_index_2 = edge_index[1] - cum_nodes[batch][edge_index[1]]
    
    # Normalize the edges on the length of pre-defined spatial objects.
    _ea = []
    for ea in edge_attr:
        _ea.append(1)

    adj[edge_index_0, edge_index_1, edge_index_2] = torch.FloatTensor(_ea).cuda()

    # Normalize objects in the diagonal
    objects_sum = x.argmax(dim=1).type(dtype)
    objects_sum[objects_sum > 0] = 1

    object_size = [batch_size, num_nodes.max().item()]
    object_mat = torch.zeros(object_size, dtype=dtype, device=edge_index.device)
    
    obj_offset = 0
    
    # Creates the adjacency matrix of size [max_node*max_node].
    for b in range(batch_size):
        temp = torch.zeros(num_nodes.max().item(), dtype=dtype, device=edge_index.device)
        _obj = objects_sum[obj_offset:(obj_offset+num_nodes[b])].type(dtype)
        obj_offset += num_nodes[b]
        dd = torch.cat((_obj, temp), dim=0)
        dd = dd[:num_nodes.max().item()]
        object_mat[b] = dd
            
    adj.as_strided(object_mat.size(), [adj.stride(0), adj.size(2) + 1]).copy_(object_mat)
    
    # Error check to see that the diagonal is not zero.
    for i in range(len(adj)):
        if torch.diag(adj[i]).sum().item() == 0:
            print("ERROR! ZERO DIAGONAL!", i)

    return adj

######
def get_adj_matrix(edge_index, x, edge_attr, batch=None, max_node=None):
    pass

####

def edge_node_features(edge_index, x, edge_attr, batch=None, max_node=None):
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    
    batch_size = batch[-1].item() + 1
    
    if max_node is None:
        max_num_nodes = num_nodes.max().item()
    else:
        max_num_nodes = max_node
    
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_add(one, batch, dim=0, dim_size=batch_size)
    
    x_prime = torch.zeros(size=[batch_size * max_num_nodes, x.size(1)], 
                        dtype=torch.float, 
                        device=edge_index.device)

    to_copy = []
    to = np.arange(batch_size * max_num_nodes)
    i = np.arange(0, batch_size * max_num_nodes, max_num_nodes)
    j = i + num_nodes.cpu().numpy()
    for a,b in zip(i,j):
        to_copy.extend(to[a:b])
        
    x_prime[to_copy, :] = x
    
    return x_prime


# def get_edge_attr_targets(edge_index, x, edge_attr, graphs, batch=None, max_node=None):
#     if batch is None:
#         batch = edge_index.new_zeros(edge_index.max().item() + 1)
    
#     batch_size = batch[-1].item() + 1
    
#     if max_node is None:
#         max_num_nodes = num_nodes.max().item()
#     else:
#         max_num_nodes = max_node
    
#     e_prime = torch.zeros(size=[batch_size * max_num_nodes * max_num_nodes, edge_attr.size(1)], 
#                         dtype=torch.float, 
#                         device=edge_index.device)
    

#     to_copy = []
#     for f in range(batch_size):
#         frame_ = graphs [f]
#         idx = (frame_==1).nonzero()[0]
#         to_copy.extend(idx + max_num_node * f)
        
#     e_prime[to_copy, :] = edge_attr

#     return e_prime

from sklearn.metrics import roc_auc_score, average_precision_score
def calc_auc_roc(gt, pred, node_list, _threshold=0.5):

    if len(gt) != len(pred):
        raise Exception("Invalid length of ground truth and prediction! GT {:d} and pred {:d}".format(len(gt), len(pred))) 
    
    num_matrix = len(gt)
    
    _correct_node_length = 0
    _correct_objects_length = 0
    _correct_objects = 0
    _total_objects = 0
    _roc_auc_score = 0
    _max_nodes_in_graph = 0
    _average_precision_score = 0
    graph_count = 0
    
    _node_diff_dict = {0: 0}
    
    for idx in range(num_matrix):
        for i in range(len(gt[idx])):
            gt_diag = np.round(np.diag(gt[idx][i]))
            cr_diag = np.round(np.diag(pred[idx][i]))
            cr_nodes = np.count_nonzero(cr_diag)
            graph_count += 1
        
            if node_list[idx][i] == cr_nodes:
                _max_nodes_in_graph = node_list[idx][i]
                
                _node_diff_dict[0] += 1
            else:
                diff = cr_nodes - node_list[idx][i]
                if _node_diff_dict.get(diff) is None:
                    _node_diff_dict[diff] = 1
                else:
                    _node_diff_dict[diff] += 1
       
            gt_flatten = np.ceil(gt[idx][i].flatten())

            pred[idx][pred[idx] <= _threshold] = 0
            pred[idx][pred[idx] > _threshold] = 1

            cr_flatten = np.ceil(pred[idx][i].flatten())
            _roc_auc_score += roc_auc_score(gt_flatten, cr_flatten)
            _average_precision_score += average_precision_score(gt_flatten, cr_flatten)
    
    print("Average ROC AUC SCORE:", _roc_auc_score/graph_count)
    print("Average precision score:", _average_precision_score/graph_count)
    
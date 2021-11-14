# Adapted from the Github link: https://github.com/dawidejdeholm/dj_graph


from torch.utils.data import Dataset
from pathlib import Path
from pathlib import PurePath
import networkx as nx
import torch_geometric as tg
import numpy as np
from torch_scatter import scatter_add
import torch
import Utils.utils as util
from Utils import SECParser as sp
import Utils.config as conf

cfg = conf.get_maniac_cfg()

'''

    Creates MANIAC dataset to work with PyTorch Geometric.

'''
class MANIAC_DS(Dataset):
    def __init__(self, root_dir):
        self.window = cfg.time_window
        self.root_dir = root_dir
        self.all_xmls = util.find_xmls(self.root_dir)
        self.sp = sp.SECp()
        self.temporal = cfg.temporal_graphs

        for xml in self.all_xmls:
            self.dict_with_graphs = self.sp(xml)

        self.samples = sp.create_big_list(self.dict_with_graphs)

    def __len__(self):
        return len(self.samples) - self.window

    def __getitem__(self, idx):

        if self.window > 0:

            x = self.samples[idx:idx+self.window]

            current_action = self.samples[idx].graph['features']
            step_back = 0

            for i in range(self.window):
                if self.samples[idx+i].graph['features'] != current_action:
                    step_back += 1

            if step_back > 0:
                x = self.samples[idx-step_back:idx+self.window-step_back]
            else:
                x = self.samples[idx:idx+self.window]
        else:
            x = self.samples[idx]

        if self.temporal:
            print("Im here", x)
            return util.concatenateTemporal(x, cfg._relations, cfg.spatial_map)
        else:
            return x

import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar)

'''

    Creates MANIAC InMemory dataset to work with PyTorch Geometric.

    Read more about it at
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets

'''
class ManiacIMDS(InMemoryDataset):
    def __init__(self,
                root,
                dset="train",
                transform=None):
        super(ManiacIMDS, self).__init__(root, transform)

        if dset == "train":
            path = self.processed_paths[0]
        elif dset == "valid":
            path = self.processed_paths[1]
        else:
            path = self.processed_paths[2]

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['maniac_training_' + str(cfg.time_window) + 'w.pt', 'maniac_validation_' + str(cfg.time_window) + 'w.pt', 'maniac_test_' + str(cfg.time_window) + 'w.pt']

    @property
    def processed_file_names(self):
        return ['maniac_training_' + str(cfg.time_window) + 'w.pt', 'maniac_validation_' + str(cfg.time_window) + 'w.pt', 'maniac_test_' + str(cfg.time_window) + 'w.pt']

    def download(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def process(self):
        big_slices = []
        for raw_path, path in zip(self.raw_paths, self.processed_paths):

            big_data = []
            graphs = torch.load(raw_path)

            # Creates torch_geometric data from networkx graphs
            # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
            for graph in graphs:
                print("graph", graph)
                graph = graph[0] #added by Gamze
                G = nx.convert_node_labels_to_integers(graph)
                G = G.to_directed() if not nx.is_directed(G) else G
                edge_index = torch.tensor(list(G.edges)).t().contiguous()

                data = {}

                for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
                    for key, value in feat_dict.items():
                        data[key] = [value] if i == 0 else data[key] + [value]

                for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
                    for key, value in feat_dict.items():
                        data[key] = [value] if i == 0 else data[key] + [value]

                for key, item in data.items():
                    try:
                        data[key] = torch.tensor(item)
                    except ValueError:
                        pass

                # Creates the tg data
                data['edge_index'] = edge_index.view(2, -1)
                data = tg.data.Data.from_dict(data)
                data.y = torch.tensor(graph.graph['features'])

                # This is not used, can be useful in future development if the sequence id is needed.
                #data.seq = torch.tensor([graph.graph['seq']])

                if cfg.skip_connections:
                    if data.edge_attr is not None:
                        big_data.append(data)
                else:
                    big_data.append(data)

            for graph in big_data:
                if graph.edge_attr is None:
                    print(graph)
                    print(graph.edge_attr)
                    print(action_map[graph.y.argmax().item()])
                    break

            torch.save(self.collate(big_data), path)

def test(loader, model, max_num_nodes, device):
    model.eval()
    ap_predict_list = np.array([])
    ap_gt_list = np.array([])
    
    reconstruct_predict_list = []
    reconstruct_gt_list = []
    num_nodes_list = []

    for data in loader:
        data = data.to(device)
        batch_size = data.batch[-1].item() + 1

        y_hat, logvar, mu, _, y_ap = model(data)
        pred = y_ap.max(1)[1]
        
        batch_size = data.batch[-1].item() + 1

        one = data.batch.new_ones(data.batch.size(0))
        num_nodes = scatter_add(one, data.batch, dim=0, dim_size=batch_size)
        num_nodes_list.append(num_nodes.cpu().detach().numpy())
        
        # Creating targets
        target_adj = util.to_dense_adj_max_node(data.edge_index, data.x, data.edge_attr, data.batch, max_num_nodes).cuda()
        target = data.y.view(cfg.batch_size, -1)
        y_ap_true = target.argmax(axis=1)
        
        prediction = y_hat.view(cfg.batch_size, -1, max_num_nodes)
        ap_predict_list = np.append(ap_predict_list, pred.cpu().detach().numpy())
        ap_gt_list = np.append(ap_gt_list, y_ap_true.cpu().detach().numpy())

        gr_pred = prediction.cpu().detach().numpy()
        gr_gt = target_adj.cpu().detach().numpy()
        
        reconstruct_predict_list.append(gr_pred)
        reconstruct_gt_list.append(gr_gt)                
        
    return reconstruct_predict_list, reconstruct_gt_list, ap_predict_list, ap_gt_list, num_nodes_list

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
import torch
import torch_geometric as tg
from torch_geometric.data import DataLoader
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar)
import networkx as nx
import Utils.utils as util
import Utils.config as conf
import Utils.bad_utils as b_utils

# Load BAC config
cfg = conf.get_bac_cfg()

'''

    Creates BA dataset to work with PyTorch Geometric.

'''
class BAD_DS(Dataset):
    """Bimanual Action Dataset dataset."""

    def __init__(self, root_dir=None, ds_type="full", transform=None):
        self.to_tensor = transforms.ToTensor()
        self.root_dir = root_dir
        self.samples = b_utils.generate_graphs(self.root_dir, _seperate=cfg.seperate_samples, _settings=ds_type)
        self.transform = transform
        self.window = cfg.time_window
        self.temporal = cfg.temporal_graphs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        if self.window > 0:              
            x = self.samples[idx:idx+self.window]            
            current_action = self.samples[idx].graph['features']
            step_back = 0

            if idx > self.window:
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
            return util.concatenateTemporal(x, cfg._relations, cfg.spatial_map)
        else:
            return x

'''

    Creates BIMANUAL ACTION InMemory dataset to work with PyTorch Geometric.

    Read more about it at
    https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets

'''
class BadIMDS(InMemoryDataset):
    def __init__(self,
                root,
                dset="train",
                transform=None):
        super(BadIMDS, self).__init__(root, transform)

        if dset == "train":
            path = self.processed_paths[0]
        elif dset == "valid":
            path = self.processed_paths[1]
        elif dset == "test":
            path = self.processed_paths[2]            

        print("LOADING PATH: ", path)
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):        
        return ['bad_training_' + str(cfg.time_window) + 'w.pt', 'bad_validation_' + str(cfg.time_window) + 'w.pt', 'bad_test_' + str(cfg.time_window) + 'w.pt']

    @property
    def processed_file_names(self):
        return ['bad_training_' + str(cfg.time_window) + 'w.pt', 'bad_validation_' + str(cfg.time_window) + 'w.pt', 'bad_test_' + str(cfg.time_window) + 'w.pt']

    def download(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def process(self):        
        big_slices = []    
        for raw_path, path in zip(self.raw_paths, self.processed_paths):   
            print(raw_path, path)         
            big_data = []
            
            graphs = torch.load(raw_path)

            j = 0            
            for graph in graphs:   
                if cfg.time_window == 1:
                    graph = graph[0]                              
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
                
                data['edge_index'] = edge_index.view(2, -1)
                data = tg.data.Data.from_dict(data)
                data.num_nodes = len(graph)
                data.y = torch.tensor(graph.graph['features'])
                
                big_data.append(data)

            torch.save(self.collate(big_data), path)

# Adapted from the Github link: https://github.com/dawidejdeholm/dj_graph


import warnings
import xml.etree.ElementTree as ET
from glob import glob
import networkx as nx
import numpy as np
import Utils.utils as util
import Utils.config as conf

cfg = conf.get_maniac_cfg()

'''
A SEC parser

Create networkx graphs from MANIACs GraphML.xml files.

'''

class SECp():

    def __init__(self, all_edges=False):
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError('We need xml.elementtree.ElementTree!')

        self.graph_dict = {}
        self.action_dict = {}
        self.actions = {}
        self.all_edges = all_edges

    def __call__(self, path=None, action=None):
        if path is not None:
            self.xml = ET.parse(path)
            self.path = path
        else:
            raise ValueError("Must specify path!")

        # Get all XML with action and sequence
        self.action, self.seq = util.parse_xml_info(self.path)

        # Create the graphs
        self.create_graph(self.xml)

        # Returns all keyframes as a list
        return self.get_keyframes_as_list()

    '''

        Method to create networkx graph

    '''
    def create_graph(self, graph_element):
        root = graph_element.getroot()
        node_count = 0
        nodes_map = {}
        keep_graph = True
        replace_object_dict = {}

        # Objects in MANIAC GraphML is defined as they appear in the frame.
        # This helps to use correct item label for each node.
        for root_replace_object in root.iter('ReplaceObject'):
            replace_objects = root_replace_object.findall('Object')

            # Creates a dict to store object id that and the new replaced values.
            for obj in replace_objects:
                replace_object_dict[int(obj.attrib['id'])] = obj.attrib['obj']

        for keyframe in root.iter('KeyFrame'):
            # Creates a Directed graph structure
            graph = nx.DiGraph()
            # Add graph features
            graph.graph["features"] = cfg._actions[cfg.action_map.index(self.action)]
            graph.graph['seq'] = self.seq

            # Gets the action's identifier
            action_id = np.argmax(cfg._actions[cfg.action_map.index(self.action)])
            self.actions[action_id] = cfg.action_map.index(self.action)

            # Add all nodes and edges to list.
            keyframe_id = keyframe.attrib['ID']
            nodes = keyframe.findall('Node')
            edges = keyframe.findall('Edge')

            skip_nodes = []

            for node in nodes:
                # This is not used yet,
                # but for future development the position of the node may be used.
                pos_X = float(node.attrib['pos_X'])
                pos_Y = float(node.attrib['pos_Y'])
                pos_Z = float(node.attrib['pos_Z'])
                n_type = node.attrib['type']
                n_id = node.attrib['id']

                # Check if the node should be added to nodes_map and gives the node a new value.
                # The value of nodes need to be incremental order such as [0, 1, 2...., n]
                if nodes_map.get(n_id) is None:
                    if cfg.objects.index(replace_object_dict.get(int(n_id))) != cfg.objects.index('null'):
                        nodes_map[n_id] = node_count
                        node_count += 1
                    else:
                        skip_nodes.append(n_id)

                # Replace the nodes id with correct x feature.
                # If acceptable, the node is added to the graph with one hot encoding of the object.
                if int(n_id) in replace_object_dict:
                    if cfg.objects.index(replace_object_dict.get(int(n_id))) != cfg.objects.index('null'):
                        graph.add_node(nodes_map[n_id], x=cfg._objects[cfg.objects.index(replace_object_dict.get(int(n_id)))])

                else:
                    # If object not defined it will be called null,
                    # this is very bad with missing features.
                    graph.add_node(nodes_map[n_id], x=cfg._objects[cfg.objects.index("null")])
                    print("This is very bad, missing features! Please look at:")
                    print("Action:", self.action, "seq:", self.seq, "keyframe:", keyframe_id, "node id:", n_id)

            # Check if the graph contains and edges. If not, the graph will be removed.
            if len(edges) == 0:
                print("[SECParser] NO edge found. Need at least 1 edge for GraphNet. Added dummy value")
                #graph.add_edge(0,0, features=cfg_relations[spatial_map.index("dummy_value")])
                keep_graph = False

            # Adds the edge relationships between nodes.
            for edge in edges:
                target = edge.attrib['target']
                relation = (edge.attrib['relation']).lower()

                # decide if noconneciton should be a edge or not.
                if relation == 'noconnection' and self.all_edges:
                    continue
                source = edge.attrib['source']

                # Nodes that shall be removed should not have any edges. This prevents uncessary edges.
                if target in skip_nodes:
                    continue
                if source in skip_nodes:
                    continue


                graph.add_edge(nodes_map[target], nodes_map[source], edge_attr=cfg._relations[cfg.spatial_map.index(relation)])

            _check_graph_remapping_nodes = check_graph(graph)
            # Check that node mapping is in the right way.
            if _check_graph_remapping_nodes:
                graph = nx.relabel_nodes(graph, _check_graph_remapping_nodes)

            # Adds the graph into a dict for the action id and corresponding sequence id.
            # Example: HIDING_SECs\Hiding_01\GraphML.xml
            # graph_dict['hiding'][1][...] will have the current graphs.
            if self.graph_dict.get(action_id) is None:
                if keep_graph:
                    self.graph_dict[action_id] = {}
                    self.graph_dict[action_id][self.seq] = [graph]
                else:
                    keep_graph = True
            else:
                if self.graph_dict[action_id].get(self.seq) is None:
                    if keep_graph:
                        self.graph_dict[action_id][self.seq] = [graph]
                    else:
                        keep_graph = True
                else:
                    if keep_graph:
                        self.graph_dict[action_id][self.seq].append(graph)
                    else:
                        print("[SECParser] Did not not add graph to list, due to 1 node. GraphNet requires at least 2 nodes.")
                        keep_graph = True

    def get_action(self, action):
        return self.actions[action]

    def get_keyframes_as_list(self):
        return self.graph_dict

    def get_graph_dict(self):
        return self.graph_dict


def _check_key(node, key):
    return node != key

'''
    Method to check if graph have data features and with creates node id to be in right order.
    Usually only called internally from SECParser.
'''
def check_graph(graph_nx):
    map_dict = {}
    for node_i, (key, data) in enumerate(graph_nx.nodes(data=True)):
        if _check_key(node_i, key) and data['x'] is not None:
            map_dict[key] = node_i

    if len(map_dict) > 0:
        return map_dict

'''
 Method to create a big list of the action dict.

 Input: dict with action, seq and graphs.

'''
def create_big_list(input_dict):
    big_list = []
    manipulation_actions = list(input_dict.keys())

    for action in sorted(manipulation_actions):
        variations_of_manipulation = sorted(input_dict[action].keys())

        for variation in variations_of_manipulation:
            num_of_graphs = len(input_dict[action][variation])

            for graph in range(num_of_graphs):
                big_list.append(input_dict[action][variation][graph])

    return big_list

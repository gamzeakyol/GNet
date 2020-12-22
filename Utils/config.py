import Utils.utils as util
from easydict import EasyDict as edict

"""
    SETTINGS FOR MANIAC DATASET

"""

def get_maniac_cfg():

    cfg = edict()

    cfg.epochs      = 500

    cfg.action_map  = ["chopping", "cutting", "hiding", "pushing", "putontop", "stirring", "takedown", "uncover"]
    cfg.spatial_map = ["noconnection", "temporal", "touching"]
    cfg.objects     = ['Apple', 'Arm', 'Ball', 'Banana', 'Body', 'Bowl', 'Box', 'Bread',
                       'Carrot', 'Chopper', 'Cucumber', 'Cup', 'Hand',
                       'Knife', 'Liquid', 'Pepper', 'Plate', 'Sausage', 'Slice', 'Spoon', 'null']

    # Creates one hot encodings for actions, relations and objects.
    cfg._relations = util.one_hot_string(cfg.spatial_map).tolist()
    cfg._actions = util.one_hot_string(cfg.action_map).tolist()
    cfg._objects = util.one_hot_string(cfg.objects).tolist()

    # Time window
    cfg.time_window = 4
    cfg.temporal_graphs = True

    cfg.skip_connections = False
    cfg.summery_writer = False

    # MODEL CONFIG
    cfg.batch_size = 32
    cfg.learning_rate = 0.01
    cfg.dropout = 0.2

    # GCL channels size
    cfg.channels = 64
    # Decoder input size
    cfg.decoder_in = 64

    return cfg

"""
    SETTINGS FOR BIMANUAL ACTION DATASET

"""
def get_bac_cfg():

    cfg = edict()

    cfg.epochs      = 500

    cfg.action_map  = ['approach', 'cut', 'drink', 'hammer', 'hold', 'idle', 'lift', 'place', 'pour', 'retreat',
                       'saw', 'screw', 'stir', 'wipe']
    cfg.spatial_map = ['above', 'behind of', 'below', 'contact', 'fixed moving together', 'getting close',
                       'halting together', 'in front of', 'inside', 'left of', 'moving apart', 'moving together',
                       'right of', 'stable', 'surround', 'temporal']
    cfg.objects     = ['LeftHand', 'RightHand', 'banana', 'bottle', 'bowl', 'cereals', 'cup', 'cuttingboard',
                       'hammer', 'harddrive', 'knife', 'saw', 'screwdriver', 'sponge', 'whisk', 'woodenwedge']

    cfg.original_index = ['idle', 'approach', 'retreat', 'lift', 'place', 'hold', 'pour', 'cut',
                          'hammer', 'saw', 'stir', 'screw', 'drink', 'wipe']


    # Creates one hot encodings for actions, relations and objects.
    cfg._relations = util.one_hot_string(cfg.spatial_map).tolist()
    cfg._actions = util.one_hot_string(cfg.action_map).tolist()
    cfg._objects = util.one_hot_string(cfg.objects).tolist()

    # Time window
    cfg.time_window = 8
    cfg.temporal_graphs = True
    cfg.seperate_samples = False

    cfg.summery_writer = False

    # MODEL CONFIG
    cfg.batch_size = 64
    cfg.learning_rate = 0.001
    cfg.dropout = 0.2

    # GCL channels size
    cfg.channels = 64
    # Decoder input size
    cfg.decoder_in = 32

    return cfg

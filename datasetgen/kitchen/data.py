''' Generate pytorch_geometric data for a kitchen scene and spatial expressions
'''
import os,sys; sys.path.insert(0, os.path.abspath('.'))

import torch 
from torch.nn.functional import one_hot
from torch_geometric.utils import grid, remove_self_loops
from torch_geometric.data import Data, Batch
import random
import json

from nlp.InferSent.models import InferSent
from .scene import SceneObject, Scene
from .cfggenerator import CFGGenerator, ProductionRule, Selection

class Config():
    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512

    MIN_OBJECTS = 6
    MAX_OBJECTS = 10
    MAX_CLASSES = 5
    MAX_QUERY_WORDS = 20
    QUERY_WORD_VSIZE = 300
    QUERY_ENCODING_VSIZE = 256
    CONV_SIZE = 128
    OBJECT_PLACEMENT_RETRYS = 5
    
    DROPOUT = .2
    LR = 0.002
    BATCH_SIZE = 256
    L2_LAMBDA = 0.01

    WORD_EMBEDDING_DIR = "./spatial_vqs"
    DATASET_PATH = "./output"
    SCENE_PREFIX = "img_test_"
    TEMPLATE_PATH = "./datasetgen/kitchen/object_templates"
    DEBUG = False
    RANDOM_SCALE = False
    RANDOM_SCALE_FACTOR = 0.2
    SCALE_BASE = 1.3

    NUM_GPUS = 1
    GENERATOR_BATCH_SIZE = 32

    class_indexes = {
        "fork" : 0,
        "spoon" : 1,
        "knife" : 2,
        "cup" : 3,
        "plate" : 4
    }        
    prod_rule_path = 'datasetgen/kitchen/production_rules_singular.json'
    NEAR_DISTANCE = 50

    V = 2
    MODEL_PATH = 'nlp/encoder/infersent{}.pkl'.format(V)
    params_model = {
        'bsize': 32, 
        'word_emb_dim': 300, 
        'enc_lstm_dim': 2048,
        'pool_type': 'max', 
        'dpout_model': 0.0, 
        'version': V
    }
    W2V_PATH = 'nlp/fastText/crawl-300d-2M.vec'

    def __init__(self):
        with open(self.prod_rule_path, 'r') as prfile:
            self.production_rules = json.load(fp=prfile, object_hook=ProductionRule.from_dict) 
        self.vocab = []
        for pr in self.production_rules.values():
            for ex in pr.expansion:
                for s in ex:
                    if isinstance(s, str) and (s[0] != '_'):
                        self.vocab.append(s)

def generate_scene(config):
    retrys = 10
    scene = None
    while (retrys >= 0) and (scene is None):
        retrys -= 1
        scene = Scene((config.IMAGE_WIDTH, config.IMAGE_HEIGHT), placement_retrys=config.OBJECT_PLACEMENT_RETRYS)
        num_objects = random.randint(config.MIN_OBJECTS, config.MAX_OBJECTS)
        
        for _ in range(num_objects):
            scale = 1.
            if config.RANDOM_SCALE:
                scale = config.SCALE_BASE + (random.random() * config.RANDOM_SCALE_FACTOR * 2.) - config.RANDOM_SCALE_FACTOR
            else:
                scale = 1.
            so = scene.add_object(rotation=None, scale=scale)
            if so is None:
                scene = None
                break

    if scene is None:
        raise ValueError("Couldn't place an object (scenes are too crowded?)")
    return scene

def fully_connect(node_count):
    edge_index = grid(node_count,node_count)[1].t().contiguous().long()
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index

def extract_scene_graph(scene, config, norm_boxes=False):
    # extract node features
    node_count = len(scene.mobile_objects)
    nodes = torch.zeros((node_count, 4 + config.MAX_CLASSES), dtype=torch.float)
    for i, so in enumerate(scene.mobile_objects):
        # bounding boxes
        nodes[i, 0:4] = torch.tensor(so.extents, dtype=torch.float) 
        # classification
        nodes[i, 4+config.class_indexes[so.scene_object_type]] = 1.

    if norm_boxes:
        nodes[i,0] /= config.IMAGE_WIDTH
        nodes[i,1] /= config.IMAGE_HEIGHT
        nodes[i,2] /= config.IMAGE_WIDTH
        nodes[i,3] /= config.IMAGE_HEIGHT

    # get fully connected edges
    edge_index = fully_connect(node_count)
    edge_attr = torch.zeros((edge_index.shape[1], 1), dtype=torch.float)
    
    return nodes, edge_index, edge_attr



def generate_examples(config, scene, max_instances=None, tag_filter=None):
    gen = CFGGenerator(scene, config.production_rules)
    selections = gen.expand_symbol('_Select')
    
    # if a filter is set, only include selections with all those tags
    if (tag_filter is not None):
        selections = [s for s in selections if bool(set(s.tags).intersection(set(tag_filter)))]
    random.shuffle(selections)

    # dont't return more than we need for the batch
    retained_selections = []
    if max_instances is None:
        retained_selections = selections
    else:
        retained_selections = selections[:max_instances]

    examples = []
    nodes, edge_index, edge_attr = extract_scene_graph(scene, config)
    for selection in retained_selections:
        targets = torch.sum(torch.eye(nodes.shape[0])[selection.scene_object_indexs], dim=0).float()
        example = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=targets)
        example.selection = selection.text
        examples.append(example)

    return examples

def gen_data_batch(config, encode_fn, batch_sz=32):
    # genrate examples
    examples = []
    while len(examples) < batch_sz:
        scene = generate_scene(config)
        examples += generate_examples(config, scene, max_instances=batch_sz-len(examples))

    # encode sentences
    encoded_selections = encode_fn([ex.selection for ex in examples])
    encoded_selections = torch.tensor(encoded_selections)
    encoded_selections = torch.unsqueeze(encoded_selections, dim=1)
    for i in range(encoded_selections.shape[0]):
        examples[i].selection = encoded_selections[i]

    # create a batch
    return Batch.from_data_list(examples)
    
def gen_scene_example(config, scene, encode_fn):
    # genrate examples
    examples = generate_examples(config, scene, max_instances=1)
    selection_text = examples[0].selection

    # encode sentences
    encoded_selections = encode_fn([ex.selection for ex in examples])
    encoded_selections = torch.tensor(encoded_selections)
    encoded_selections = torch.unsqueeze(encoded_selections, dim=1)
    for i in range(encoded_selections.shape[0]):
        examples[i].selection = encoded_selections[i]

    # create a batch
    return Batch.from_data_list(examples), selection_text

''' Example data batch generation
config = Config()
SceneObject.class_init(config)

infersent = InferSent(config.params_model)
infersent.load_state_dict(torch.load(config.MODEL_PATH))
infersent.set_w2v_path(config.W2V_PATH)
infersent.build_vocab(config.vocab, tokenize=True)

gen_data_batch(config=config, encode_fn=infersent.encode)
'''

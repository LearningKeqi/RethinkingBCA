from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .Mixed_model import mixed_model
from .braingnn_orig import BrainGNN_orig
from .mlp_graph import MLP_Graph
from .mlp_node import MLP_Node
from .neurograph import Neurograph
from .braingb import BrainGB


def model_factory(config: DictConfig):

    return eval(config.model.name)(config).cuda()

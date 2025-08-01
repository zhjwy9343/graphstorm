"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Builtin configs
"""
import dataclasses
import typing
import hashlib
from typing import List

BUILTIN_GNN_ENCODER = ["gat", "rgat", "rgcn", "sage", "hgt", "gatv2"]
BUILTIN_INPUT_ONLY_ENCODER = ["lm", "mlp", "learnable_embed"]
BUILTIN_ENCODER = BUILTIN_INPUT_ONLY_ENCODER + BUILTIN_GNN_ENCODER
SUPPORTED_BACKEND = ["gloo", "nccl"]
BUILTIN_EDGE_FEAT_MP_OPS = ["concat", "add", "sub", "mul", "div"]

GRAPHSTORM_MODEL_EMBED_LAYER = "embed"
GRAPHSTORM_MODEL_DENSE_EMBED_LAYER = "dense_embed"
GRAPHSTORM_MODEL_SPARSE_EMBED_LAYER = "sparse_embed"
GRAPHSTORM_MODEL_GNN_LAYER = "gnn"
GRAPHSTORM_MODEL_DECODER_LAYER = "decoder"
GRAPHSTORM_MODEL_ALL_LAYERS = [GRAPHSTORM_MODEL_EMBED_LAYER,
                               GRAPHSTORM_MODEL_GNN_LAYER,
                               GRAPHSTORM_MODEL_DECODER_LAYER]
GRAPHSTORM_MODEL_LAYER_OPTIONS = GRAPHSTORM_MODEL_ALL_LAYERS + \
        [GRAPHSTORM_MODEL_DENSE_EMBED_LAYER,
         GRAPHSTORM_MODEL_SPARSE_EMBED_LAYER]

BUILTIN_CLASS_LOSS_CROSS_ENTROPY = "cross_entropy"
BUILTIN_CLASS_LOSS_FOCAL = "focal"
BUILTIN_CLASS_LOSS_FUNCTION = [BUILTIN_CLASS_LOSS_CROSS_ENTROPY, BUILTIN_CLASS_LOSS_FOCAL]

BUILTIN_REGRESSION_LOSS_MSE = "mse"
BUILTIN_REGRESSION_LOSS_SHRINKAGE = "shrinkage"
BUILTIN_REGRESSION_LOSS_FUNCTION = [BUILTIN_REGRESSION_LOSS_MSE,
                                    BUILTIN_REGRESSION_LOSS_SHRINKAGE]

BUILTIN_LP_LOSS_CROSS_ENTROPY = "cross_entropy"
BUILTIN_LP_LOSS_LOGSIGMOID_RANKING = "logsigmoid"
BUILTIN_LP_LOSS_CONTRASTIVELOSS = "contrastive"
BUILTIN_LP_LOSS_BPR = "bpr"
BUILTIN_LP_LOSS_FUNCTION = [BUILTIN_LP_LOSS_CROSS_ENTROPY, \
    BUILTIN_LP_LOSS_LOGSIGMOID_RANKING, BUILTIN_LP_LOSS_CONTRASTIVELOSS,
    BUILTIN_LP_LOSS_BPR]

GRAPHSTORM_LP_EMB_L2_NORMALIZATION = "l2_norm"
GRAPHSTORM_LP_EMB_NORMALIZATION_METHODS = [GRAPHSTORM_LP_EMB_L2_NORMALIZATION]

BUILDIN_GNN_BATCH_NORM = 'batch'
BUILDIN_GNN_LAYER_NORM = 'layer'
BUILTIN_GNN_NORM = [BUILDIN_GNN_BATCH_NORM, BUILDIN_GNN_LAYER_NORM]

BUILTIN_TASK_NODE_CLASSIFICATION = "node_classification"
BUILTIN_TASK_NODE_REGRESSION = "node_regression"
BUILTIN_TASK_EDGE_CLASSIFICATION = "edge_classification"
BUILTIN_TASK_EDGE_REGRESSION = "edge_regression"
BUILTIN_TASK_LINK_PREDICTION = "link_prediction"
BUILTIN_TASK_COMPUTE_EMB = "compute_emb"
BUILTIN_TASK_RECONSTRUCT_NODE_FEAT = "reconstruct_node_feat"
BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT = "reconstruct_edge_feat"
BUILTIN_TASK_MULTI_TASK = "multi_task"

LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL = "ALL"

SUPPORTED_TASKS  = [BUILTIN_TASK_NODE_CLASSIFICATION, \
    BUILTIN_TASK_NODE_REGRESSION, \
    BUILTIN_TASK_EDGE_CLASSIFICATION, \
    BUILTIN_TASK_LINK_PREDICTION, \
    BUILTIN_TASK_EDGE_REGRESSION, \
    BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
    BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT]

EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY = "consecutive_increase"
EARLY_STOP_AVERAGE_INCREASE_STRATEGY = "average_increase"

# Task tracker
GRAPHSTORM_SAGEMAKER_TASK_TRACKER = "sagemaker_task_tracker"
GRAPHSTORM_TENSORBOARD_TASK_TRACKER = "tensorboard_task_tracker"

SUPPORTED_TASK_TRACKER = [GRAPHSTORM_SAGEMAKER_TASK_TRACKER,
                          GRAPHSTORM_TENSORBOARD_TASK_TRACKER]

# Link prediction decoder
BUILTIN_LP_DOT_DECODER = "dot_product"
BUILTIN_LP_DISTMULT_DECODER = "distmult"
BUILTIN_LP_ROTATE_DECODER = "rotate"
BUILTIN_LP_TRANSE_L1_DECODER = "transe_l1"
BUILTIN_LP_TRANSE_L2_DECODER = "transe_l2"

SUPPORTED_LP_DECODER = [BUILTIN_LP_DOT_DECODER,
                        BUILTIN_LP_DISTMULT_DECODER,
                        BUILTIN_LP_ROTATE_DECODER,
                        BUILTIN_LP_TRANSE_L1_DECODER,
                        BUILTIN_LP_TRANSE_L2_DECODER]

# Filename constants

# Filename for GS training yaml, updated with runtime args
GS_RUNTIME_TRAINING_CONFIG_FILENAME = "GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml"

# Filename for output GConstruct graph data configuration, updated with data-derived transformations
GS_RUNTIME_GCONSTRUCT_FILENAME = "data_transform_new.json"

################ Task info data classes ############################
def get_mttask_id(task_type, ntype=None, etype=None, label=None):
    """ Generate task ID for multi-task learning tasks.
        The ID is composed of the task type, the node type
        or the edge type(s) and the label of a task.

    Parameters
    ----------
    task_type: str
        Task type.
    ntype: str
        Node type.
    etype: str, tuple or list of tuple
        Edge type. It can be "ALL_ETYPE" meaning all the etypes.
        It can be a tuple representing an edge type.
        It can be a list of tuples representing a list of edge types.
    label: str
        Label name.

    Return
    ------
    str: Task ID.
    """
    task_id = [task_type]
    if ntype is not None:
        task_id.append(ntype) # node task
    if etype is not None:
        if isinstance(etype, str):
            task_id.append(etype)
        elif isinstance(etype, tuple):
            task_id.append("_".join(etype))
        elif isinstance(etype, list): # a list of etypes
            etype_info = "__".join(["_".join(et) for et in etype])
            # In case the task id is too long, trim it
            # Set the max etype information into 64
            # Add a hash information to avoid task id naming conflict
            if len(etype_info) > 64:
                hasher = hashlib.sha256()
                hasher.update(etype_info.encode('utf-8'))
                id_hash = hasher.hexdigest()
                etype_info = etype_info[:64] + id_hash[:8]

            task_id.append(etype_info)
        else:
            raise TypeError(f"Unknown etype format: {etype}. Must be a string " \
                            "or a tuple of strings or a list of tuples of strings.")
    if label is not None:
        task_id.append(label)

    return "-".join(task_id)

@dataclasses.dataclass
class TaskInfo:
    """Information of a training task in multi-task learning

    Parameters
    ----------
    task_type: str
        Task type.
    task_id: str
        Task id. Unique id for each task.
    batch_size: int
        Batch size of the current task.
    mask_fields: list
        Train/validation/test mask fields.
    dataloader:
        Task dataloader.
    eval_metric: list
        Evaluation metrics
    task_weight: float
        Weight of the task in final loss.
    """
    task_type : str
    task_id : str
    task_config : typing.Any = None
    dataloader : typing.Any = None # dataloder

@dataclasses.dataclass
class FeatureGroup:
    """ Feature names of groups of features.

        Users can define multiple group of node features that
        GraphStorm will use different encoders to encode them

    .. versionadded:: 0.5.0
        Since 0.5.0, GraphStorm supports using different encoders
        to encode different input node features of the same node.

    Parameters
    feature_group: list of strings
        Feature group
    """
    feature_group: List[str]

@dataclasses.dataclass
class FeatureGroupSize:
    """ Feature sizes of groups of features.

    .. versionadded:: 0.5.0
        Since 0.5.0, GraphStorm supports using different encoders
        to encode different input node features of the same node.

    Parameters
    feature_group_sizes: list of int
        Feature group sizes
    """
    feature_group_sizes: List[int]

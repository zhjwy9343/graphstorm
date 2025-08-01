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
"""
import json
import math
import os
import shutil
import sys
import tempfile
import yaml
import pytest
import hashlib
from argparse import Namespace
from pathlib import Path

import dgl
import torch as th
from dgl.distributed.constants import DEFAULT_NTYPE, DEFAULT_ETYPE
import pytest

from graphstorm.config import GSConfig, FeatureGroup
from graphstorm.config.config import (BUILTIN_CLASS_LOSS_CROSS_ENTROPY,
                                      BUILTIN_CLASS_LOSS_FOCAL,
                                      BUILTIN_LP_LOSS_CROSS_ENTROPY,
                                      BUILTIN_LP_LOSS_LOGSIGMOID_RANKING,
                                      BUILTIN_LP_LOSS_CONTRASTIVELOSS,
                                      BUILTIN_LP_LOSS_BPR,
                                      BUILTIN_REGRESSION_LOSS_MSE,
                                      BUILTIN_REGRESSION_LOSS_SHRINKAGE,
                                      GS_RUNTIME_TRAINING_CONFIG_FILENAME,
                                      GS_RUNTIME_GCONSTRUCT_FILENAME,)
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION,
                               BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_EDGE_REGRESSION,
                               BUILTIN_TASK_LINK_PREDICTION,
                               BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                               BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT)
from graphstorm.config.config import GRAPHSTORM_LP_EMB_L2_NORMALIZATION
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from graphstorm.config.config import (GRAPHSTORM_SAGEMAKER_TASK_TRACKER,
                                      GRAPHSTORM_TENSORBOARD_TASK_TRACKER)
from graphstorm.config import (BUILTIN_LP_DOT_DECODER,
                               BUILTIN_LP_DISTMULT_DECODER,
                               BUILTIN_LP_ROTATE_DECODER,
                               BUILTIN_LP_TRANSE_L1_DECODER,
                               BUILTIN_LP_TRANSE_L2_DECODER)
from graphstorm.config.config import LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL
from config_utils import create_dummy_config_obj, create_basic_config
from graphstorm.config.config import get_mttask_id

# Get location of test file
_ROOT = os.path.abspath(os.path.dirname(__file__))

def test_get_mttask_id():
    # node classification task
    task_type = "node_classification"
    ntype = "type0"
    etype = None
    label = "label"

    task_id = get_mttask_id(task_type, ntype=ntype, etype=etype, label=label)
    assert task_id == "-".join([task_type, ntype, label])

    # edge classification task
    task_type = "edge_classification"
    ntype = None
    etype = ("type0", "r0", "type1")
    label = "label"
    task_id = get_mttask_id(task_type, ntype=ntype, etype=etype, label=label)
    assert task_id == "-".join([task_type, "_".join(etype), label])

    # link prediction task
    task_type = "link_prediction"
    ntype = None
    etype = [("type0", "r0", "type1")]
    label = None
    task_id = get_mttask_id(task_type, ntype=ntype, etype=etype, label=label)
    etype_info = "__".join(["_".join(et) for et in etype])
    assert task_id == "-".join([task_type, etype_info])

    task_type = "link_prediction"
    ntype = None
    etype = [("type0", "r0", "type1"), ("type0", "r0", "type2")]
    task_id = get_mttask_id(task_type, ntype=ntype, etype=etype, label=label)
    etype_info = "__".join(["_".join(et) for et in etype])
    assert task_id == "-".join([task_type, etype_info])

    # the etypes are too long
    task_type = "link_prediction"
    ntype = None
    etype = [("type0", "r0", "type1"), ("type0", "1"*64, "type2")]
    hasher = hashlib.sha256()
    etype_info = "__".join(["_".join(et) for et in etype])
    hasher.update(etype_info.encode('utf-8'))
    task_id = get_mttask_id(task_type, ntype=ntype, etype=etype, label=label)
    etype_info = etype_info[:64] + hasher.hexdigest()[:8]
    assert task_id == "-".join([task_type, etype_info])

def check_failure(config, field):
    has_error = False
    try:
        dummy = getattr(config, field)
    except:
        has_error = True
    assert has_error



def copy_gconstruct_config(tmp_path, file_name=GS_RUNTIME_GCONSTRUCT_FILENAME):
    """Copy a GConstruct config file to the given path/filename"""
    ML_GCONSTRUCT_FILEPATH = os.path.join(
        _ROOT, "../end2end-tests/data_gen/movielens.json")
    shutil.copy2(
        ML_GCONSTRUCT_FILEPATH,
        os.path.join(tmp_path, file_name)
    )


def test_load_basic_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_basic_config(Path(tmpdirname), 'basic_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'basic_test.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        # success load
        assert config.backend == "gloo"
        assert config.ip_config == os.path.join(Path(tmpdirname), "ip.txt")
        assert config.part_config == os.path.join(Path(tmpdirname), "part.json")
        assert config.verbose == False
        assert config.eval_frequency == 100
        assert config.no_validation == True

        # Change config's variables to do further testing
        config._backend = "nccl"
        assert config.backend == "nccl"
        config._model_encoder_type = "lm"
        assert config.model_encoder_type == "lm"

        # Check default values
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'basic_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.backend == "gloo"
        assert config.eval_frequency == sys.maxsize
        assert config.no_validation == False
        check_failure(config, "model_encoder_type") # must provide model_encoder_type

        # Check failures
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'basic_test_fail.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "backend")
        check_failure(config, "part_config")
        check_failure(config, "eval_frequency")
        check_failure(config, "model_encoder_type")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'basic_test_fail2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "ip_config")
        check_failure(config, "part_config")

def create_task_tracker_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["output"] = {
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["output"] = {
        "task_tracker": "sagemaker_task_tracker",
        "log_report_frequency": 100,
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"0.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["output"] = {
        "task_tracker": "tensorboard_task_tracker",
        "log_report_frequency": 100,
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["output"] = {
        "task_tracker": "tensorboard_task_tracker:./log/",
        "log_report_frequency": 100,
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["output"] = {
        "task_tracker": "mlflow",
        "log_report_frequency": 0,
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_task_tracker_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_task_tracker_config(Path(tmpdirname), 'task_tracker_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'task_tracker_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.task_tracker == GRAPHSTORM_SAGEMAKER_TASK_TRACKER
        assert config.task_tracker_logpath == None
        assert config.log_report_frequency == 1000

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'task_tracker_test0.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.task_tracker == GRAPHSTORM_SAGEMAKER_TASK_TRACKER
        assert config.task_tracker_logpath == None
        assert config.log_report_frequency == 100

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'task_tracker_test1.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.task_tracker == GRAPHSTORM_TENSORBOARD_TASK_TRACKER
        assert config.task_tracker_logpath == None
        assert config.log_report_frequency == 100

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'task_tracker_test2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.task_tracker == GRAPHSTORM_TENSORBOARD_TASK_TRACKER
        assert config.task_tracker_logpath == "./log/"
        assert config.log_report_frequency == 100

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'task_tracker_test_fail.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "task_tracker")
        check_failure(config, "log_report_frequency")

def create_train_config(tmp_path: Path, file_name: str):
    yaml_object = create_dummy_config_obj()
    _, part_config = tempfile.mkstemp(dir=str(tmp_path))
    yaml_object["gsf"]["basic"].update({
        "part_config": part_config
    })
    yaml_object["gsf"]["hyperparam"] = {
    }


    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # normal value
    yaml_object["gsf"]["hyperparam"] = {
        "dropout" : 0.1,
        "lr": 0.001,
        "num_epochs": 10,
        "batch_size": 64,
        "eval_batch_size": 128,
        "wd_l2norm": 0.1,
        "alpha_l2norm": 0.00001,
        "eval_frequency": 1000,
        'save_model_frequency': 1000,
        "topk_model_to_save": 3,
        "lm_tune_lr": 0.0001,
        "sparse_optimizer_lr": 0.001,
        "use_node_embeddings": False,
        "use_self_loop": False,
        "use_early_stop": True,
        "save_model_path": os.path.join(tmp_path, "save"),
    }

    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["hyperparam"] = {
        "topk_model_to_save": 4,
        "save_model_path": os.path.join(tmp_path, "save"),
        "wd_l2norm": 5e-5,
        "alpha_l2norm": 5e-5,
    }
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["hyperparam"] = {
        "eval_frequency": 1000,
        'save_model_frequency': 2000,
        "topk_model_to_save": 5,
        "save_model_path": os.path.join(tmp_path, "save"),
        "wd_l2norm": "1e-3",
        "alpha_l2norm": "1e-3",
    }
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # eval_frequency = 1000 and save_model_frequency uses default (-1)
    yaml_object["gsf"]["hyperparam"] = {
        "eval_frequency": 1000,
        "topk_model_to_save": 5,
        "save_model_path": os.path.join(tmp_path, "save"),
    }
    with open(os.path.join(tmp_path, file_name+"3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # for failures
    yaml_object["gsf"]["hyperparam"] = {
        "dropout" : -1.0,
        "lr": 0.,
        "num_epochs": -1,
        "batch_size": 0,
        "eval_batch_size": 0,
        "lm_tune_lr": 0.,
        "sparse_optimizer_lr": 0.,
        "use_node_embeddings": True,
        "use_self_loop": "error",
        "eval_frequency": 1000,
        'save_model_frequency': 700,
        "topk_model_to_save": 3,
        "use_early_stop": True,
        "early_stop_burnin_rounds": -1,
        "early_stop_rounds": 0,
        "wd_l2norm": "NA",
        "alpha_l2norm": "NA",
    }

    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["hyperparam"] = {
        "eval_frequency": 1100,
        'save_model_frequency': 2000,
        "topk_model_to_save": 3,
        "save_model_path": os.path.join(tmp_path, "save"),
        "wd_l2norm": "",
        "alpha_l2norm": "",
    }
    with open(os.path.join(tmp_path, file_name+"_fail1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_train_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_train_config(Path(tmpdirname), 'train_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'train_test_default.yaml'), local_rank=0)
        config = GSConfig(args)

        assert config.dropout == 0
        check_failure(config, "lr")
        assert config.num_epochs == 0
        check_failure(config, "batch_size")
        config._batch_size = 32
        assert config.batch_size == 32
        assert config.eval_batch_size == 10000
        assert config.wd_l2norm == 0
        assert config.alpha_l2norm == 0
        assert config.topk_model_to_save == math.inf
        config._lr = 0.01
        assert config.lm_tune_lr == 0.01
        assert config.sparse_optimizer_lr == 0.01
        assert config.use_node_embeddings == False
        assert config.use_self_loop == True
        assert config.use_early_stop == False

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'train_test.yaml'),
            local_rank=0)
        config = GSConfig(args)

        assert config.dropout == 0.1
        assert config.lr == 0.001
        assert config.num_epochs == 10
        assert config.batch_size == 64
        assert config.eval_batch_size == 128
        assert config.wd_l2norm == 0.1
        assert config.alpha_l2norm == 0.00001
        assert config.topk_model_to_save == 3
        assert config.lm_tune_lr == 0.0001
        assert config.sparse_optimizer_lr == 0.001
        assert config.use_node_embeddings == False
        assert config.use_self_loop == False
        assert config.use_early_stop == True
        assert config.early_stop_burnin_rounds == 0
        assert config.early_stop_rounds == 3

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'train_test1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.topk_model_to_save == 4
        assert config.wd_l2norm == 5e-5
        assert config.alpha_l2norm == 5e-5

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'train_test2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.eval_frequency == 1000
        assert config.save_model_frequency == 2000
        assert config.topk_model_to_save == 5
        assert config.wd_l2norm == 1e-3
        assert config.alpha_l2norm == 1e-3

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'train_test3.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.eval_frequency == 1000
        assert config.save_model_frequency == -1
        assert config.topk_model_to_save == 5

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'train_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "dropout")
        check_failure(config, "lr")
        check_failure(config, "num_epochs")
        check_failure(config, "batch_size")
        check_failure(config, "eval_batch_size")
        check_failure(config, "lm_tune_lr")
        check_failure(config, "sparse_optimizer_lr")
        assert config.use_node_embeddings == True
        check_failure(config, "use_self_loop")
        config._dropout = 1.0
        check_failure(config, "dropout")
        assert config.use_early_stop == True
        check_failure(config, "topk_model_to_save")
        check_failure(config, "early_stop_burnin_rounds")
        check_failure(config, "early_stop_rounds")
        check_failure(config, "wd_l2norm")
        check_failure(config, "alpha_l2norm")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'train_test_fail1.yaml'), local_rank=0)
        config = GSConfig(args)
        # in PR # 893 we loose the constraints of model saving frequency and eval frequency
        # so here we do not check failure, but check the topk model argument
        assert config.topk_model_to_save == 3
        check_failure(config, "wd_l2norm")
        check_failure(config, "alpha_l2norm")

def create_rgcn_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["rgcn"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgcn"] = {
        "num_bases": 2,
    }
    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgcn"] = {
        "num_bases": 0.1,
    }
    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgcn"] = {
        "num_bases": -2,
    }
    with open(os.path.join(tmp_path, file_name+"_fail2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_rgcn_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_rgcn_config(Path(tmpdirname), 'rgcn_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgcn_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_bases == -1

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgcn_test.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_bases == 2

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgcn_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "num_bases")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgcn_test_fail2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "num_bases")

def create_rgat_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["rgat"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgat"] = {
        "num_heads": 2,
    }
    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["rgat"] = {
        "num_heads": 0,
    }
    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_rgat_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_rgat_config(Path(tmpdirname), 'rgat_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgat_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_heads == 4

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgat_test.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_heads == 2

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'rgat_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "num_heads")

def create_node_class_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["node_classification"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "target_ntype": "a",
        "label_field": "label",
        "multilabel": True,
        "num_classes": 20,
        "class_loss_func": "cross_entropy"
    }
    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "target_ntype": "a",
        "label_field": "label",
        "multilabel": True,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2",
        "num_classes": 20,
        "class_loss_func": "focal"
    }
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": "F1_score",
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2",
    }
    with open(os.path.join(tmp_path, file_name+"_metric1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": ["F1_score", "precision_recall", "ROC_AUC", "hit_at_10", \
                        "precision", "recall", "fscore_at_2.5", \
                        "precision_at_recall_0.8", "recall_at_precision_0.8"],
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2",
    }
    with open(os.path.join(tmp_path, file_name+"_metric2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "multilabel": "error",
        "num_classes": 0,
        "class_loss_func": "unknown"
    }
    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 1,
        "class_loss_func": "focal"
    }
    with open(os.path.join(tmp_path, file_name+"_num_class_1_focal.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "num_classes": {
            "n1": 1,
            "n2": 2,
        },
        "class_loss_func": "focal"
    }
    with open(os.path.join(tmp_path, file_name+"_num_class_1_focal2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 1,
        "class_loss_func": "cross_entropy"
    }
    with open(os.path.join(tmp_path, file_name+"_num_class_1_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "num_classes": {
            "n1": 1,
            "n2": 2,
        },
        "class_loss_func": "cross_entropy"
    }
    with open(os.path.join(tmp_path, file_name+"_num_class_1_fail2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric and multi-label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2", # multilabel is not set to True
        "eval_metric": "unknown"
    }

    with open(os.path.join(tmp_path, file_name+"_fail_metric1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric hit_at_ten is an error.
    # should be hit_at_10
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": "hit_at_ten"
    }

    with open(os.path.join(tmp_path, file_name+"_fail_metric2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric hit_at_ten is an error.
    # should be hit_at_10
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": ["F1_score", "hit_at_ten"]
    }

    with open(os.path.join(tmp_path, file_name+"_fail_metric3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric fscore_at_ten is an error. Should be fscore_at_10
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": "fscore_at_ten"
    }

    with open(os.path.join(tmp_path, file_name+"_fail_metric4.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric fscore_at_2.0. is an error. Should be fscore_at_2.0
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": "fscore_at_2.0."
    }

    with open(os.path.join(tmp_path, file_name+"_fail_metric5.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric precision_at_recall_one is an error. Should be precision_at_recall_1
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 2,
        "eval_metric": "precision_at_recall_one"
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric6.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric precision_at_recall_0.5. is an error. Should be precision_at_recall_0.5
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 2,
        "eval_metric": "precision_at_recall_0.5."
    }
    with open(os.path.join(tmp_path, file_name + "_fail_metric7.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric precision_at_recall_0 is an error. Beta should be in (0, 1].
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 2,
        "eval_metric": "precision_at_recall_0"
    }
    with open(os.path.join(tmp_path, file_name + "_fail_metric8.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric precision_at_recall_2 is an error. Beta should be in (0, 1].
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 2,
        "eval_metric": "precision_at_recall_2"
    }
    with open(os.path.join(tmp_path, file_name + "_fail_metric9.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric recall_at_precision_one is an error. Should be recall_at_precision_1
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 2,
        "eval_metric": "recall_at_precision_one"
    }
    with open(os.path.join(tmp_path, file_name + "_fail_metric10.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric recall_at_precision_0.5. is an error. Should be recall_at_precision_0.5
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 2,
        "eval_metric": "recall_at_precision_0.5."
    }
    with open(os.path.join(tmp_path, file_name + "_fail_metric11.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric recall_at_precision_0 is an error. Beta should be in (0, 1]
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 2,
        "eval_metric": "recall_at_precision_0"
    }
    with open(os.path.join(tmp_path, file_name + "_fail_metric12.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval_metric recall_at_precision_2 is an error. Beta should be in (0, 1]
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 2,
        "eval_metric": "recall_at_precision_2"
    }
    with open(os.path.join(tmp_path, file_name + "_fail_metric13.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric and multi-label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": {}, # eval metric must be string or list
        "multilabel": False,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2", # Do not need multilabel_weights
    }

    with open(os.path.join(tmp_path, file_name+"_fail_ml_w1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test eval metric and multi-label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "eval_metric": ["F1_score", "unknown"], # one of metrics is not supported
        "multilabel": True,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2", # Does not match num_classes
    }

    with open(os.path.join(tmp_path, file_name+"_fail_ml_w2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test multi-label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "multilabel": True,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1,-0.1", # weight can not be negative
    }

    with open(os.path.join(tmp_path, file_name+"_fail_ml_w3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test return-proba
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "multilabel": True,
        "return_proba": True,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1,-0.1", # weight can not be negative
    }

    with open(os.path.join(tmp_path, file_name+"_fail_ml_w3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "multilabel": True,
        "return_proba": False,
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1,-0.1", # weight can not be negative
    }

    with open(os.path.join(tmp_path, file_name+"_fail_ml_w3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test imbalance label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1", # len(weight) != num_classes
    }

    with open(os.path.join(tmp_path, file_name+"_fail_imb_l_w1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test imbalance label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1,0", # weight must larger than 0
    }

    with open(os.path.join(tmp_path, file_name+"_fail_imb_l_w2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test imbalance label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "multilabel": True,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2", # len mismatch
    }

    with open(os.path.join(tmp_path, file_name+"_fail_imb_w3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test imbalance label
    yaml_object["gsf"]["node_classification"] = {
        "num_classes": 20,
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,0.1,0.2,0.3,0.1,abc123", # weights must be float
    }

    with open(os.path.join(tmp_path, file_name+"_fail_imb_l_w4.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_node_class_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_node_class_config(Path(tmpdirname), 'node_class_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_ntype == DEFAULT_NTYPE
        check_failure(config, "label_field")
        assert config.multilabel == False
        assert config.multilabel_weights == None
        assert config.imbalance_class_weights == None
        check_failure(config, "num_classes")
        assert config.class_loss_func == BUILTIN_CLASS_LOSS_CROSS_ENTROPY

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_ntype == "a"
        assert config.label_field == "label"
        assert config.multilabel == True
        assert config.multilabel_weights == None
        assert config.imbalance_class_weights == None
        assert config.num_classes == 20
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "accuracy"
        assert config.class_loss_func == BUILTIN_CLASS_LOSS_CROSS_ENTROPY

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.multilabel == True
        assert config.imbalance_class_weights.tolist() == [1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2]
        assert config.class_loss_func == BUILTIN_CLASS_LOSS_FOCAL

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_metric1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.eval_metric[0] == "f1_score"
        assert config.imbalance_class_weights.tolist() == [1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2]

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_metric2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert len(config.eval_metric) == 9
        assert config.eval_metric[0] == "f1_score"
        assert config.eval_metric[1] == "precision_recall"
        assert config.eval_metric[2] == "roc_auc"
        assert config.eval_metric[3] == "hit_at_10"
        assert config.eval_metric[7] == "precision_at_recall_0.8"
        assert config.eval_metric[8] == "recall_at_precision_0.8"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "multilabel")
        check_failure(config, "num_classes")
        check_failure(config, "eval_metric")
        check_failure(config, "class_loss_func")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_num_class_1_focal.yaml'), local_rank=0)
        config = GSConfig(args)
        with pytest.warns(DeprecationWarning) as record:
            assert config.num_classes == 1
            assert config.class_loss_func == BUILTIN_CLASS_LOSS_FOCAL

        # Verify that the warning was raised
        assert len(record) == 1
        # Verify the warning message
        assert "Allowing num_classes=1 with focal loss is deprecated" in str(record[0].message)

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_num_class_1_focal2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert len(config.num_classes) == 2
        assert config.num_classes["n2"] == 2
        assert config.class_loss_func == BUILTIN_CLASS_LOSS_FOCAL
        with pytest.warns(DeprecationWarning) as record:
            assert config.num_classes["n1"] == 1

        # Verify that the warning was raised
        assert len(record) == 1
        # Verify the warning message
        assert "Allowing num_classes=1 with focal loss is deprecated" in str(record[0].message)

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_num_class_1_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.class_loss_func == BUILTIN_CLASS_LOSS_CROSS_ENTROPY
        check_failure(config, "num_classes")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_num_class_1_fail2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.class_loss_func == BUILTIN_CLASS_LOSS_CROSS_ENTROPY
        check_failure(config, "num_classes")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")
        check_failure(config, "multilabel_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric3.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric4.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric5.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric6.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 2
        check_failure(config, "eval_metric")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric7.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 2
        check_failure(config, "eval_metric")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric8.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 2
        check_failure(config, "eval_metric")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric9.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 2
        check_failure(config, "eval_metric")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric10.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 2
        check_failure(config, "eval_metric")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric11.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 2
        check_failure(config, "eval_metric")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric12.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 2
        check_failure(config, "eval_metric")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_metric13.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 2
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_ml_w1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")
        assert config.multilabel == False
        check_failure(config, "multilabel_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_ml_w2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "eval_metric")
        assert config.multilabel == True
        check_failure(config, "multilabel_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_ml_w3.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        assert config.multilabel == True
        check_failure(config, "multilabel_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_imb_l_w1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "imbalance_class_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_imb_l_w2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "imbalance_class_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_imb_w3.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        assert config.multilabel == True
        check_failure(config, "imbalance_class_weights")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_class_test_fail_imb_l_w4.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.num_classes == 20
        check_failure(config, "imbalance_class_weights")

def create_node_regress_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["node_regression"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "target_ntype": "a",
        "label_field": "label",
        "eval_metric": "Mse",
        "regression_loss_func": "mse"
    }
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "target_ntype": "a",
        "label_field": "label",
        "eval_metric": ["mse", "RMSE"],
        "regression_loss_func": "shrinkage"
    }
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "eval_metric": "error",
        "regression_loss_func": "unknown"
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "eval_metric": ["MSE", "error"], # one of metrics is not supported
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_regression"] = {
        "eval_metric": {}, # eval metric must be string or list
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_node_regress_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_node_regress_config(Path(tmpdirname), 'node_regress_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_ntype == DEFAULT_NTYPE
        check_failure(config, "label_field")
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "rmse"
        assert config.regression_loss_func == BUILTIN_REGRESSION_LOSS_MSE

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_ntype == "a"
        assert config.label_field == "label"
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "mse"
        assert config.regression_loss_func == BUILTIN_REGRESSION_LOSS_MSE

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert len(config.eval_metric) == 2
        assert config.eval_metric[0] == "mse"
        assert config.eval_metric[1] == "rmse"
        assert config.regression_loss_func == BUILTIN_REGRESSION_LOSS_SHRINKAGE

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test_fail_metric1.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")
        check_failure(config, "regression_loss_func")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test_fail_metric2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_regress_test_fail_metric3.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

def create_edge_class_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["edge_classification"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["edge_classification"] = {
        "target_etype": ["query,match,asin"],
        "reverse_edge_types_map": [],
        "label_field": "label",
        "multilabel": True,
        "num_classes": 4,
        "num_decoder_basis": 4,
        "remove_target_edge_type": False,
        "decoder_type": "MLPDecoder",
        "decoder_edge_feat": ["feat"],
        "class_loss_func": "cross_entropy"
    }

    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["edge_classification"] = {
        "target_etype": ["query,match,asin", "query,click,asin"],
        "reverse_edge_types_map": ["query,match,rev-match,asin", "query,click,rev-click,asin"],
        "num_classes": 4,
        "eval_metric": ["Per_class_f1_score", "Precision_Recall", "hit_at_20"],
        "decoder_edge_feat": ["query,match,asin:feat0,feat1"],
        "class_loss_func": "focal"
    }

    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # test failure
    yaml_object["gsf"]["edge_classification"] = {
        "target_etype": "query,match,asin",
        "reverse_edge_types_map": "query,match,rev-match,asin",
        "multilabel": "error",
        "num_classes": 0,
        "num_decoder_basis": 1,
        "remove_target_edge_type": "error",
        "decoder_edge_feat": ["query,no-match,asin:feat0,feat1"],
        "class_loss_func": "unknown"
    }

    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["edge_classification"] = {
        "target_etype": [],
        "num_classes": 4,
        "eval_metric": ["per_class_f1_score", "rmse"],
        "decoder_edge_feat": ["query,no-match,asin::feat0,feat1"]
    }
    with open(os.path.join(tmp_path, file_name+"_fail2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # eval metric hit_at_one will cause failure
    yaml_object["gsf"]["edge_classification"] = {
        "target_etype": ["query,match,asin"],
        "num_classes": 4,
        "eval_metric": ["hit_at_one", "rmse"]
    }
    with open(os.path.join(tmp_path, file_name+"_fail3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_edge_class_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_edge_class_config(Path(tmpdirname), 'edge_class_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_etype == [DEFAULT_ETYPE]
        assert config.decoder_type == "DenseBiDecoder"
        assert config.num_decoder_basis == 2
        assert config.remove_target_edge_type == True
        assert len(config.reverse_edge_types_map) == 0
        check_failure(config, "label_field")
        assert config.multilabel == False
        check_failure(config, "num_classes")
        assert config.class_loss_func == BUILTIN_CLASS_LOSS_CROSS_ENTROPY

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_etype[0] == ("query", "match", "asin")
        assert len(config.target_etype) == 1
        assert config.decoder_type == "MLPDecoder"
        assert config.remove_target_edge_type == False
        assert len(config.reverse_edge_types_map) == 0
        assert config.label_field == "label"
        assert config.multilabel == True
        assert config.num_classes == 4
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "accuracy"
        assert config.decoder_edge_feat == "feat"
        assert config.class_loss_func == BUILTIN_CLASS_LOSS_CROSS_ENTROPY

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_etype[0] == ("query", "match", "asin")
        assert config.target_etype[1] == ("query", "click", "asin")
        assert len(config.target_etype) == 2
        assert len(config.reverse_edge_types_map) == 2
        assert config.reverse_edge_types_map[("query","match","asin")] == \
             ("asin","rev-match","query")
        assert config.reverse_edge_types_map[("query","click","asin")] == \
             ("asin","rev-click","query")
        assert len(config.eval_metric) == 3
        assert config.eval_metric[0] == "per_class_f1_score"
        assert config.eval_metric[1] == "precision_recall"
        assert config.eval_metric[2] == "hit_at_20"
        assert len(config.decoder_edge_feat) == 1
        assert config.decoder_edge_feat[("query","match","asin")] == ["feat0", "feat1"]
        assert config.class_loss_func == BUILTIN_CLASS_LOSS_FOCAL

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "target_etype")
        check_failure(config, "reverse_edge_types_map")
        check_failure(config, "multilabel")
        check_failure(config, "num_classes")
        check_failure(config, "num_decoder_basis")
        check_failure(config, "remove_target_edge_type")
        check_failure(config, "decoder_edge_feat")
        check_failure(config, "class_loss_func")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test_fail2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "target_etype")
        check_failure(config, "eval_metric")
        check_failure(config, "decoder_edge_feat")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'edge_class_test_fail3.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.target_etype[0] == ("query", "match", "asin")
        check_failure(config, "eval_metric")

def create_lp_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["link_prediction"] = {
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "train_negative_sampler": BUILTIN_LP_JOINT_NEG_SAMPLER,
        "num_negative_edges": 4,
        "num_negative_edges_eval": 100,
        "train_etype": ["query,exactmatch,asin"],
        "eval_etype": ["query,exactmatch,asin"],
        "exclude_training_targets": True,
        "reverse_edge_types_map": ["query,exactmatch,rev-exactmatch,asin"],
        "gamma": 2.0,
        "lp_loss_func": BUILTIN_LP_LOSS_LOGSIGMOID_RANKING,
        "lp_embed_normalizer": GRAPHSTORM_LP_EMB_L2_NORMALIZATION,
        "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
        "eval_metric": "MRR",
        "lp_decoder_type": "dot_product",
        "lp_edge_weight_for_loss": ["weight"],
        "model_select_etype": "query,click,asin"
    }
    # config for check default value
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "train_negative_sampler": "udf", # we allow udf sampler
        "train_etype": ["query,exactmatch,asin","query,click,asin"],
        "eval_etype": ["query,exactmatch,asin","query,click,asin"],
        "exclude_training_targets": False,
        "reverse_edge_types_map": None,
        "eval_metric": ["mrr"],
        "gamma": 1.0,
        "alpha": 2.0,
        "lp_loss_func": BUILTIN_LP_LOSS_CONTRASTIVELOSS
    }
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "train_negative_sampler": "udf", # we allow udf sampler
        "train_etype": ["query,exactmatch,asin","query,click,asin"],
        "eval_metric": ["mrr"],
        "lp_loss_func": BUILTIN_LP_LOSS_BPR
    }
    with open(os.path.join(tmp_path, file_name+"3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "num_negative_edges": 0,
        "num_negative_edges_eval": 0,
        "train_etype": "query,exactmatch,asin",
        "eval_etype": "query,exactmatch,asin",
        "exclude_training_targets": "error",
        "reverse_edge_types_map": "query,exactmatch,rev-exactmatch,asin",
        "lp_loss_func": "unknown",
        "lp_decoder_type": "transr",
        "lp_edge_weight_for_loss": ["query,click,asin:weight1"],
        "model_select_etype": "fail"
    }
    # config for check error value
    with open(os.path.join(tmp_path, file_name+"_fail1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "exclude_training_targets": True,
        "reverse_edge_types_map": [],
        "train_etype": "query,exactmatch,asin",
        "lp_edge_weight_for_loss": ["query,exactmatch,asin:weight0", "query,exactmatch,asin:weight1"] # define edge weight multiple times
    }
    with open(os.path.join(tmp_path, file_name+"_fail2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "eval_metric": "error"
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "eval_metric": ["mrr", "error"], # one of metrics is not supported
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "eval_metric": {}, # eval metric must be string or list
    }
    with open(os.path.join(tmp_path, file_name+"_fail_metric3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "adversarial_temperature": 0.1,
    }
    with open(os.path.join(tmp_path, file_name+"_adv_temp.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "lp_loss_func": BUILTIN_LP_LOSS_CONTRASTIVELOSS,
        "adversarial_temperature": 0.1,
    }
    with open(os.path.join(tmp_path, file_name+"_adv_temp_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "lp_decoder_type": "rotate",
    }
    with open(os.path.join(tmp_path, file_name+"_rotate.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "lp_decoder_type": "transe_l1",
    }
    with open(os.path.join(tmp_path, file_name + "_transe_l1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "lp_decoder_type": "transe_l2",
    }
    with open(os.path.join(tmp_path, file_name + "_transe_l2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["link_prediction"] = {
        "lp_decoder_type": "transe_l3",
    }
    with open(os.path.join(tmp_path, file_name + "_fail_transe.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_lp_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_lp_config(Path(tmpdirname), 'lp_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_default.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.train_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER
        assert config.num_negative_edges == 16
        assert config.num_negative_edges_eval == 1000
        assert config.lp_decoder_type == BUILTIN_LP_DISTMULT_DECODER
        assert config.train_etype == None
        assert config.eval_etype == None
        check_failure(config, "exclude_training_targets")
        assert len(config.reverse_edge_types_map) == 0
        assert config.gamma == None
        assert config.alpha == None
        assert config.lp_loss_func == BUILTIN_LP_LOSS_CROSS_ENTROPY
        assert config.adversarial_temperature == None
        assert config.lp_embed_normalizer == None
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "mrr"
        assert config.gamma == None
        assert config.alpha == None
        assert config.lp_edge_weight_for_loss == None
        assert config.model_select_etype == LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test1.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.train_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER
        assert config.num_negative_edges == 4
        assert config.num_negative_edges_eval == 100
        assert config.lp_decoder_type == BUILTIN_LP_DOT_DECODER
        assert len(config.train_etype) == 1
        assert config.train_etype[0] == ("query", "exactmatch", "asin")
        assert len(config.eval_etype) == 1
        assert config.eval_etype[0] == ("query", "exactmatch", "asin")
        assert config.exclude_training_targets == True
        assert len(config.reverse_edge_types_map) == 1
        assert config.reverse_edge_types_map[("query", "exactmatch","asin")] == \
            ("asin", "rev-exactmatch","query")
        assert config.lp_loss_func == BUILTIN_LP_LOSS_LOGSIGMOID_RANKING
        assert config.lp_embed_normalizer == GRAPHSTORM_LP_EMB_L2_NORMALIZATION
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "mrr"
        assert config.lp_edge_weight_for_loss == "weight"
        assert config.model_select_etype == ("query", "click", "asin")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test2.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.train_negative_sampler == "udf"
        assert len(config.train_etype) == 2
        assert config.train_etype[0] == ("query", "exactmatch", "asin")
        assert config.train_etype[1] == ("query", "click", "asin")
        assert len(config.eval_etype) == 2
        assert config.eval_etype[0] == ("query", "exactmatch", "asin")
        assert config.eval_etype[1] == ("query", "click", "asin")
        assert config.exclude_training_targets == False
        assert len(config.reverse_edge_types_map) == 0
        assert config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS
        assert config.lp_embed_normalizer == GRAPHSTORM_LP_EMB_L2_NORMALIZATION
        assert len(config.eval_metric) == 1
        assert config.eval_metric[0] == "mrr"
        assert config.gamma == 1.0
        assert config.alpha == 2.0
        assert config.lp_edge_weight_for_loss is None
        assert config.model_select_etype == LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test3.yaml'), local_rank=0)
        config = GSConfig(args)
        assert len(config.train_etype) == 2
        assert config.train_etype[0] == ("query", "exactmatch", "asin")
        assert config.train_etype[1] == ("query", "click", "asin")
        assert config.eval_metric[0] == "mrr"
        assert config.lp_loss_func == BUILTIN_LP_LOSS_BPR

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail1.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "num_negative_edges")
        check_failure(config, "num_negative_edges_eval")
        check_failure(config, "train_etype")
        check_failure(config, "eval_etype")
        check_failure(config, "exclude_training_targets")
        check_failure(config, "reverse_edge_types_map")
        check_failure(config, "lp_loss_func")
        check_failure(config, "lp_decoder_type")
        check_failure(config, "lp_edge_weight_for_loss")
        check_failure(config, "model_select_etype")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "exclude_training_targets")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail_metric1.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail_metric2.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail_metric3.yaml'), local_rank=0)
        config = GSConfig(args)
        check_failure(config, "eval_metric")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_adv_temp.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.lp_loss_func == BUILTIN_LP_LOSS_CROSS_ENTROPY
        assert config.adversarial_temperature == 0.1

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_adv_temp_fail.yaml'), local_rank=0)
        config = GSConfig(args)
        assert config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS
        check_failure(config, "adversarial_temperature")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_rotate.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.lp_decoder_type == BUILTIN_LP_ROTATE_DECODER

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_transe_l1.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.lp_decoder_type == BUILTIN_LP_TRANSE_L1_DECODER

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_transe_l2.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.lp_decoder_type == BUILTIN_LP_TRANSE_L2_DECODER

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'lp_test_fail_transe.yaml'),
            local_rank=0)
        config = GSConfig(args)
        check_failure(config, "lp_decoder_type")

def create_gnn_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["link_prediction"] = {}
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgat"
    }
    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": ["test_feat"],
        "edge_feat_name": ["test_feat"],
        "fanout": "10,20,30",
        "num_layers": 3,
        "hidden_size": 128,
    }
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgcn"
    }
    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": ["ntype0:feat_name"],
        "edge_feat_name": ["ntype0, rel0, ntype1:feat_name"],
        "edge_feat_mp_op": "mul",
        "fanout": "n1/a/n2:10@n1/b/n2:10,n1/a/n2:10@n1/b/n2:10@n1/c/n2:20",
        "eval_fanout": "-1,10",
        "num_layers": 2,
        "hidden_size": 128,
        "use_mini_batch_infer": True,
        "num_ffn_layers_in_gnn": 1,
        "num_ffn_layers_in_input": 1
    }
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {}
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgat"
    }
    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": ["ntype0:feat_name,feat_name2", "ntype1:fname"],
        "edge_feat_name": ["ntype0, rel0, ntype1:feat_name, feat_name2",
                           "ntype1, rel1, ntype2:fname"],
        "edge_feat_mp_op": "add",
    }
    with open(os.path.join(tmp_path, file_name+"3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["edge_classification"] = {}
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgat"
    }
    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": ["ntype0:feat_name,fname", "ntype1:fname"],
        "edge_feat_name": ["ntype0, rel0, ntype1:feat_name, fname",
                           "ntype1, rel1, ntype2:fname"]
    }
    with open(os.path.join(tmp_path, file_name+"4.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["node_classification"] = {}
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgat"
    }
    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": ["ntype0:feat_name, feat_name2 ", "ntype1: fname"],
    }
    with open(os.path.join(tmp_path, file_name+"5.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # The config here should be the last one,
    # otherwise the following default test will have wrong config
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "lm"
    }
    yaml_object["gsf"]["gnn"] = {
        "num_layers": 2, # for encoder of lm, num_layers will always be 0
        "hidden_size": 128,
    }
    with open(os.path.join(tmp_path, file_name+"6.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for check default value
    yaml_object["gsf"]["gnn"] = {
    }

    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgcn"
    }
    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": ["ntype0:feat_name", "ntype0:feat_name"],
        "edge_feat_name": ["ntype0, rel0, ntype1:feat_name", "ntype0, rel0, ntype1:feat_name"], # set feat_name twice
        "edge_feat_mp_op": "dot", # not in support ops list
        "fanout": "error", # error fanout
        "eval_fanout": "error",
        "hidden_size": 0,
        "num_layers": 0,
        "use_mini_batch_infer": "error"
    }
    with open(os.path.join(tmp_path, file_name+"_error1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": {"ntype0":"feat_name"}, # not a list
        "edge_feat_name": ["ntype0, rel0:feat_name"], # error of can_etype format
        "fanout": "10,10", # error fanout
        "eval_fanout": "10,10",
        "hidden_size": 32,
        "num_layers": 1,
    }
    with open(os.path.join(tmp_path, file_name+"_error2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["gnn"] = {
        "num_layers": 2,
        "out_emb_size": 8
    }
    with open(os.path.join(tmp_path, file_name + "_outembsize.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["gnn"] = {
        "num_layers": 1,
        "out_emb_size": 8
    }
    with open(os.path.join(tmp_path, file_name+"_outembsize_ignored.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["gnn"] = {
        "out_emb_size": 0
    }
    with open(os.path.join(tmp_path, file_name + "_outembsize_error.yaml"), "w") as f:
        yaml.dump(yaml_object, f)


def test_gnn_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_gnn_config(Path(tmpdirname), 'gnn_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test1.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.node_feat_name == "test_feat"
        assert config.edge_feat_name == "test_feat"
        assert config.edge_feat_mp_op == "concat"
        assert config.fanout == [10,20,30]
        assert config.eval_fanout == [-1, -1, -1]
        assert config.num_layers == 3
        assert config.hidden_size == 128
        assert config.use_mini_batch_infer == False

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert len(config.node_feat_name) == 1
        assert 'ntype0' in config.node_feat_name
        assert config.node_feat_name['ntype0'] == ["feat_name"]
        assert config.edge_feat_name[("ntype0", "rel0", "ntype1")] == ["feat_name"]
        assert config.edge_feat_mp_op == "mul"
        assert config.fanout[0][("n1","a","n2")] == 10
        assert config.fanout[0][("n1","b","n2")] == 10
        assert config.fanout[1][("n1","a","n2")] == 10
        assert config.fanout[1][("n1","b","n2")] == 10
        assert config.fanout[1][("n1","c","n2")] == 20
        assert config.eval_fanout == [-1,10]
        assert config.num_layers == 2
        assert config.hidden_size == 128
        assert config.use_mini_batch_infer == True
        assert config.num_ffn_layers_in_input == 1
        assert config.num_ffn_layers_in_gnn == 1

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test3.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert len(config.node_feat_name) == 2
        assert 'ntype0' in config.node_feat_name
        assert 'ntype1' in config.node_feat_name
        assert config.node_feat_name['ntype0'] == ["feat_name", "feat_name2"]
        assert config.node_feat_name['ntype1'] == ["fname"]
        assert ("ntype0", "rel0", "ntype1") in config.edge_feat_name
        assert ("ntype1", "rel1", "ntype2") in config.edge_feat_name
        assert config.edge_feat_name[("ntype0", "rel0", "ntype1")] == ["feat_name", "feat_name2"]
        assert config.edge_feat_name[("ntype1", "rel1", "ntype2")] == ["fname"]
        assert config.edge_feat_mp_op == "add"
        assert config.use_mini_batch_infer == True

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test4.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert len(config.node_feat_name) == 2
        assert 'ntype0' in config.node_feat_name
        assert 'ntype1' in config.node_feat_name
        assert len(config.node_feat_name['ntype0']) == 2
        assert "feat_name" in config.node_feat_name['ntype0']
        assert "fname" in config.node_feat_name['ntype0']
        assert config.node_feat_name['ntype1'] == ["fname"]
        assert len(config.edge_feat_name) == 2
        assert ("ntype0", "rel0", "ntype1") in config.edge_feat_name
        assert ("ntype1", "rel1", "ntype2") in config.edge_feat_name
        assert len(config.edge_feat_name[("ntype0", "rel0", "ntype1")]) == 2
        assert "feat_name" in config.edge_feat_name[("ntype0", "rel0", "ntype1")]
        assert "fname" in config.edge_feat_name[("ntype0", "rel0", "ntype1")]
        assert config.edge_feat_name[("ntype1", "rel1", "ntype2")] == ["fname"]
        assert config.edge_feat_mp_op == "concat"
        assert config.use_mini_batch_infer == True

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test5.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert len(config.node_feat_name) == 2
        assert 'ntype0' in config.node_feat_name
        assert 'ntype1' in config.node_feat_name
        assert len(config.node_feat_name['ntype0']) == 2
        assert "feat_name" in config.node_feat_name['ntype0']
        assert "feat_name2" in config.node_feat_name['ntype0']
        assert config.node_feat_name['ntype1'] == ["fname"]
        assert config.edge_feat_name is None
        assert config.edge_feat_mp_op == "concat"
        assert config.use_mini_batch_infer == True

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test6.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.num_layers == 0 # lm model does not need n layers

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.node_feat_name is None
        assert config.num_layers == 0 # lm model does not need n layers
        check_failure(config, "hidden_size") # lm model may not need hidden size
        assert config.use_mini_batch_infer == True
        assert config.num_ffn_layers_in_input == 0
        assert config.num_ffn_layers_in_gnn == 0

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_error1.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "edge_feat_name")
        check_failure(config, "edge_feat_mp_op")
        check_failure(config, "fanout")
        check_failure(config, "eval_fanout")
        check_failure(config, "hidden_size")
        check_failure(config, "num_layers")
        check_failure(config, "use_mini_batch_infer")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_error2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "node_feat_name")
        check_failure(config, "edge_feat_name")
        check_failure(config, "fanout")
        check_failure(config, "eval_fanout")

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_outembsize.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.out_emb_size == 8

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_outembsize_ignored.yaml'),
            local_rank=0)
        config = GSConfig(args)
        assert config.out_emb_size == None

        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_test_outembsize_error.yaml'),
            local_rank=0)
        config = GSConfig(args)
        check_failure(config, "out_emb_size")


def create_io_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    _, part_config = tempfile.mkstemp(dir=str(tmp_path))
    yaml_object["gsf"]["basic"].update({
        "part_config": part_config
    })
    yaml_object["gsf"]["input"] = {
    }
    yaml_object["gsf"]["output"] = {
    }

    # config for check default value
    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["input"] = {
        "restore_model_path": "./restore",
        "restore_model_layers": "dense_embed",
        "restore_optimizer_path": "./opt_restore",
    }

    yaml_object["gsf"]["output"] = {
        "save_model_path": os.path.join(tmp_path, "save"),
        "save_model_frequency": 100,
        "save_embed_path": "./save_emb",
    }

    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["output"] = {
        "save_model_path": os.path.join(tmp_path, "save"),
        "save_model_frequency": 100,
        "save_embed_path": "./save_emb",
        "save_prediction_path": "./prediction",
    }

    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_load_io_info():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_io_config(Path(tmpdirname), 'io_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'io_test_default.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.restore_model_path == None
        assert config.restore_optimizer_path == None
        assert config.save_model_path == None
        assert config.save_model_frequency == -1
        assert config.save_embed_path == None

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'io_test.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.restore_model_path == "./restore"
        assert config.restore_model_layers == ["dense_embed"]
        assert config.restore_optimizer_path == "./opt_restore"
        assert config.save_model_path == os.path.join(Path(tmpdirname), "save")
        assert config.save_model_frequency == 100
        assert config.save_embed_path == "./save_emb"
        assert config.save_prediction_path == "./save_emb"

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'io_test2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.save_embed_path == "./save_emb"
        assert config.save_prediction_path == "./prediction"

def create_lm_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "rgcn"
    }

    # config for check default value
    yaml_object["gsf"]["lm"] = {
    }

    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # With language model configured for ode type 'a'
    yaml_object["gsf"]["lm"] = {
        "lm_train_nodes": 10,
        "lm_infer_batch_size": 64,
        "freeze_lm_encoder_epochs": 0,
        "node_lm_configs": [{"lm_type": "bert",
                             "model_name": "bert-base-uncased",
                             "gradient_checkpoint": True,
                             "node_types": ['a']}]
    }

    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # With language model configured for ode type 'a'
    # There is a conflict between freeze_lm_encoder_epochs and gradient_checkpoint
    # gradient_checkpoint will be set to False if freeze_lm_encoder_epochs > 0
    yaml_object["gsf"]["lm"] = {
        "lm_train_nodes": 10,
        "lm_infer_batch_size": 64,
        "freeze_lm_encoder_epochs": 3,
        "node_lm_configs": [{"lm_type": "bert",
                             "model_name": "bert-base-uncased",
                             "gradient_checkpoint": True,
                             "node_types": ['a']}]
    }

    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # This is not language model
    yaml_object["gsf"]["lm"] = {
        "lm_train_nodes": -1,
        "lm_infer_batch_size": 1,
        "freeze_lm_encoder_epochs": 0,
        "node_lm_configs": None
    }

    with open(os.path.join(tmp_path, file_name+"3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # Invalid value for lm_train_nodes, lm_infer_batch_size and freeze_lm_encoder_epochs
    yaml_object["gsf"]["output"] = {
        "lm_train_nodes": -2,
        "lm_infer_batch_size": -1,
        "freeze_lm_encoder_epochs": -1,
    }

    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # "node_lm_configs" should not be an empty list
    yaml_object["gsf"]["output"] = {
        "node_lm_configs": []
    }

    with open(os.path.join(tmp_path, file_name+"_fail2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for check default value with gsf encoder type lm
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "lm"
    }

    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # freeze_lm_encoder_epochs does not work with model_encoder_type lm
    yaml_object["gsf"]["lm"] = {
        "freeze_lm_encoder_epochs": 3,
    }
    with open(os.path.join(tmp_path, file_name+"_fail3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for check default value with gsf encoder type mlp
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["basic"] = {
        "model_encoder_type": "mlp"
    }

    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # freeze_lm_encoder_epochs does not work with model_encoder_type mlp
    yaml_object["gsf"]["lm"] = {
        "freeze_lm_encoder_epochs": 3,
    }
    with open(os.path.join(tmp_path, file_name+"_fail4.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_lm():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_lm_config(Path(tmpdirname), 'lm_test')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lm_test_default.yaml'),
                         local_rank=0)

        config = GSConfig(args)
        assert config.lm_train_nodes == 0
        assert config.lm_infer_batch_size == 32
        assert config.freeze_lm_encoder_epochs == 0
        assert config.node_lm_configs == None

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lm_test.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.lm_train_nodes == 10
        assert config.lm_infer_batch_size == 64
        assert config.freeze_lm_encoder_epochs == 0
        assert config.node_lm_configs is not None
        assert len(config.node_lm_configs) == 1
        assert config.node_lm_configs[0]['lm_type'] == "bert"
        assert config.node_lm_configs[0]['gradient_checkpoint'] == True
        assert len(config.node_lm_configs[0]['node_types']) == 1

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lm_test2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.freeze_lm_encoder_epochs == 3
        assert config.node_lm_configs is not None
        assert len(config.node_lm_configs) == 1
        assert config.node_lm_configs[0]['gradient_checkpoint'] == False

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lm_test3.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert config.lm_train_nodes == -1
        assert config.lm_infer_batch_size == 1
        assert config.freeze_lm_encoder_epochs == 0
        assert config.node_lm_configs is None

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lm_test_fail.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "lm_train_nodes")
        check_failure(config, "lm_infer_batch_size")
        check_failure(config, "freeze_lm_encoder_epochs")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lm_test_fail2.yaml'),
                         local_rank=0)
        has_error = False
        try:
            config = GSConfig(args)
        except:
            has_error = True
        assert has_error

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lm_test_fail3.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "freeze_lm_encoder_epochs")

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'lm_test_fail4.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        check_failure(config, "freeze_lm_encoder_epochs")

def test_check_node_lm_config():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yaml_object = create_dummy_config_obj()

        with open(os.path.join(tmpdirname, "check_lm_config_default.yaml"), "w") as f:
            yaml.dump(yaml_object, f)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname),
            'check_lm_config_default.yaml'), local_rank=0)
        config = GSConfig(args)
        lm_config = {"lm_type": "bert",
                     "model_name": "bert-base-uncased",
                     "gradient_checkpoint": True,
                     "node_types": ['a']}
        old_config = dict(lm_config)
        config._check_node_lm_config(lm_config)
        assert old_config == lm_config
        lm_config = {"lm_type": "bert",
                     "model_name": "bert-base-uncased",
                     "node_types": ['a', 'b', 'c']}
        config._check_node_lm_config(lm_config)
        assert "gradient_checkpoint" in lm_config
        assert lm_config["gradient_checkpoint"] == False

        def must_fail(conf):
            has_error = False
            try:
                config._check_node_lm_config(conf)
            except:
                has_error = True
            assert has_error
        lm_config = [{}]
        must_fail(lm_config)

        lm_config = [{"lm_type": "bert"}]
        must_fail(lm_config)

        lm_config = [{"lm_type": "bert",
                      "model_name": "bert-base-uncased",}]
        must_fail(lm_config)

        lm_config = [{"lm_type": "bert",
                      "model_name": "bert-base-uncased",
                      "node_types": []}]
        must_fail(lm_config)

def test_id_mapping_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yaml_object = create_dummy_config_obj()
        part_path = os.path.join(tmpdirname, "graph")
        yaml_object["gsf"]["basic"] = {
            "part_config": os.path.join(part_path, "graph.json"),
        }

        with open(os.path.join(tmpdirname, "check_lm_config_default.yaml"), "w") as f:
            yaml.dump(yaml_object, f)
            os.mkdir(part_path)
            with open(os.path.join(part_path, "graph.json"), "w") as j_f:
                json.dump({}, j_f)

            part_path_p0 = os.path.join(part_path, "part0")
            part_path_p1 = os.path.join(part_path, "part1")
            os.mkdir(part_path_p0)
            os.mkdir(part_path_p1)

            args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname),
                'check_lm_config_default.yaml'), local_rank=0)
            config = GSConfig(args)
            assert config.node_id_mapping_file is None
            assert config.edge_id_mapping_file is None

            # create dummpy node id mapping files
            id_map = {
                "n0": th.arange(10),
                "n1": th.arange(20)
            }
            # GConstruct node id mapping files are stored under dist graph folder
            nid_map_file = os.path.join(part_path, "node_mapping.pt")
            eid_map_file = os.path.join(part_path, "edge_mapping.pt")
            th.save(id_map, nid_map_file)
            th.save(id_map, eid_map_file)

            assert config.node_id_mapping_file == nid_map_file
            assert config.edge_id_mapping_file == eid_map_file

            os.remove(nid_map_file)
            os.remove(eid_map_file)

            assert config.node_id_mapping_file is None
            assert config.edge_id_mapping_file is None

            # Dist partition node id mapping files are stored under part0,
            # part1 ... folders.
            nid_map_file = os.path.join(part_path_p0, "orig_nids.dgl")
            eid_map_file = os.path.join(part_path_p0, "orig_eids.dgl")
            dgl.data.utils.save_tensors(nid_map_file, id_map)
            dgl.data.utils.save_tensors(eid_map_file, id_map)
            nid_map_file = os.path.join(part_path_p1, "orig_nids.dgl")
            eid_map_file = os.path.join(part_path_p1, "orig_eids.dgl")
            dgl.data.utils.save_tensors(nid_map_file, id_map)
            dgl.data.utils.save_tensors(eid_map_file, id_map)

            assert config.node_id_mapping_file == part_path
            assert config.edge_id_mapping_file == part_path

def create_dummy_nc_config():
    return {
        "target_ntype": "a",
        "label_field": "label_c",
        "multilabel": True,
        "num_classes": 20,
        "eval_metric": ["F1_score", "precision_recall", "ROC_AUC"],
        "multilabel_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2",
        "imbalance_class_weights": "1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2",
        "batch_size": 20,
        "task_weight": 1,
        "gamma": 2.,
        "alpha": 0.25,
        "mask_fields": ["class_train_mask", "class_eval_mask", "class_test_mask"]
    }

def create_dummy_nr_config():
    return {
        "target_ntype": "a",
        "label_field": "label_r",
        "task_weight": 0.5,
        "mask_fields": ["reg_train_mask", "reg_eval_mask", "reg_test_mask"]
    }

def create_dummy_ec_config():
    return {
        "target_etype": ["query,match,asin"],
        "reverse_edge_types_map": [],
        "label_field": "label_ec",
        "multilabel": True,
        "num_classes": 4,
        "num_decoder_basis": 4,
        "remove_target_edge_type": False,
        "decoder_type": "MLPDecoder",
        "decoder_edge_feat": ["feat"],
        "eval_metric": ["precision_recall"],
        "multilabel_weights": "1,2,3,1",
        "imbalance_class_weights": "1,2,3,1",
        "batch_size": 20,
        "task_weight": 1,
        "mask_fields": ["ec_train_mask", "ec_eval_mask", "ec_test_mask"]
    }

def create_dummy_er_config():
    return {
        "target_etype": ["query,match-2,asin"],
        "label_field": "label_er",
        "eval_metric": ["mse"],
        "decoder_edge_feat": ["query,match-2,asin:feat0,feat1"],
        "task_weight": 1,
        "mask_fields": ["er_train_mask", "er_eval_mask", "er_test_mask"]
    }

def create_dummy_lp_config():
    return {
        "train_negative_sampler": BUILTIN_LP_JOINT_NEG_SAMPLER,
        "num_negative_edges": 4,
        "num_negative_edges_eval": 100,
        "train_etype": ["query,exactmatch,asin"],
        "eval_etype": ["query,exactmatch,asin"],
        "exclude_training_targets": True,
        "reverse_edge_types_map": ["query,exactmatch,rev-exactmatch,asin"],
        "gamma": 2.0,
        "lp_loss_func": BUILTIN_LP_LOSS_CROSS_ENTROPY,
        "lp_embed_normalizer": GRAPHSTORM_LP_EMB_L2_NORMALIZATION,
        "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
        "eval_metric": "MRR",
        "lp_edge_weight_for_loss": ["weight"],
        "task_weight": 1,
        "mask_fields": ["lp_train_mask", "lp_eval_mask", "lp_test_mask"]
    }

def create_dummy_lp_config2():
    return {
        "lp_loss_func": BUILTIN_LP_LOSS_CONTRASTIVELOSS,
        "lp_decoder_type": BUILTIN_LP_DISTMULT_DECODER,
        "task_weight": 2,
        "mask_fields": ["lp2_train_mask", "lp2_eval_mask", "lp2_test_mask"],
        "exclude_training_targets": False
    }

def create_dummy_nfr_config():
    return {
        "target_ntype": "a",
        "reconstruct_nfeat_name": "rfeat",
        "task_weight": 0.5,
        "mask_fields": ["nfr_train_mask", "nfr_eval_mask", "nfr_test_mask"]
    }

def create_dummy_nfr_config2():
    return {
        "target_ntype": "a",
        "reconstruct_nfeat_name": "rfeat",
        "mask_fields": ["nfr_train_mask", "nfr_eval_mask", "nfr_test_mask"],
        "eval_metric": "rmse"
    }

def create_dummy_efr_config():
    return {
        "target_etype": ["a,r,b"],
        "reconstruct_efeat_name": "rfeat",
        "task_weight": 0.5,
        "mask_fields": ["efr_train_mask", "efr_eval_mask", "efr_test_mask"]
    }

def create_dummy_efr_config2():
    return {
        "target_etype": ["a,r,b"],
        "reconstruct_efeat_name": "rfeat",
        "mask_fields": ["efr_train_mask", "efr_eval_mask", "efr_test_mask"],
        "eval_metric": "rmse"
    }

def create_multi_task_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["basic"] = {
        "backend": "gloo",
    }
    yaml_object["gsf"]["hyperparam"] = {
        "batch_size": 64,
        "eval_batch_size": 128,
    }
    yaml_object['gsf']["multi_task_learning"] = [
        {
            BUILTIN_TASK_NODE_CLASSIFICATION : create_dummy_nc_config()
        },
        {
            BUILTIN_TASK_NODE_REGRESSION : create_dummy_nr_config()
        },
        {
            BUILTIN_TASK_EDGE_CLASSIFICATION : create_dummy_ec_config()
        },
        {
            BUILTIN_TASK_EDGE_REGRESSION : create_dummy_er_config()

        },
        {
            BUILTIN_TASK_LINK_PREDICTION : create_dummy_lp_config()
        },
        {
            BUILTIN_TASK_LINK_PREDICTION : create_dummy_lp_config2()
        },
        {
            BUILTIN_TASK_RECONSTRUCT_NODE_FEAT: create_dummy_nfr_config()
        },
        {
            BUILTIN_TASK_RECONSTRUCT_NODE_FEAT: create_dummy_nfr_config2()
        },
        {
            BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT: create_dummy_efr_config()
        },
        {
            BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT: create_dummy_efr_config2()
        },
    ]

    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_multi_task_config():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_multi_task_config(Path(tmpdirname), 'multi_task_test')

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'multi_task_test_default.yaml'), local_rank=0)
        config = GSConfig(args)

        assert len(config.multi_tasks) == 10
        nc_config = config.multi_tasks[0]
        assert nc_config.task_type == BUILTIN_TASK_NODE_CLASSIFICATION
        assert nc_config.task_id == f"{BUILTIN_TASK_NODE_CLASSIFICATION}-a-label_c"
        nc_config = nc_config.task_config
        assert nc_config.task_weight == 1
        assert nc_config.train_mask == "class_train_mask"
        assert nc_config.val_mask == "class_eval_mask"
        assert nc_config.test_mask == "class_test_mask"
        assert nc_config.target_ntype == "a"
        assert nc_config.label_field == "label_c"
        assert nc_config.multilabel == True
        assert nc_config.num_classes == 20
        assert len(nc_config.eval_metric) == 3
        assert nc_config.eval_metric[0] == "f1_score"
        assert nc_config.eval_metric[1] == "precision_recall"
        assert nc_config.eval_metric[2] == "roc_auc"
        assert nc_config.imbalance_class_weights.tolist() == [1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2]
        assert nc_config.multilabel_weights.tolist() == [1,2,3,1,2,1,2,3,1,2,1,2,3,1,2,1,2,3,1,2]
        assert nc_config.batch_size == 20
        assert nc_config.gamma == 2.
        assert nc_config.alpha == 0.25

        nr_config = config.multi_tasks[1]
        assert nr_config.task_type == BUILTIN_TASK_NODE_REGRESSION
        assert nr_config.task_id == f"{BUILTIN_TASK_NODE_REGRESSION}-a-label_r"
        nr_config = nr_config.task_config
        assert nr_config.task_weight == 0.5
        assert nr_config.train_mask == "reg_train_mask"
        assert nr_config.val_mask == "reg_eval_mask"
        assert nr_config.test_mask == "reg_test_mask"
        assert nr_config.target_ntype == "a"
        assert nr_config.label_field == "label_r"
        assert len(nr_config.eval_metric) == 1
        assert nr_config.eval_metric[0] == "rmse"
        assert nr_config.batch_size == 64

        ec_config = config.multi_tasks[2]
        assert ec_config.task_type == BUILTIN_TASK_EDGE_CLASSIFICATION
        assert ec_config.task_id == f"{BUILTIN_TASK_EDGE_CLASSIFICATION}-query_match_asin-label_ec"
        ec_config = ec_config.task_config
        assert ec_config.task_weight == 1
        assert ec_config.train_mask == "ec_train_mask"
        assert ec_config.val_mask == "ec_eval_mask"
        assert ec_config.test_mask == "ec_test_mask"
        assert ec_config.target_etype[0] == ("query", "match", "asin")
        assert ec_config.label_field == "label_ec"
        assert ec_config.multilabel == True
        assert ec_config.num_classes == 4
        assert ec_config.num_decoder_basis == 4
        assert ec_config.remove_target_edge_type == False
        assert ec_config.decoder_type == "MLPDecoder"
        assert ec_config.decoder_edge_feat == "feat"
        assert len(ec_config.eval_metric) == 1
        assert ec_config.eval_metric[0] == "precision_recall"
        assert ec_config.batch_size == 20
        assert ec_config.imbalance_class_weights.tolist() == [1,2,3,1]
        assert ec_config.multilabel_weights.tolist() == [1,2,3,1]

        er_config = config.multi_tasks[3]
        assert er_config.task_type == BUILTIN_TASK_EDGE_REGRESSION
        assert er_config.task_id == f"{BUILTIN_TASK_EDGE_REGRESSION}-query_match-2_asin-label_er"
        er_config = er_config.task_config
        assert er_config.task_weight == 1
        assert er_config.train_mask == "er_train_mask"
        assert er_config.val_mask == "er_eval_mask"
        assert er_config.test_mask == "er_test_mask"
        assert er_config.target_etype[0] == ("query", "match-2", "asin")
        assert er_config.label_field == "label_er"
        assert len(er_config.eval_metric) == 1
        assert er_config.eval_metric[0] == "mse"
        assert len(er_config.decoder_edge_feat) == 1
        assert er_config.decoder_edge_feat[("query","match-2","asin")] == ["feat0", "feat1"]
        assert er_config.batch_size == 64
        assert er_config.remove_target_edge_type == True
        assert er_config.decoder_type == "DenseBiDecoder"
        assert er_config.num_decoder_basis == 2

        lp_config = config.multi_tasks[4]
        assert lp_config.task_type == BUILTIN_TASK_LINK_PREDICTION
        assert lp_config.task_id == f"{BUILTIN_TASK_LINK_PREDICTION}-query_exactmatch_asin"
        lp_config = lp_config.task_config
        assert lp_config.task_weight == 1
        assert lp_config.train_mask == "lp_train_mask"
        assert lp_config.val_mask == "lp_eval_mask"
        assert lp_config.test_mask == "lp_test_mask"
        assert lp_config.train_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER
        assert lp_config.num_negative_edges == 4
        assert lp_config.num_negative_edges_eval == 100
        assert len(lp_config.train_etype) == 1
        assert lp_config.train_etype[0] == ("query", "exactmatch", "asin")
        assert len(lp_config.eval_etype) == 1
        assert lp_config.eval_etype[0] == ("query", "exactmatch", "asin")
        assert lp_config.exclude_training_targets == True
        assert len(lp_config.reverse_edge_types_map) == 1
        assert lp_config.reverse_edge_types_map[("query", "exactmatch","asin")] == \
            ("asin", "rev-exactmatch","query")
        assert lp_config.gamma == 2.0
        assert lp_config.lp_loss_func == BUILTIN_LP_LOSS_CROSS_ENTROPY
        assert lp_config.lp_embed_normalizer == GRAPHSTORM_LP_EMB_L2_NORMALIZATION
        assert lp_config.lp_decoder_type == BUILTIN_LP_DOT_DECODER
        assert len(lp_config.eval_metric) == 1
        assert lp_config.eval_metric[0] == "mrr"
        assert lp_config.lp_edge_weight_for_loss == "weight"

        lp_config = config.multi_tasks[5]
        assert lp_config.task_type == BUILTIN_TASK_LINK_PREDICTION
        assert lp_config.task_id == f"{BUILTIN_TASK_LINK_PREDICTION}-ALL_ETYPE"
        lp_config = lp_config.task_config
        assert lp_config.task_weight == 2
        assert lp_config.train_mask == "lp2_train_mask"
        assert lp_config.val_mask == "lp2_eval_mask"
        assert lp_config.test_mask == "lp2_test_mask"
        assert lp_config.train_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER
        assert lp_config.num_negative_edges == 16
        assert lp_config.train_etype == None
        assert lp_config.eval_etype == None
        assert lp_config.exclude_training_targets == False
        assert len(lp_config.reverse_edge_types_map) == 0
        assert lp_config.gamma == None
        assert lp_config.lp_loss_func == BUILTIN_LP_LOSS_CONTRASTIVELOSS
        assert lp_config.lp_embed_normalizer == GRAPHSTORM_LP_EMB_L2_NORMALIZATION
        assert lp_config.lp_decoder_type == BUILTIN_LP_DISTMULT_DECODER
        assert len(lp_config.eval_metric) == 1
        assert lp_config.eval_metric[0] == "mrr"
        assert config.lp_edge_weight_for_loss == None
        assert config.model_select_etype == LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL

        nfr_config = config.multi_tasks[6]
        assert nfr_config.task_type == BUILTIN_TASK_RECONSTRUCT_NODE_FEAT
        assert nfr_config.task_id == f"{BUILTIN_TASK_RECONSTRUCT_NODE_FEAT}-a-rfeat"
        nfr_config = nfr_config.task_config
        assert nfr_config.task_weight == 0.5
        assert nfr_config.train_mask == "nfr_train_mask"
        assert nfr_config.val_mask == "nfr_eval_mask"
        assert nfr_config.test_mask == "nfr_test_mask"
        assert nfr_config.target_ntype == "a"
        assert nfr_config.reconstruct_nfeat_name == "rfeat"
        assert len(nfr_config.eval_metric) == 1
        assert nfr_config.eval_metric[0] == "mse"
        assert nfr_config.batch_size == 64

        nfr_config = config.multi_tasks[7]
        assert nfr_config.task_type == BUILTIN_TASK_RECONSTRUCT_NODE_FEAT
        assert nfr_config.task_id == f"{BUILTIN_TASK_RECONSTRUCT_NODE_FEAT}-a-rfeat"
        nfr_config = nfr_config.task_config
        assert nfr_config.task_weight == 1.0
        assert nfr_config.train_mask == "nfr_train_mask"
        assert nfr_config.val_mask == "nfr_eval_mask"
        assert nfr_config.test_mask == "nfr_test_mask"
        assert nfr_config.target_ntype == "a"
        assert nfr_config.reconstruct_nfeat_name == "rfeat"
        assert len(nfr_config.eval_metric) == 1
        assert nfr_config.eval_metric[0] == "rmse"
        assert nfr_config.batch_size == 64

        # reconstruct edge feat
        efr_config = config.multi_tasks[8]
        assert efr_config.task_type == BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT
        assert efr_config.task_id == f"{BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT}-a_r_b-rfeat"
        efr_config = efr_config.task_config
        assert efr_config.task_weight == 0.5
        assert efr_config.train_mask == "efr_train_mask"
        assert efr_config.val_mask == "efr_eval_mask"
        assert efr_config.test_mask == "efr_test_mask"
        assert efr_config.target_etype[0] == ("a", "r", "b")
        assert efr_config.reconstruct_efeat_name == "rfeat"
        assert len(efr_config.eval_metric) == 1
        assert efr_config.eval_metric[0] == "mse"
        assert efr_config.batch_size == 64

        efr_config = config.multi_tasks[9]
        assert efr_config.task_type == BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT
        assert efr_config.task_id == f"{BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT}-a_r_b-rfeat"
        efr_config = efr_config.task_config
        assert efr_config.task_weight == 1.0
        assert efr_config.train_mask == "efr_train_mask"
        assert efr_config.val_mask == "efr_eval_mask"
        assert efr_config.test_mask == "efr_test_mask"
        assert efr_config.target_etype[0] == ("a", "r", "b")
        assert efr_config.reconstruct_efeat_name == "rfeat"
        assert len(efr_config.eval_metric) == 1
        assert efr_config.eval_metric[0] == "rmse"
        assert efr_config.batch_size == 64


def test_save_combined_config():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a basic config file
        create_basic_config(Path(tmpdirname), 'combined_test')
        save_model_path = os.path.join(tmpdirname, "model")

        # Create args with an override
        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'combined_test.yaml'),
            local_rank=0,
            # Set save_model_path to trigger combined config saving
            save_model_path=os.path.join(tmpdirname, "model"),
            lr=0.02,  # Override the lr, fanout values from the yaml
            fanout="15,10",
        )

        # Create GSConfig, this will also create the updated yaml file
        _ = GSConfig(args)

        # Updated config should exist under the save model path
        updated_yaml = os.path.join(save_model_path, GS_RUNTIME_TRAINING_CONFIG_FILENAME)

        # Verify the file exists
        assert os.path.exists(updated_yaml)

        # Load the saved config and verify it contains the overridden value
        with open(updated_yaml, 'r') as f:
            updated_config = yaml.safe_load(f)

        # Check that the existing arg values were updated, under their existing section
        assert updated_config['gsf']['hyperparam']['lr'] == 0.02
        assert updated_config['gsf']['gnn']['fanout'] == "15,10"

def test_save_combined_new_argument():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a basic config file
        create_basic_config(Path(tmpdirname), 'combined_test')
        save_model_path = os.path.join(tmpdirname, "model")

        # Create args with an override, where the new arg did not exist in the original yaml
        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'combined_test.yaml'),
            local_rank=0,
            # Set save_model_path to trigger combined config saving
            save_model_path=save_model_path,
            wd_l2norm=0.0001  # Insert a new runtime value that did not exist in the yaml
        )

        # Create GSConfig, this will also create the updated yaml file
        gs_config = GSConfig(args)
        # Ensure the new runtime value was added to the config
        assert gs_config.wd_l2norm == 0.0001

        # Updated config should exist under the save model path
        updated_yaml = os.path.join(save_model_path, GS_RUNTIME_TRAINING_CONFIG_FILENAME)

        # Verify the file exists
        assert os.path.exists(updated_yaml)

        # Load the saved config and verify it contains the overridden value
        with open(updated_yaml, 'r') as f:
            updated_config = yaml.safe_load(f)

        # Check that the wd_l2norm value was added to the 'runtime' key
        assert updated_config['gsf']['runtime']['wd_l2norm'] == 0.0001

def test_save_combined_without_partconfig():
    """When the part_config file listed in the train YAML file is missing,
    we should only log a warning, not fail the program"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a basic config file
        create_basic_config(Path(tmpdirname), 'combined_test')
        save_model_path = os.path.join(tmpdirname, "model")
        part_config_file = os.path.join(tmpdirname, "part.json")

        # Create args with an override, where the new arg did not exist in the original yaml
        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'combined_test.yaml'),
            local_rank=0,
            # Set save_model_path to trigger combined config saving
            save_model_path=save_model_path,
        )

        # Remove the part config file
        os.remove(part_config_file)

        # Try to create GSConfing, this should only log a warning about the missing part config
        with pytest.warns(UserWarning, match="Partition config file .* does not exist. .*"):
            _ = GSConfig(args)

def test_copy_gconstruct_config():
    """Ensure that we save a copy of the GConstruct config with model, if one exists"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a basic config file
        create_basic_config(Path(tmpdirname), 'combined_test')
        save_model_path = os.path.join(tmpdirname, "model")
        # Copy a gconstruct file into the graph data input path
        copy_gconstruct_config(tmpdirname, GS_RUNTIME_TRAINING_CONFIG_FILENAME)
        # Test assertion, there needs to be a part config file under the input
        assert os.path.exists(os.path.join(tmpdirname, "part.json"))

        # Create runtime args, makings sure to include a save model path
        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'combined_test.yaml'),
            local_rank=0,
            # Set save_model_path to trigger combined config saving
            save_model_path=save_model_path,
        )

        # Create GSConfig, this will also copy the GConstruct config
        _ = GSConfig(args)

        # Copied GConstruct config should exist under the save model path
        copied_gc_config = os.path.join(save_model_path, GS_RUNTIME_TRAINING_CONFIG_FILENAME)

        # Verify the file exists
        assert os.path.exists(copied_gc_config)

def test_missing_gconstruct_config():
    """Ensure that we log a warning if the GConstruct config is missing"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a basic config file
        create_basic_config(Path(tmpdirname), 'combined_test')
        save_model_path = os.path.join(tmpdirname, "model")

        # Create runtime args, makings sure to include a save model path
        args = Namespace(
            yaml_config_file=os.path.join(Path(tmpdirname), 'combined_test.yaml'),
            local_rank=0,
            # Set save_model_path to trigger combined config saving
            save_model_path=save_model_path,
        )

        # Create GSConfig, this will try to copy the GConstruct config, ensure we log a warning:
        with pytest.warns(UserWarning, match="Graph construction config .* not found in .*"):
            _ = GSConfig(args)

def create_fname_test_gnn_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["link_prediction"] = {}

    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": ["ntype0:feat_name, feat_name2 ", "ntype1:fname"],
    }
    with open(os.path.join(tmp_path, file_name+"1.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": ["ntype0:feat_name, feat_name2 ", "ntype1:fname", "ntype0:feat_name3"],
    }
    with open(os.path.join(tmp_path, file_name+"2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    yaml_object["gsf"]["gnn"] = {
        "node_feat_name": ["ntype0:feat_name, feat_name2 ", "ntype1:fname,fname1", "ntype0:feat_name,feat_name3", "ntype1:fname"],
    }
    with open(os.path.join(tmp_path, file_name+"3.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

def test_node_feat_name():
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_fname_test_gnn_config(Path(tmpdirname), 'node_feat_test')

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_feat_test1.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert len(config.node_feat_name) == 2
        assert 'ntype0' in config.node_feat_name
        assert config.node_feat_name['ntype0'] == ["feat_name", "feat_name2"]
        assert 'ntype1' in config.node_feat_name
        assert config.node_feat_name['ntype1'] == ["fname"]

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_feat_test2.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert len(config.node_feat_name) == 2
        assert 'ntype0' in config.node_feat_name
        assert len(config.node_feat_name['ntype0']) == 2
        assert isinstance(config.node_feat_name['ntype0'][0], FeatureGroup)
        assert isinstance(config.node_feat_name['ntype0'][1], FeatureGroup)
        assert len(config.node_feat_name['ntype0'][0].feature_group) == 2
        assert config.node_feat_name['ntype0'][0].feature_group == ["feat_name", "feat_name2"]
        assert len(config.node_feat_name['ntype0'][1].feature_group) == 1
        assert config.node_feat_name['ntype0'][1].feature_group == ["feat_name3"]
        assert 'ntype1' in config.node_feat_name
        assert config.node_feat_name['ntype1'] == ["fname"]

        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'node_feat_test3.yaml'),
                         local_rank=0)
        config = GSConfig(args)
        assert len(config.node_feat_name) == 2
        assert 'ntype0' in config.node_feat_name
        assert len(config.node_feat_name['ntype0']) == 2
        assert isinstance(config.node_feat_name['ntype0'][0], FeatureGroup)
        assert isinstance(config.node_feat_name['ntype0'][1], FeatureGroup)
        assert len(config.node_feat_name['ntype0'][0].feature_group) == 2
        assert config.node_feat_name['ntype0'][0].feature_group == ["feat_name", "feat_name2"]
        assert len(config.node_feat_name['ntype0'][1].feature_group) == 2
        assert config.node_feat_name['ntype0'][1].feature_group == ["feat_name", "feat_name3"]

        assert 'ntype1' in config.node_feat_name
        assert len(config.node_feat_name['ntype1']) == 2
        assert isinstance(config.node_feat_name['ntype1'][0], FeatureGroup)
        assert isinstance(config.node_feat_name['ntype1'][1], FeatureGroup)
        assert len(config.node_feat_name['ntype1'][0].feature_group) == 2
        assert config.node_feat_name['ntype1'][0].feature_group == ["fname", "fname1"]
        assert len(config.node_feat_name['ntype1'][1].feature_group) == 1
        assert config.node_feat_name['ntype1'][1].feature_group == ["fname"]

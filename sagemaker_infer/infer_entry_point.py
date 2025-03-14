#-*- coding:utf-8 -*-

import os
import json
import dgl
from datetime import datetime as dt

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
import numpy as np

import graphstorm as gs
from graphstorm.config import GSConfig
from argparse import Namespace

# INPUT_SIZE = 390
# HIDDEN_SIZE = int(os.getenv('HIDDEN_SIZE', '16'))
# N_LAYERS = 2
# OUT_SIZE = 2
# EMBEDDING_SIZE = 390
BASE_PATH = '/opt/ml/model/code/'
# TARGET_FEAT_MEAN = None
# TARGET_FEAT_STD = None


# SageMaker inference functions
def model_fn(model_dir):
    """
    """
    print('--START model loading... ')

    s_t = dt.now()

    gs.initialize()
    args = Namespace(yaml_config_file=os.path.join(model_dir, 'acm_nc.yaml'), local_rank=0)
    config = GSConfig(args)
    # load the dummy distributed graph
    # TODO: should get the graph name from either user's input argument or by autmatically
    #       extracted from JSON file.
    dummy_g = dgl.distributed.DistGraph('acm', \
                                        part_config=os.path.join(model_dir, 'acm_gs_1p/acm.json'))
    # rebuild the model
    # TODO: should like gsf.py to check what kind of models we need to create
    model = gs.create_builtin_node_gnn_model(dummy_g, config, train_task=False)
    model.restore_model(config.restore_model_path)

    e_t = dt.now()
    print(f'--model_fn: used {(e_t - s_t).microseconds} ms ...')

    print(model)

    return model


def input_fn(request_body, request_content_type='application/json'):
    """
    Preprocessing request_body that is in JSON format.
    :param request_body:
    :param request_content_type:
    :return:
    """
    print('--START processing input data... ')

    # --------------------- receive request ------------------------------------------------ #
    input_data = json.loads(request_body)

    s_t = dt.now()

    #TODO: Data Extraction and Transformation

    e_t = dt.now()
    print(f'--input_fn: used {(e_t - s_t).microseconds} ms ...')

    return input_data


def predict_fn(input_data, model):
    """ Make prediction
    """
    print('--START model prediction... ')

    s_t = dt.now()

    graph, new_n_feats, new_pred_target_id = input_data

    with th.no_grad():
        logits = model(graph, new_n_feats)
        res = logits[new_pred_target_id].cpu().detach().numpy()

    e_t = dt.now()
    print(f'--predict_fn: used {(e_t - s_t).microseconds} ms ...')

    return res[1]


if __name__ == '__main__':
    # method for local testing

    # --- load saved model ---
    # s_t = dt.now()
    #
    # model = model_fn('../')
    #
    # e_t = dt.now()
    # print('--Load Model: {}'.format((e_t - s_t).microseconds))

    # --- load subgraph data ---
    # s_t = dt.now()

    # subgraph_file = 'subgraph_100_101.pkl'
    # with open('../clients_python/subgraph_100_101.pkl', 'rb') as f:
    #     subgraph_dict = pickle.load(f)

    # e_t = dt.now()
    # print('--Load Graph Data: {}'.format((e_t - s_t).microseconds))

    # --- build a new subgraph ---
    # s_t = dt.now()

    # g, n_feats, new_pred_target_id = recreate_grpha_data(subgraph_dict, None, 100)


    # e_t = dt.now()
    # print('--Convert Graph: {}'.format((e_t - s_t).microseconds))

    # --- use saved model to run prediction ---
    # print('------------------ Predict Logits -------------------')
    # s_t = dt.now()
    #
    # logits = model(g, n_feats)
    #
    # e_t = dt.now()
    # print('--Convert Graph: {}'.format((e_t - s_t).microseconds))
    #
    # print(logits[new_pred_target_id])
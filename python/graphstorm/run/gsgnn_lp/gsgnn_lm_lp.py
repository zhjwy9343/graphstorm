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

    GSgnn pure gpu link prediction.
"""
import os
import logging

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.trainer import GSgnnLinkPredictionTrainer
from graphstorm.dataloading import GSgnnData
from graphstorm.model.utils import save_full_node_embeddings
from graphstorm.model import do_full_graph_inference
from graphstorm.utils import rt_profiler, sys_tracker, get_device

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(True)

    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank, use_graphbolt=config.use_graphbolt)
    rt_profiler.init(config.profile_path, rank=gs.get_rank())
    sys_tracker.init(config.verbose, rank=gs.get_rank())
    # The model only uses language model(s) as its encoder
    # It will not use node or edge features
    # except LM related features.
    train_data = GSgnnData(config.part_config)
    model = gs.create_builtin_lp_model(train_data.g, config, train_task=True)
    trainer = GSgnnLinkPredictionTrainer(model, topk_model_to_save=config.topk_model_to_save)
    if config.restore_model_path is not None:
        trainer.restore_model(model_path=config.restore_model_path,
                              model_layer_to_load=config.restore_model_layers)
    trainer.setup_device(device=get_device())
    if not config.no_validation:
        # TODO(zhengda) we need to refactor the evaluator.
        # Currently, we only support mrr
        evaluator = gs.create_lp_evaluator(config)
        trainer.setup_evaluator(evaluator)
        val_idxs = train_data.get_edge_val_set(config.eval_etype)
        assert len(val_idxs) > 0, "The training data do not have validation set."
        # TODO(zhengda) we need to compute the size of the entire validation set to make sure
        # we have validation data.
    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        tracker.log_params(config.__dict__)
    trainer.setup_task_tracker(tracker)

    dataloader_cls = gs.get_builtin_lp_train_dataloader_class(config)
    train_idxs = train_data.get_edge_train_set(config.train_etype)
    dataloader = dataloader_cls(train_data,
                                train_idxs, [],
                                config.batch_size, config.num_negative_edges,
                                node_feats=config.node_feat_name,
                                edge_feats=config.edge_feat_name,
                                pos_graph_edge_feats=config.lp_edge_weight_for_loss,
                                train_task=True,
                                edge_dst_negative_field=config.train_etypes_negative_dstnode,
                                num_hard_negs=config.num_train_hard_negatives)

    # TODO(zhengda) let's use full-graph inference for now.
    test_dataloader_cls = gs.get_builtin_lp_eval_dataloader_class(config)
    val_dataloader = None
    test_dataloader = None

    val_idxs = train_data.get_edge_val_set(config.eval_etype)
    if len(val_idxs) > 0:
        if config.eval_etypes_negative_dstnode is not None:
            val_dataloader = test_dataloader_cls(train_data, val_idxs,
                config.eval_batch_size,
                config.eval_etypes_negative_dstnode,
                fixed_test_size=config.fixed_test_size,
                node_feats=config.node_feat_name,
                edge_feats=config.edge_feat_name,
                pos_graph_edge_feats=config.lp_edge_weight_for_loss)
        else:
            val_dataloader = test_dataloader_cls(train_data, val_idxs,
                config.eval_batch_size,
                config.num_negative_edges_eval,
                fixed_test_size=config.fixed_test_size,
                node_feats=config.node_feat_name,
                edge_feats=config.edge_feat_name,
                pos_graph_edge_feats=config.lp_edge_weight_for_loss)

    test_idxs = train_data.get_edge_test_set(config.eval_etype)
    if len(test_idxs) > 0:
        if config.eval_etypes_negative_dstnode is not None:
            test_dataloader = test_dataloader_cls(train_data, test_idxs,
                config.eval_batch_size,
                config.eval_etypes_negative_dstnode,
                fixed_test_size=config.fixed_test_size,
                node_feats=config.node_feat_name,
                edge_feats=config.edge_feat_name,
                pos_graph_edge_feats=config.lp_edge_weight_for_loss)
        else:
            test_dataloader = test_dataloader_cls(train_data, test_idxs,
                config.eval_batch_size,
                config.num_negative_edges_eval,
                fixed_test_size=config.fixed_test_size,
                node_feats=config.node_feat_name,
                edge_feats=config.edge_feat_name,
                pos_graph_edge_feats=config.lp_edge_weight_for_loss)

    # Preparing input layer for training or inference.
    # The input layer can pre-compute node features in the preparing step if needed.
    # For example pre-compute all BERT embeddings
    model.prepare_input_encoder(train_data)
    if config.save_model_path is not None:
        save_model_path = config.save_model_path
    elif config.save_embed_path is not None:
        # If we need to save embeddings, we need to save the model somewhere.
        save_model_path = os.path.join(config.save_embed_path, "model")
    else:
        save_model_path = None
    trainer.fit(train_loader=dataloader, val_loader=val_dataloader,
                test_loader=test_dataloader, num_epochs=config.num_epochs,
                save_model_path=save_model_path,
                use_mini_batch_infer=config.use_mini_batch_infer,
                save_model_frequency=config.save_model_frequency,
                save_perf_results_path=config.save_perf_results_path,
                max_grad_norm=config.max_grad_norm,
                grad_norm_type=config.grad_norm_type)

    if config.save_embed_path is not None:
        assert config.edge_feat_name is None, 'GraphStorm node prediction training command ' + \
            'uses full-graph inference by default to generate node embeddings at the end of ' + \
            'training. But the full-graph inference method does not support edge features ' + \
            'in the current version. Please set the \"save_embed_path\" to none ' + \
            ' if you want to use edge features in training command.'
        model = gs.create_builtin_lp_model(train_data.g, config, train_task=False)
        best_model_path = trainer.get_best_model_path()
        # TODO(zhengda) the model path has to be in a shared filesystem.
        model.restore_model(best_model_path)
        model = model.to(get_device())
        # Preparing input layer for training or inference.
        # The input layer can pre-compute node features in the preparing step if needed.
        # For example pre-compute all BERT embeddings
        model.prepare_input_encoder(train_data)
        # TODO(zhengda) we may not want to only use training edges to generate GNN embeddings.
        embeddings = do_full_graph_inference(model, train_data, fanout=config.eval_fanout,
                                             edge_mask="train_mask", task_tracker=tracker)
        save_full_node_embeddings(
            train_data.g,
            config.save_embed_path,
            embeddings,
            node_id_mapping_file=config.node_id_mapping_file,
            save_embed_format=config.save_embed_format)

def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    arg_parser=generate_parser()

    # Ignore unknown args to make script more robust to input arguments
    gs_args, unknown_args = arg_parser.parse_known_args()
    logging.warning("Unknown arguments for command "
                    "graphstorm.run.gs_link_prediction: %s",
                    unknown_args)
    main(gs_args)

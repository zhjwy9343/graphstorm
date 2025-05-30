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

    Relational GNN
"""

from functools import partial
import logging

import abc
import dgl
import torch as th
from torch import nn
from dgl.distributed import node_split
from dgl.nn.pytorch.hetero import get_aggregate_fn
from .gs_layer import GSLayer

from ..utils import get_rank, barrier, is_distributed, create_dist_tensor, is_wholegraph
from ..distributed import flush_data

class GSgnnGNNEncoderInterface:
    """ The interface for builtin GraphStorm gnn encoder layer.

        The interface defines two functions that are useful in multi-task learning.
        Any GNN encoder that implements these two functions can work with
        GraphStorm multi-task learning pipeline.

        Note: We can define more functions when necessary.
    """
    @abc.abstractmethod
    def skip_last_selfloop(self):
        """ Skip the self-loop of the last GNN layer.
        """

    @abc.abstractmethod
    def reset_last_selfloop(self):
        """ Reset the self-loop setting of the last GNN layer.
        """

class GraphConvEncoder(GSLayer):     # pylint: disable=abstract-method
    r"""General encoder for graph data.

    .. versionchanged:: 0.4.0
        Add two new arguments ``edge_feat_name`` and ``edge_feat_mp_op`` in v0.4.0 to
        support edge features in encoders.

    Parameters
    ----------
    h_dim : int
        Hidden dimension
    out_dim : int
        Output dimension
    num_hidden_layers : int
        Number of hidden layers. Total GNN layers is equal to num_hidden_layers + 1. Default 1.
    edge_feat_name: dict of list of str
        User provided edge feature names in the format of {etype1:[feat1, feat2, ...],
        etype2:[...], ...}, or None if not provided.
    edge_feat_mp_op: str
        The opration method to combine source node embeddings with edge embeddings in message
        passing. Options include `concat`, `add`, `sub`, `mul`, and `div`.
        ``concat`` operation will concatenate the source node features with edge features;
        ``add`` operation will add the source node features with edge features together;
        ``sub`` operation will subtract the source node features by edge features;
        ``mul`` operation will multiply the source node features with edge features; and
        ``div`` operation will divide the source node features by edge features.
    """
    def __init__(self,
                 h_dim,
                 out_dim,
                 num_hidden_layers=1,
                 edge_feat_name=None,
                 edge_feat_mp_op='concat'):
        super(GraphConvEncoder, self).__init__()
        self._h_dim = h_dim
        self._out_dim = out_dim
        self._num_hidden_layers = num_hidden_layers
        self._layers = nn.ModuleList()  # GNN layers.
        self.edge_feat_name = edge_feat_name
        self.edge_feat_mp_op = edge_feat_mp_op
        self.is_support_edge_feat()

    def is_support_edge_feat(self):
        """ Check if a GraphConvEncoder child class supports edge feature in message passing.
        
        By default GNN encoders do not support edge feature. A child class can 
        overwrite this method when it supports edge feature in message passing
        computation.
        """
        assert self.edge_feat_name is None, 'Edge features are not supported in the ' + \
                                            f'\"{self.__class__}\" encoder.'

    def is_using_edge_feat(self):
        """ Check if an instance of this class is using edge features.

        This method is for functions related to trainers and inferrers, e.g.,
        ``do_full_graph_inference()``, to determine if they allow to use edge features.

        The current implementation only checks if the initialization of GraphConvEncoder or its
        child classes enables edge feature support with non-None ``edge_feat_name``.
        """
        return self.edge_feat_name is not None

    @property
    def in_dims(self):
        return self._h_dim

    @property
    def out_dims(self):
        return self._out_dim

    @property
    def h_dims(self):
        """ The hidden dimension size.
        """
        return self._h_dim

    @property
    def num_layers(self):
        """ The number of GNN layers.
        """
        # The number of GNN layer is the number of hidden layers + 1
        return self._num_hidden_layers + 1

    @property
    def layers(self):
        """ GNN layers
        """
        return self._layers

    def dist_inference(self, g, get_input_embeds, batch_size, fanout,
                       edge_mask=None, task_tracker=None):
        """Distributed inference of final representation over all node types.
        Parameters
        ----------
        g : DistGraph
            The distributed graph.
        gnn_encoder : GraphConvEncoder
            The GNN encoder on the graph.
        get_input_embeds : callable
            Get the node features of the input nodes.
        batch_size : int
            The batch size for the GNN inference.
        fanout : list of int
            The fanout for computing the GNN embeddings in a GNN layer.
        edge_mask : str
            The edge mask indicates which edges are used to compute GNN embeddings.
        task_tracker : GSTaskTrackerAbc
            The task tracker.
        Returns
        -------
        dict of Tensor : the final GNN embeddings of all nodes.
        """
        return dist_inference(g, self, get_input_embeds, batch_size, fanout,
                            edge_mask=edge_mask, task_tracker=task_tracker)

def prepare_for_wholegraph(g, input_nodes, input_edges=None):
    """ Add missing ntypes in input_nodes for wholegraph compatibility

    Parameters
    ----------
    g : DistGraph
        Input graph
    input_nodes : dict of Tensor
        Input nodes retrieved from the dataloder
    input_edges : dict of Tensor
        Input edges retrieved from the dataloder
    """
    if input_nodes is not None:
        for ntype in g.ntypes:
            if ntype not in input_nodes:
                input_nodes[ntype] = th.empty((0,), dtype=g.idtype)

    if input_edges is not None:
        for etype in g.canonical_etypes:
            if etype not in input_edges:
                input_edges[etype] = th.empty((0,), dtype=g.idtype)

def dist_minibatch_inference(g, gnn_encoder, get_input_embeds, batch_size, fanout,
                             edge_mask=None, target_ntypes=None, task_tracker=None):
    """Distributed inference of final representation over all node types
       using mini-batch inference.

    .. versionchanged:: 0.4.0
        Change ``get_input_embeds`` outputs in v0.4.0 to support edge features in message
        passing computation.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    gnn_encoder : GraphConvEncoder
        The GNN encoder on the graph.
    get_input_embeds : func
        A function used ot get input embeddings.
    batch_size : int
        The batch size for the GNN inference.
    fanout : list of int
        The fanout for computing the GNN embeddings in a GNN layer.
    edge_mask : str
        The edge mask indicates which edges are used to compute GNN embeddings.
        task_tracker : GSTaskTrackerAbc
        The task tracker.
    target_ntypes: list of str
        Node types that need to compute node embeddings.
    task_tracker: GSTaskTrackerAbc
        Task tracker

    Returns
    -------
    dict of Tensor : the final GNN embeddings of all nodes.
    """
    device = gnn_encoder.device
    fanout = [-1] * gnn_encoder.num_layers \
        if fanout is None or len(fanout) == 0 else fanout
    target_ntypes = g.ntypes if target_ntypes is None else target_ntypes
    with th.no_grad():
        infer_nodes = {}
        out_embs = {}
        for ntype in target_ntypes:
            h_dim = gnn_encoder.out_dims
            # Create dist tensor to store the output embeddings
            out_embs[ntype] = create_dist_tensor((g.number_of_nodes(ntype), h_dim),
                                                 dtype=th.float32, name='h-last',
                                                 part_policy=g.get_node_partition_policy (ntype),
                                                 # TODO(zhengda) this makes the tensor persistent.
                                                 persistent=True)
            infer_nodes[ntype] = node_split(th.ones((g.number_of_nodes(ntype),),
                                                        dtype=th.bool),
                                                partition_book=g.get_partition_book(),
                                                ntype=ntype, force_even=False)

        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout, mask=edge_mask)
        dataloader = dgl.dataloading.DistNodeDataLoader(g, infer_nodes, sampler,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            drop_last=False)

        # Follow
        # https://github.com/dmlc/dgl/blob/1.0.x/python/dgl/distributed/dist_dataloader.py#L116
        # DistDataLoader.expected_idxs is the length of the datalaoder
        len_dataloader = max_num_batch = dataloader.expected_idxs

        tensor = th.tensor([len_dataloader], device=device)
        if is_distributed():
            th.distributed.all_reduce(tensor, op=th.distributed.ReduceOp.MAX)
            max_num_batch = tensor[0]
        dataloader_iter = iter(dataloader)

        # WholeGraph does not support imbalanced batch numbers across processes/trainers
        # TODO (IN): Fix dataloader to have same number of minibatches.
        for iter_l in range(max_num_batch):
            tmp_keys = []
            blocks = None
            if iter_l < len_dataloader:
                input_nodes, output_nodes, blocks = next(dataloader_iter)
                if not isinstance(input_nodes, dict):
                    # This happens on a homogeneous graph.
                    assert len(g.ntypes) == 1
                    input_nodes = {g.ntypes[0]: input_nodes}
                if not isinstance(output_nodes, dict):
                    # This happens on a homogeneous graph.
                    assert len(g.ntypes) == 1
                    output_nodes = {g.ntypes[0]: output_nodes}
            if is_wholegraph():
                tmp_keys = [ntype for ntype in g.ntypes if ntype not in input_nodes]
                prepare_for_wholegraph(g, input_nodes)
            if iter_l % 100000 == 0 and get_rank() == 0:
                logging.info("[Rank 0] dist inference: " \
                        "finishes %d iterations.", iter_l)
            if task_tracker is not None:
                task_tracker.keep_alive(report_step=iter_l)

            if blocks is None:
                continue
            n_h, e_hs = get_input_embeds(input_nodes, blocks)
            # Remove additional keys (ntypes) added for WholeGraph compatibility
            for ntype in tmp_keys:
                del input_nodes[ntype]
            blocks = [block.to(device) for block in blocks]
            # Check if edge embeddings have values
            if any(e_hs):
                output = gnn_encoder(blocks, n_h, e_hs)
            else:
                output = gnn_encoder(blocks, n_h)

            for ntype, out_nodes in output_nodes.items():
                out_embs[ntype][out_nodes] = output[ntype].cpu()
        # The nodes are split in such a way that all processes only need to compute
        # the embeddings of the nodes in the local partition. Therefore, a barrier
        # is enough to ensure that all data have been written to memory for distributed
        # read after this function is returned.
        # Note: there is a risk here. If the nodes for inference on each partition
        # are very skewed, some of the processes may timeout in the barrier.
        barrier()
    return out_embs

def dist_inference_one_layer(layer_id, g, dataloader, target_ntypes, layer, get_input_embeds,
                             device, task_tracker):
    """ Run distributed inference for one GNN layer.

    Parameters
    ----------
    layer_id : str
        The layer ID.
    g : DistGraph
        The full distributed graph.
    target_ntypes : list of str
        The node types where we compute GNN embeddings.
    dataloader : Pytorch dataloader
        The iterator over the nodes for computing GNN embeddings.
    layer : nn module
        A GNN layer
    get_input_embeds : callable
        Get the node features.
    device : Pytorch device
        The device to run mini-batch computation.
    task_tracker : GSTaskTrackerAbc
        The task tracker.

    Returns
    -------
        dict of Tensors : the inferenced tensors.
    """
    # Follow
    # https://github.com/dmlc/dgl/blob/1.0.x/python/dgl/distributed/dist_dataloader.py#L116
    # DistDataLoader.expected_idxs is the length of the datalaoder
    len_dataloader = max_num_batch = dataloader.expected_idxs
    tensor = th.tensor([len_dataloader], device=device)
    if is_distributed():
        th.distributed.all_reduce(tensor, op=th.distributed.ReduceOp.MAX)
        max_num_batch = tensor[0]

    dataloader_iter = iter(dataloader)
    y = {}

    # WholeGraph does not support imbalanced batch numbers across processes/trainers
    # TODO (IN): Fix dataloader to have same number of minibatches.
    for iter_l in range(max_num_batch):
        tmp_keys = []
        if iter_l < len_dataloader:
            input_nodes, output_nodes, blocks = next(dataloader_iter)
            if not isinstance(input_nodes, dict):
                # This happens on a homogeneous graph.
                assert len(g.ntypes) == 1
                input_nodes = {g.ntypes[0]: input_nodes}
            if not isinstance(output_nodes, dict):
                # This happens on a homogeneous graph.
                assert len(g.ntypes) == 1
                output_nodes = {g.ntypes[0]: output_nodes}
            if layer_id == "0":
                tmp_keys = [ntype for ntype in g.ntypes if ntype not in input_nodes]
                # All samples should contain all the ntypes for wholegraph compatibility
                input_nodes.update({ntype: th.empty((0,), dtype=g.idtype) \
                    for ntype in tmp_keys})
        else:
            # For the last few iterations, some processes may not have mini-batches,
            # we should create empty input tensors to trigger the computation. This is
            # necessary for WholeGraph, which requires all processes to perform
            # computations in every iteration.
            input_nodes = {ntype: th.empty((0,), dtype=g.idtype) for ntype in g.ntypes}
            blocks = None
        if iter_l % 100000 == 0 and get_rank() == 0:
            logging.info("[Rank 0] dist_inference: finishes %d iterations.", iter_l)

        if task_tracker is not None:
            task_tracker.keep_alive(report_step=iter_l)

        h = get_input_embeds(input_nodes)
        if blocks is None:
            continue
        # Remove additional keys (ntypes) added for WholeGraph compatibility
        for ntype in tmp_keys:
            del input_nodes[ntype]
        block = blocks[0].to(device)
        h = layer(block, h)

        # For the first iteration, we need to create output tensors.
        if iter_l == 0:
            # Infer the hidden dim size.
            # Here we assume all node embeddings have the same dim size.
            h_dim = 0
            dtype = None
            for k in h:
                assert len(h[k].shape) == 2, \
                        "The embedding tensors should have only two dimensions."
                h_dim = h[k].shape[1]
                dtype = h[k].dtype
            assert h_dim > 0, "Cannot inference the hidden dim size."

            # Create distributed tensors to store the embeddings.
            for k in target_ntypes:
                y[k] = create_dist_tensor((g.number_of_nodes(k), h_dim),
                                          dtype=dtype, name=f'h-{layer_id}',
                                          part_policy=g.get_node_partition_policy(k),
                                          # TODO(zhengda) this makes the tensor persistent.
                                          persistent=True)

        for k in h.keys():
            # some ntypes might be in the tensor h but are not in the output nodes
            # that have empty tensors
            if k in output_nodes:
                assert k in y, "All mini-batch outputs should have the same tensor names."
                y[k][output_nodes[k]] = h[k].cpu()
    flush_data()
    return y

def dist_inference(g, gnn_encoder, get_input_embeds, batch_size, fanout,
                   edge_mask=None, task_tracker=None):
    """Distributed inference of final representation over all node types
       using layer-by-layer inference.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    gnn_encoder : GraphConvEncoder
        The GNN encoder on the graph.
    get_input_embeds : callable
        Get the node features.
    batch_size : int
        The batch size for the GNN inference.
    fanout : list of int
        The fanout for computing the GNN embeddings in a GNN layer.
    edge_mask : str
        The edge mask indicates which edges are used to compute GNN embeddings.
    task_tracker : GSTaskTrackerAbc
        The task tracker.

    Returns
    -------
    dict of Tensor : the final GNN embeddings of all nodes.
    """
    device = gnn_encoder.device
    with th.no_grad():
        next_layer_input = None
        for i, layer in enumerate(gnn_encoder.layers):
            infer_nodes = {}
            for ntype in g.ntypes:
                infer_nodes[ntype] = node_split(th.ones((g.number_of_nodes(ntype),),
                                                        dtype=th.bool),
                                                partition_book=g.get_partition_book(),
                                                ntype=ntype, force_even=False)
            # need to provide the fanout as a list, the number of layers is one obviously here
            fanout_i = [-1] if fanout is None or len(fanout) == 0 else [fanout[i]]
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout_i, mask=edge_mask)
            dataloader = dgl.dataloading.DistNodeDataLoader(g, infer_nodes, sampler,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            drop_last=False)

            if i > 0:
                def get_input_embeds1(input_nodes, node_feats):
                    return {k: node_feats[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                get_input_embeds = partial(get_input_embeds1, node_feats=next_layer_input)
            next_layer_input = dist_inference_one_layer(str(i), g, dataloader,
                                                        list(infer_nodes.keys()),
                                                        layer, get_input_embeds, device,
                                                        task_tracker)
    return next_layer_input


class HeteroGraphConv(nn.Module):
    r"""A generic module for computing convolution on heterogeneous graphs.

    Parameters
    ----------
    mods: dict[str, nn.Module]
        Modules associated with every edge types. The forward function of each
        module must have a `DGLGraph` object as the first argument, and
        its second argument is either a tensor object representing the node
        features or a pair of tensor object representing the source and destination
        node features.
    aggregate: str, callable, optional
        Method for aggregating node features generated by different relations.
        Allowed string values are 'sum', 'max', 'min', 'mean', 'stack'.
        The 'stack' aggregation is performed along the second dimension, whose order
        is deterministic.
        User can also customize the aggregator by providing a callable instance.
        For example, aggregation by summation is equivalent to the follows:

    Attributes
    ----------
    mods: dict[str, nn.Module]
        Modules associated with every edge types.
    """

    def __init__(self, mods, aggregate="sum"):
        super(HeteroGraphConv, self).__init__()
        self.mod_dict = mods
        mods = {str(k): v for k, v in mods.items()}
        # Register as child modules
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(
                v, "set_allow_zero_in_degree", None
            )
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def _get_module(self, etype):
        mod = self.mod_dict.get(etype, None)
        if mod is not None:
            return mod
        if isinstance(etype, tuple):
            # etype is canonical
            _, etype, _ = etype
            return self.mod_dict[etype]
        raise KeyError("Cannot find module with edge type %s" % etype)

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        .. versionchanged:: 0.4.1
            Modify the argument inputs to accept a tuple of dict[str, Tensor] to support
            edge features in graph convolution.

        Parameters
        ----------
        g: DGLGraph
            Graph data.
        inputs: dict[str, Tensor] or tuple of dict[str, Tensor]
            Input node features, and edge feature if provided.
        mod_args: dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs: dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple) and len(inputs)==3:
                src_inputs, dst_inputs, edge_inputs = inputs
            elif isinstance(inputs, tuple) and len(inputs)==2:
                src_inputs, dst_inputs = inputs
                edge_inputs = {}
            else:
                src_inputs = inputs
                dst_inputs = {
                    k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                }
                edge_inputs = {}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                # check if the edge type has inputs
                if (stype, etype, dtype) in edge_inputs:
                    dstdata = self._get_module((stype, etype, dtype))(
                        rel_graph,
                        (src_inputs[stype], dst_inputs[dtype], edge_inputs[(stype, etype, dtype)]),
                        *mod_args.get((stype, etype, dtype), ()),
                        **mod_kwargs.get((stype, etype, dtype), {})
                    )
                else:
                    dstdata = self._get_module((stype, etype, dtype))(
                        rel_graph,
                        (src_inputs[stype], dst_inputs[dtype]),
                        *mod_args.get((stype, etype, dtype), ()),
                        **mod_kwargs.get((stype, etype, dtype), {})
                    )
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self._get_module((stype, etype, dtype))(
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get((stype, etype, dtype), ()),
                    **mod_kwargs.get((stype, etype, dtype), {})
                )
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts

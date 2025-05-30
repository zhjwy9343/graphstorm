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

    Utility functions for evaluation
"""
import math
import torch as th

from ..utils import get_backend, is_distributed
from ..data.utils import alltoallv_cpu, alltoallv_nccl

def calc_distmult_pos_score(h_emb, t_emb, r_emb, device=None):
    """ Calculate DistMulti Score for positive pairs

        score = sum(head_emb * relation_emb * tail_emb)

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding
        t_emb: th.Tensor
            Tail node embedding
        r_emb: th.Tensor
            Relation type embedding

        Return
        ------
        Distmult score: th.Tensor
    """
    # DistMult
    if device is not None:
        r_emb = r_emb.to(device)
        h_emb = h_emb.to(device)
        t_emb = t_emb.to(device)

    score = th.sum(h_emb * r_emb * t_emb, dim=-1)
    return score

def calc_distmult_neg_tail_score(heads, tails, r_emb, num_chunks, chunk_size,
    neg_sample_size, device=None):
    """ Calculate DistMulti Score for negative pairs when tail nodes are negative

        score = sum(head_emb * relation_emb * tail_emb)

        Parameters
        ----------
        heads: th.Tensor
            Head node embedding
        tails: th.Tensor
            Tail node embedding
        r_emb: th.Tensor
            Relation type embedding
        num_chunks: int
            Number of shared negative chunks
        chunk_size: int
            Chunk size
        neg_sample_size: int
            Number of negative samples for each positive node
        device: th.device
            Device to run the computation

        Return
        ------
        Distmult score: th.Tensor
    """
    hidden_dim = heads.shape[1]
    r = r_emb

    if device is not None:
        r = r.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
    tails = th.transpose(tails, 1, 2)
    tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, tails)

def calc_distmult_neg_head_score(heads, tails, r_emb, num_chunks, chunk_size,
    neg_sample_size, device=None):
    """ Calculate DistMulti Score for negative pairs when head nodes are negative

        score = sum(head_emb * relation_emb * tail_emb)

        Parameters
        ----------
        heads: th.Tensor
            Head node embedding
        tails: th.Tensor
            Tail node embedding
        r_emb: th.Tensor
            Relation type embedding
        num_chunks: int
            Number of shared negative chunks
        chunk_size: int
            Chunk size
        neg_sample_size: int
            Number of negative samples for each positive node
        device: th.device
            Device to run the computation

        Return
        ------
        Distmult score: th.Tensor
    """
    hidden_dim = tails.shape[1]
    r = r_emb
    if device is not None:
        r = r.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
    heads = th.transpose(heads, 1, 2)
    tmp = (tails * r).reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, heads)

def calc_dot_pos_score(h_emb, t_emb):
    """ Calculate Dot product Score for positive pairs

        score = sum(head_emb * tail_emb)

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding
        t_emb: th.Tensor
            Tail node embedding

        Returns
        -------
        Dot product score: th.Tensor
    """
    # DistMult
    score = th.sum(h_emb * t_emb, dim=-1)
    return score

def calc_dot_neg_tail_score(heads, tails, num_chunks, chunk_size,
    neg_sample_size, device=None):
    """ Calculate Dot product Score for negative pairs when tail nodes are negative

        score = sum(head_emb * tail_emb)

        Parameters
        ----------
        heads: th.Tensor
            Head node embedding
        tails: th.Tensor
            Tail node embedding
        num_chunks: int
            Number of shared negative chunks
        chunk_size: int
            Chunk size
        neg_sample_size: int
            Number of negative samples for each positive node
        device: th.device
            Device to run the computation

        Returns
        -------
        Dot product score: th.Tensor
    """
    hidden_dim = heads.shape[1]

    if device is not None:
        heads = heads.to(device)
        tails = tails.to(device)
    tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
    tails = th.transpose(tails, 1, 2)
    tmp = heads.reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, tails)

def calc_dot_neg_head_score(heads, tails, num_chunks, chunk_size,
    neg_sample_size, device=None):
    """ Calculate Dot product Score for negative pairs when head nodes are negative

        score = sum(head_emb * tail_emb)

        Parameters
        ----------
        heads: th.Tensor
            Head node embedding
        tails: th.Tensor
            Tail node embedding
        num_chunks: int
            Number of shared negative chunks
        chunk_size: int
            Chunk size
        neg_sample_size: int
            Number of negative samples for each positive node
        device: th.device
            Device to run the computation

        Returns
        -------
        Dot product score: th.Tensor
    """
    hidden_dim = tails.shape[1]
    if device is not None:
        heads = heads.to(device)
        tails = tails.to(device)
    heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
    heads = th.transpose(heads, 1, 2)
    tmp = tails.reshape(num_chunks, chunk_size, hidden_dim)
    return th.bmm(tmp, heads)

def calc_rotate_pos_score(h_emb, t_emb, r_emb, rel_emb_init, gamma, device=None):
    r""" Calculate RotatE Score for positive pairs

        Score function of RotateE measures the angular distance between
        head and tail elements. The angular distance is defined as:

        .. math::

            d_r(h, t)=\|h\circ r-t\|

        The RotatE score function is defined as:

        .. math::

            gamma - \|h\circ r-t\|^2

        where gamma is a margin.

        For more detials please refer to https://arxiv.org/abs/1902.10197
        or https://dglke.dgl.ai/doc/kg.html#rotatee.

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding.
        t_emb: th.Tensor
            Tail node embedding.
        r_emb: th.Tensor
            Relation type embedding.
        rel_emb_init: float
            The initial value used to bound the relation embedding initialization.
        gamma: float
            The gamma value used for shifting the optimization target.
        device: th.device
            Device to run the computation.

        Return
        ------
        rotate_score: th.Tensor
            The RotatE score.
    """
    if device is not None:
        r_emb = r_emb.to(device)
        h_emb = h_emb.to(device)
        t_emb = t_emb.to(device)

    real_head, imag_head = th.chunk(h_emb, 2, dim=-1)
    real_tail, imag_tail = th.chunk(t_emb, 2, dim=-1)

    phase_rel = r_emb / (rel_emb_init / th.tensor(math.pi))
    real_rel, imag_rel = th.cos(phase_rel), th.sin(phase_rel)
    real_score = real_head * real_rel - imag_head * imag_rel
    imag_score = real_head * imag_rel + imag_head * real_rel
    real_score = real_score - real_tail
    imag_score = imag_score - imag_tail
    score = th.stack([real_score, imag_score], dim=0)
    score = score.norm(dim=0)

    rotate_score = gamma - score.sum(-1)
    return rotate_score

def calc_rotate_neg_head_score(heads, tails, r_emb, num_chunks,
                               chunk_size, neg_sample_size,
                               rel_emb_init, gamma,
                               device=None):
    """ Calculate RotatE Score for negative pairs when head nodes are negative.

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding.
        t_emb: th.Tensor
            Tail node embedding.
        r_emb: th.Tensor
            Relation type embedding.
        num_chunks: int
            Number of shared negative chunks.
        chunk_size: int
            Chunk size.
        neg_sample_size: int
            Number of negative samples for each positive node.
        rel_emb_init: float
            The initial value used to bound the relation embedding initialization.
        gamma: float
            The gamma value used for shifting the optimization target.
        device: th.device
            Device to run the computation.

        Return
        ------
        rotate_score: th.Tensor
            The RotatE score.
    """
    if device is not None:
        r_emb = r_emb.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    hidden_dim = heads.shape[1]
    emb_real, emb_imag = th.chunk(tails, 2, dim=-1)

    phase_rel = r_emb / (rel_emb_init / th.tensor(math.pi))
    rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
    # Rotate tail embeddings to head embeding space
    real = emb_real * rel_real + emb_imag * rel_imag
    imag = -emb_real * rel_imag + emb_imag * rel_real

    emb_complex = th.cat((real, imag), dim=-1)
    tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
    heads = heads.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
    score = tmp - heads
    score = th.stack([score[..., :hidden_dim // 2],
                      score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)
    rotate_score = gamma - score.sum(-1)
    return rotate_score

def calc_rotate_neg_tail_score(heads, tails, r_emb, num_chunks,
                               chunk_size, neg_sample_size,
                               rel_emb_init, gamma,
                               device=None):
    """ Calculate RotatE Score for negative pairs when tail nodes are negative.

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding.
        t_emb: th.Tensor
            Tail node embedding.
        r_emb: th.Tensor
            Relation type embedding.
        num_chunks: int
            Number of shared negative chunks.
        chunk_size: int
            Chunk size.
        neg_sample_size: int
            Number of negative samples for each positive node.
        rel_emb_init: float
            The initial value used to bound the relation embedding initialization.
        gamma: float
            The gamma value used for shifting the optimization target.
        device: th.device
            Device to run the computation.

        Return
        ------
        rotate_score: th.Tensor
            The RotatE score.
    """
    if device is not None:
        r_emb = r_emb.to(device)
        heads = heads.to(device)
        tails = tails.to(device)
    hidden_dim = heads.shape[1]
    emb_real, emb_imag = th.chunk(heads, 2, dim=-1)

    phase_rel = r_emb / (rel_emb_init / th.tensor(math.pi))
    rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
    # Rotate head embeddings to tail embeding space
    real = emb_real * rel_real - emb_imag * rel_imag
    imag = emb_real * rel_imag + emb_imag * rel_real

    emb_complex = th.cat((real, imag), dim=-1)
    tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
    tails = tails.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
    score = tmp - tails
    score = th.stack([score[..., :hidden_dim // 2],
                      score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)

    rotate_score = gamma - score.sum(-1)
    return rotate_score

def calc_transe_pos_score(h_emb, t_emb, r_emb, gamma, norm='l2', device=None):
    r""" Calculate TransE Score for positive pairs

        Score function of TransE measures the angular distance between
        head and tail elements. The angular distance is defined as:

        .. math::

            d_r(h, t)= -\|h+r-t\|

        The TransE score function is defined as:

        .. math::

            gamma - \|h+r-t\|^{frac{1}{2}} \text{or} gamma - \|h+r-t\|

        where gamma is a margin.

        For more details, please refer to
        https://papers.nips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html
        or https://dglke.dgl.ai/doc/kg.html#transe.

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding.
        t_emb: th.Tensor
            Tail node embedding.
        r_emb: th.Tensor
            Relation type embedding.
        gamma: float
            The gamma value used for shifting the optimization target.
        norm: str
            L1 or L2 norm on the angular distance.
        device: th.device
            Device to run the computation.

        Return
        ------
        transe_score: th.Tensor
            The TransE score.
    """
    if device is not None:
        r_emb = r_emb.to(device)
        h_emb = h_emb.to(device)
        t_emb = t_emb.to(device)

    score = (h_emb + r_emb) - t_emb

    if norm == 'l1':
        transe_score = gamma - th.norm(score, p=1, dim=-1)
    elif norm == 'l2':
        transe_score = gamma - th.norm(score, p=2, dim=-1)
    else:
        raise ValueError("Unknown norm on the angular distance. Only support L1 and L2.")
    return transe_score

def calc_transe_neg_head_score(h_emb, t_emb, r_emb, num_chunks,
                               chunk_size, neg_sample_size,
                               gamma, norm='l2',
                               device=None):
    """ Calculate TransE Score for negative pairs when head nodes are negative.

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding.
        t_emb: th.Tensor
            Tail node embedding.
        r_emb: th.Tensor
            Relation type embedding.
        num_chunks: int
            Number of shared negative chunks.
        chunk_size: int
            Chunk size.
        neg_sample_size: int
            Number of negative samples for each positive node.
        gamma: float
            The gamma value used for shifting the optimization target.
        norm: str
            L1 or L2 norm on the angular distance.
        device: th.device
            Device to run the computation.

        Return
        ------
        transe_score: th.Tensor
            The TransE score.
    """
    if device is not None:
        r_emb = r_emb.to(device)
        h_emb = h_emb.to(device)
        t_emb = t_emb.to(device)

    hidden_dim = h_emb.shape[1]
    h_emb = h_emb.reshape(num_chunks, neg_sample_size, hidden_dim)
    t_emb = t_emb - r_emb
    t_emb = t_emb.reshape(num_chunks, chunk_size, hidden_dim)

    if norm == 'l1':
        transe_score = gamma - th.cdist(t_emb, h_emb, p=1)
    elif norm == 'l2':
        transe_score = gamma - th.cdist(t_emb, h_emb, p=2)
    else:
        raise ValueError("Unknown norm on the angular distance. Only support L1 and L2.")
    return transe_score

def calc_transe_neg_tail_score(h_emb, t_emb, r_emb, num_chunks,
                               chunk_size, neg_sample_size,
                               gamma, norm='l2',
                               device=None):
    """ Calculate TransE Score for negative pairs when tail nodes are negative.

        Parameters
        ----------
        h_emb: th.Tensor
            Head node embedding.
        t_emb: th.Tensor
            Tail node embedding.
        r_emb: th.Tensor
            Relation type embedding.
        num_chunks: int
            Number of shared negative chunks.
        chunk_size: int
            Chunk size.
        neg_sample_size: int
            Number of negative samples for each positive node.
        gamma: float
            The gamma value used for shifting the optimization target.
        norm: str
            L1 or L2 norm on the angular distance.
        device: th.device
            Device to run the computation.

        Return
        ------
        transe_score: th.Tensor
            The TransE score.
    """
    if device is not None:
        r_emb = r_emb.to(device)
        h_emb = h_emb.to(device)
        t_emb = t_emb.to(device)

    hidden_dim = h_emb.shape[1]
    h_emb = h_emb + r_emb
    h_emb = h_emb.reshape(num_chunks, chunk_size, hidden_dim)
    t_emb = t_emb.reshape(num_chunks, neg_sample_size, hidden_dim)

    if norm == 'l1':
        transe_score = gamma - th.cdist(h_emb, t_emb, p=1)
    elif norm == 'l2':
        transe_score = gamma - th.cdist(h_emb, t_emb, p=2)
    else:
        raise ValueError("Unknown norm on the angular distance. Only support L1 and L2.")
    return transe_score

def calc_ranking(pos_score, neg_score):
    """ Calculate ranking of positive scores among negative scores

        Parameters
        ----------
        pos_score: torch.Tensor
            positive scores
        neg_score: torch.Tensor
            negative screos

        Returns
        -------
        ranking of positive scores: th.Tensor
    """
    pos_score = pos_score.view(-1, 1)
    # perturb object
    scores = th.cat([pos_score, neg_score], dim=1)
    scores = th.sigmoid(scores)
    _, indices = th.sort(scores, dim=1, descending=True)
    indices = th.nonzero(indices == 0)
    rankings = indices[:, 1].view(-1) + 1
    rankings = rankings.detach()
    if is_distributed() and get_backend() == "gloo":
        rankings = rankings.cpu() # Save GPU memory

    return rankings

def gen_lp_score(ranking):
    """ Get link prediction metrics

        Parameters
        ----------
        ranking:
            ranking of each positive edge

        Returns
        -------
        link prediction eval metrics: list of dict
    """
    logs = []
    for rank in ranking:
        logs.append({
            'mrr': 1.0 / rank,
            'mr': float(rank),
            'hits@1': 1.0 if rank <= 1 else 0.0,
            'hits@3': 1.0 if rank <= 3 else 0.0,
            'hits@10': 1.0 if rank <= 10 else 0.0
        })
    metrics = {}
    for metric in logs[0]:
        metrics[metric] = th.tensor(sum(log[metric] for log in logs) / len(logs))
    return metrics

def gen_mrr_score(ranking):
    """ Get link prediction mrr metrics

        Parameters
        ----------
        ranking:
            ranking of each positive edge

        Returns
        -------
        link prediction eval metrics: list of dict
    """
    logs = th.div(1.0, ranking)
    metrics = {"mrr": th.tensor(th.div(th.sum(logs),len(logs)))}
    return metrics


def broadcast_data(rank, world_size, data_tensor):
    """ Broadcast local data to all trainers in the cluster using all2all

        After broadcast_data, each trainer will get all the data (data_tensor)

        Parameters
        ----------
        rank : int
            The rank of current worker
        world_size : int
            The size of the entire
        data_tensor:
            Data to exchange
    """
    if world_size == 1: # world size is 1, nothing to do
        return data_tensor

    # exchange the data size of each trainer
    if get_backend() == "gloo":
        device = "cpu"
    elif get_backend() == "nccl":
        data_tensor = data_tensor.cuda()
        device = data_tensor.device
    else:
        assert False, f"backend {get_backend()} not supported."

    data_size = th.zeros((world_size,), dtype=th.int64, device=device)
    data_size[rank] = data_tensor.shape[0]
    th.distributed.all_reduce(data_size,
        op=th.distributed.ReduceOp.SUM)

    gather_list = [th.empty([int(size)]+list(data_tensor.shape[1:]),
        dtype=data_tensor.dtype,
        device=device) for size in data_size]
    data_tensors = [data_tensor for _ in data_size]
    if get_backend() == "gloo":
        alltoallv_cpu(rank, world_size, gather_list, data_tensors)
    else: #get_backend() == "nccl"
        alltoallv_nccl(gather_list, data_tensors)

    data_tensor = th.cat(gather_list, dim=0)
    return data_tensor

def is_float(val_str):
    """ Check if the given string is a valid float number.
    
    The required string format should be all digits with at most 1 period.
    """
    digit_str = val_str.replace('.', '')
    period_cnt = val_str.count('.')
    return digit_str.isnumeric() and period_cnt <= 1

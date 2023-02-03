# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-03 11:10:18
# @Filepath: scripts/random_walk.py

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor

from torch_geometric.utils.num_nodes import maybe_num_nodes

try:
    import torch_cluster
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    print('`torch_cluster` is not installed. Please install it via `pip install torch-cluster`.')
    
class RandomWalk(object):
    def __init__(self, context_size: int, walk_length: int, p: float=1, q: float=1): 
        
        self.context_size = context_size
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.adj = None
        
    def fit(self, edge_index: Tensor):
        if self.adj is None:
            print('Fitting the model')
        else:
            print("removing old adj matrix!")
        row, col = edge_index
        self.N = maybe_num_nodes(edge_index)
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to('cpu')
        return self

    def transform(self, batch_size=64) -> Tensor:
        start = torch.randint(self.N, (batch_size,))
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr, col, self.batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]
        
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)
    
    def save(self, path):
        torch.save(self.adj, path)
        

import sys, os
residual2vec_ = '../../residual2vec_'
sys.path.insert(0, residual2vec_)

from utils import snakemake_utils
dataset = snakemake_utils.get_dataset("polbook")
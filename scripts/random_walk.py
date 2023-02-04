# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-02-03 11:10:18
# @Filepath: scripts/random_walk.py

import sys, os
residual2vec_ = '../../residual2vec_'
sys.path.insert(0, residual2vec_)

from utils import snakemake_utils
import getopt

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
        N = maybe_num_nodes(edge_index)
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to('cpu')
        self.N = N
        return self

    def transform(self, batch_size=64) -> Tensor:
        start = torch.randint(self.N, (batch_size,))
        rowptr, col, _ = self.adj.csr()
        rw = random_walk(rowptr,col, start, self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]
        
        return rw
    
def save_text(path, tensor, format="%d"):
    assert isinstance(tensor, Tensor)
    import numpy as np
        
    np.savetxt(path, tensor.numpy(), fmt='%d')


# take dataset as an argument here

def main(argv):

    print(argv)
    try:
        # take dataset as an argument here

        opts, args = getopt.getopt(argv,"hd:o:n:",["dataset=","output=", "num="])
        dataset = 'polbook'
        output = 'polbook.txt'
        num_sentences = None
        for opt, arg in opts:
            print(opt, arg)
            if opt in ['-d', '--dataset']:
                dataset = arg
                print('dataset: ', dataset)
            if opt in ['-o', '--output']:
                output = arg
                print('output: ', output)
            if opt in ['-n', '--num']:
                num_sentences = int(arg)
                print('num_sentences: ', num_sentences)
    except getopt.GetoptError:
        print('test.py -d <dataset> -o <output> -n <num_sentences>')
        sys.exit(2)

    
    edge_index = snakemake_utils.get_dataset(dataset).edge_index
    rw = RandomWalk(context_size=10, walk_length=40).fit(edge_index)
    num_sentences = rw.N * 100 if num_sentences is None else num_sentences
    save_text(output, rw.transform(num_sentences))

if __name__ == "__main__":
    main(sys.argv[1:])

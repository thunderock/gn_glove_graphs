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
import numpy as np
from tqdm import tqdm

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
    
def save_text(path, array, format="%d"):
    assert isinstance(array, np.ndarray)
    np.savetxt(path, array, fmt='%d')

def generate_man_woman_words(dataset, pivot_count_ratio=.1):
    # step 1: pick the two most popular classes
    # step 2: pick pivot_count_ratio * .5 nodes of those classes which are most connected to the same class
    y = dataset.get_grouped_col().numpy()
    edge_index = dataset.edge_index.numpy()
    K, counts = np.unique(y, return_counts=True)
    K = K[np.argsort(counts)[::-1]][:2]
    num_words = int(pivot_count_ratio * .5 * y.shape[0])
    ret = np.zeros((num_words, 2), dtype=np.int32)
    for i, k in enumerate(K):
        idx = np.where(y == k)[0]
        # all other nodes
        other_idx = np.where(y != k)[0]
        # remove edges which have other nodes
        mask = np.isin(edge_index[0], idx) & np.isin(edge_index[1], idx)
        edge_index_ = edge_index[:, mask].flatten()
        # count number of edges for each node

        u, counts = np.unique(edge_index_, return_counts=True)
        # sort by counts
        u = u[np.argsort(counts)[::-1]][:num_words]

        ret[:, i] = u
    return ret[:, 0], ret[:, 1]





# take dataset as an argument here

def main(argv):

    print(argv)
    try:
        # take dataset as an argument here

        opts, args = getopt.getopt(argv,"hd:o:n:m:w:",["dataset=","output=", "num=", "man=", "woman="])
        dataset = 'polbook'
        output = 'polbook.txt'
        num_sentences = None
        man_file = None
        woman_file = None
        for opt, arg in opts:
            print(opt, arg)
            if opt in ['-d', '--dataset']:
                dataset = arg
            if opt in ['-o', '--output']:
                output = arg
            if opt in ['-n', '--num']:
                num_sentences = int(arg)
            if opt in ['-m', '--man']:
                man_file = arg
            if opt in ['-w', '--woman']:
                woman_file = arg
    except getopt.GetoptError:
        print('test.py -d <dataset> -o <output> -n <num_sentences> -m <man> -w <woman>')
        sys.exit(2)

    
    print("generating and writing random walks to file: ", output)
    dataset = snakemake_utils.get_dataset(dataset)
    rw = RandomWalk(context_size=10, walk_length=40).fit(dataset.edge_index.to('cpu'))
    num_sentences = rw.N * 100 if num_sentences is None else num_sentences
    save_text(output, rw.transform(num_sentences).numpy())

    print("generating and writing man and woman words to file: {} and {}".format(man_file, woman_file))
    man, woman = generate_man_woman_words(dataset)
    save_text(man_file, man)
    save_text(woman_file, woman)


if __name__ == "__main__":
    main(sys.argv[1:])

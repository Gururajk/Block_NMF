import spectral.io.envi as envi
import spectral.io.aviris as aviris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from ristretto.nmfg import compute_nmf, compute_rnmf



def Block_NMF(Y,n_components,max_iter):
    p = Y.shape[0]
    k = Y.shape[1]
    q = 50000
    l = int(p / q)
    block_Y_list = []
    for i in range(l):
        block_Y_list.append(Y[i*q:(i+1)*q,:])
    block_Y_list.append(Y[l*q:,:])
    _, H = compute_nmf(block_Y_list[0],rank=n_components,init='nndsvda',maxiter=5000,tol=1e-8)
    for iters in range(max_iter):
        block_W_list = []
        for i in range(l):
            W, H = compute_nmf(block_Y_list[i],rank=n_components,init='hinit',maxiter=5000,tol=1e-8,H_init=H)
            block_W_list.append(W)
    return block_W_list,H

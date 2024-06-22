import numpy as np
import os.path as osp
import gudhi as gd
import networkx as nx
import pandas as pd
from torch_geometric.datasets import TUDataset
from sklearn.metrics.pairwise import cosine_similarity as cos
from scipy.spatial.distance import pdist, squareform


def apply_graph_extended_persistence(A, filtration_val):
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]
    dgmOrd0 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmOrd0 if p[0] == 0]) if len(
        dgmOrd0) else np.empty([0, 2])
    dgmRel1 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmRel1 if p[0] == 1]) if len(
        dgmRel1) else np.empty([0, 2])
    dgmExt0 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt0 if p[0] == 0]) if len(
        dgmExt0) else np.empty([0, 2])
    dgmExt1 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt1 if p[0] == 1]) if len(
        dgmExt1) else np.empty([0, 2])
    final_dgm = np.concatenate([dgmOrd0, dgmExt0, dgmRel1, dgmExt1], axis=0)
    return final_dgm

def persistence_images(dgm, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1.):
    PXs, PYs = dgm[:, 0], dgm[:, 1]
    if PXs.shape[0]==0 and PYs.shape[0]==0:
        norm_output = np.zeros((resolution)) + 1e-6
    else:
        xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
        x = np.linspace(xm, xM, resolution[0])
        y = np.linspace(ym, yM, resolution[1])
        X, Y = np.meshgrid(x, y)
        Zfinal = np.zeros(X.shape)
        X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

        # Compute persistence image
        P0, P1 = np.reshape(dgm[:, 0], [1, 1, -1]), np.reshape(dgm[:, 1], [1, 1, -1])
        weight = np.abs(P1 - P0)
        distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

        if return_raw:
            lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
            lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
        else:
            weight = weight ** power
            Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

        output = [lw, lsum] if return_raw else Zfinal

        if normalization:
            norm_output = (output - np.min(output)) / (np.max(output) - np.min(output))
        else:
            norm_output = output

        if np.sum(np.isnan(norm_output))>0:
            norm_output = np.zeros((resolution)) + 1e-6
    return norm_output

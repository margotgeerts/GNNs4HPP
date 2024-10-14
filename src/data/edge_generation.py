import torch
import numpy as np
import scipy as sp
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import OPTICS
from libpysal import weights
import itertools
import pandas as pd


def knn_edges(lat_lon_rad, n_neighbors):
    tree = BallTree(lat_lon_rad, metric='haversine')
    row = [[i]*n_neighbors for i in range(lat_lon_rad.shape[0])]
    col = tree.query(lat_lon_rad, k=n_neighbors, return_distance=False)
    row = np.array(row).flatten()
    col = col.flatten()
    return torch.tensor([row, col], dtype=torch.long)

def minmax_edges(lat_lon_rad, cutoff):
    tree = BallTree(lat_lon_rad, metric='haversine')
    k = lat_lon_rad.shape[0]
    distances, _ = tree.query(lat_lon_rad, k=k, return_distance=True)
    del tree
    min_dist = np.min(distances[distances > 0]) 
    max_dist = np.max(distances)
    distances = 0.98 - (distances - min_dist) / (max_dist - min_dist)
    distances[distances < kwargs['cutoff']] = 0
    return torch.tensor(distances.nonzero(), dtype=torch.long)

def gaussian_edges(lat_lon_rad, cutoff):
    tree = BallTree(lat_lon_rad, metric='haversine')
    k = lat_lon_rad.shape[0]
    distances, _ = tree.query(lat_lon_rad, k=k, return_distance=True)
    del tree
    std = np.std(distances)
    adj = np.exp(-distances**2 / std**2)
    adj[adj < kwargs['cutoff']] = 0
    return torch.tensor(adj.nonzero(), dtype=torch.long)

def rng_edges(lat_lon):
    w = weights.Relative_Neighborhood(lat_lon)
    adj = utils.from_scipy_sparse_matrix(w.sparse)
    return adj[0]

def gabriel_edges(lat_lon):
    tri = sp.spatial.Delaunay(lat_lon)
    simplices = tri.simplices
    edges = []
    for i in simplices:
        for j in range(0,3):
            for k in range(0,3):
                if j != k:
                    edges.append((i[j],i[k]))
    new_df = pd.DataFrame(edges).drop_duplicates().sort_values([0, 1]).groupby(0)[1].apply(list).to_dict()
    del edges

    lil = sp.sparse.lil_matrix((tri.npoints, tri.npoints))
    indices, indptr = tri.vertex_neighbor_vertices
    for k in range(tri.npoints):
        lil.rows[k] = indptr[indices[k]:indices[k+1]].tolist()
        lil.data[k] = np.ones_like(lil.rows[k]).tolist()  # dummy data of same shape as row
    del indices
    del indptr
    coo = lil.tocoo()
    del lil
    conns = np.vstack((coo.row, coo.col)).T
    del coo
    
    delaunay_conns = np.sort(conns, axis=1)
    del conns
    
    c = tri.points[delaunay_conns]
    m = (c[:, 0, :] + c[:, 1, :])/2
    r = np.sqrt(np.sum((c[:, 0, :] - c[:, 1, :])**2, axis=1))/2
    del c
    tree = BallTree(lat_lon)
    n = tree.query(X=m, k=1)[0].reshape(-1)
    del tree
    del m
    g = n >= r*(0.999)  # The factor is to avoid precision errors in the distances
    del n
    del r
    return torch.tensor(delaunay_conns[g], dtype=torch.long).T

def optics_edges(lat_lon_rad):
    tree = BallTree(lat_lon_rad, metric='haversine')
    k = lat_lon_rad.shape[0]
    distances, _ = tree.query(lat_lon_rad, k=k, return_distance=True)
    del tree
    min_samples = kwargs['min_samples'] #if 'min_samples' in kwargs else 5
    optics_clustering = OPTICS(min_samples=min_samples, metric='precomputed')
    y_db = optics_clustering.fit_predict(distances)
    print(y_db.shape)
    print(f"number of clusters = {np.unique(y_db)}")
    print(f'number of points in no cluster = { lat_lon_rad[y_db == -1].shape[0]}')
    
    row = []
    col = []
    for i,j in enumerate(y_db):
        for k,l in enumerate(y_db):
            # if j == l:
            if j == l and j != -1 and l != -1:
            
                row.append(i)
                col.append(k)
    return torch.tensor([row, col], dtype=torch.long)

def kmeans_edges(lat_lon, k):
    # k is not the number of clusters, but the number of points in each cluster (avg node degree)
    num_clusters = int(len(lat_lon) / k) if k >= 1 else int(len(lat_lon) / 5) 
    centroids, mean_dist = sp.cluster.vq.kmeans(lat_lon, num_clusters, seed=1)
    clusters, dist = sp.cluster.vq.vq(lat_lon, centroids)
    row = []
    col = []
    for i,j in enumerate(clusters):
        for k,l in enumerate(clusters):
            if j == l and j != -1 and l != -1:
                row.append(i)
                col.append(k)
    return torch.tensor([row, col], dtype=torch.long)

def random_edges(lat_lon, edge_prob):
    num_nodes = lat_lon.shape[0]
    edges = []
    for e in itertools.combinations(range(num_nodes), 2):
        if np.random.rand() < edge_prob:
            edges.append(e)
    return torch.tensor(edges, dtype=torch.long).T

def generate_edges(pos, generator, kwargs):    
    lat_lon = np.array([[x[1], x[0]] for x in pos])
    lat_lon_rad = np.deg2rad(lat_lon) 

    if generator == 'knn':
        assert 'n_neighbors' in kwargs.keys()
        return knn_edges(lat_lon_rad, kwargs['n_neighbors'])
        
    
    elif generator == 'minmax':
        assert 'cutoff' in kwargs.keys()
        return minmax_edges(lat_lon_rad, kwargs['cutoff'])
        
    
    elif generator == 'gaussian':
        assert 'cutoff' in kwargs.keys()
        return gaussian_edges(lat_lon_rad, kwargs['cutoff'])

    elif generator == 'relative_neighborhood':
        return rng_edges(lat_lon)
        

    elif generator == 'gabriel':
        return gabriel_edges(lat_lon)

    elif generator == 'optics':
        assert 'min_samples' in kwargs.keys()
        return optics_edges(lat_lon_rad)
    
    elif generator == 'kmeans':
        assert 'k' in kwargs.keys()
        # k is the average node degree
        return kmeans_edges(lat_lon, kwargs['k'])
    
    elif generator == 'random':
        assert 'edge_prob' in kwargs.keys()
        edge_prob = kwargs['edge_prob'] if 'edge_prob' in kwargs else 0.00002
        return random_edges(lat_lon, edge_prob)

    elif generator == 'unconnected':
        num_nodes = lat_lon.shape[0]
        edge_index = torch.tensor([], dtype=torch.long)
        edge_index,_ = utils.add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index
    
    else:
        raise NotImplementedError
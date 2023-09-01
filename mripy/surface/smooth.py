#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from joblib import Parallel, delayed
from tqdm import tqdm
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
from .. import utils, io, dependency
from .surface import create_surf_patch


class ScalarValuedMesh(utils.Savable):
    def __init__(self, verts, faces, scalars, values=None):
        self.verts = verts    # List of vertices, e.g., [(x1,y1,z1), (x2,y2,z2), ...]
        self.faces = faces    # List of triangular faces, e.g., [(v1,v2,v3), (v4,v5,v6), ...]
        self.scalars = scalars    # List of scalar values associate with each vertex, e.g., [0, 1, 0, 0, 1, ...]
        assert(len(self.scalars)==len(self.verts)) # Sanity check
        self.values = np.unique(self.scalars) if values is None else values # Values to consider
        self._nbhd = dict()    # Neighborhood
        self._dist = dict()    # Shortest distance
        self._cc = dict()    # Connected component
        self._cdist = None    # Distance matrix for each component cdist[value,comp_idx] = dist[N,N]

    def get_neighborhood(self, value):
        '''
        Get the neighborhood dict, return cache if exists.
        '''
        if value not in self._nbhd:
            self._nbhd[value] = self.compute_neighborhood(value)
        return self._nbhd[value]

    def compute_neighborhood(self, value, order=1):
        '''
        Compute the set of neighboring vertices for each vertex as a dict.

        Note that: 
        1) The index for the vertices may not be continuous because 
           here we only care for the vertices associated with the given value.
        2) Only vertices with the same value are considered valid neighbors.
        '''
        nbhd = {k: set() for k, scalar in enumerate(self.scalars) if scalar == value}
        for face in self.faces:
            # Neighbors of a vertex may not have the same value as the vertex.
            # for k in range(3):
            #     if self.scalars[face[k]] == value:
            #         nbhd[face[k]].update([face[(k + 1) % 3], face[(k + 2) % 3]])
            # Only vertices with the same value are considered valid neighbors
            vertices = set([v for v in face if self.scalars[v] == value])
            for v in vertices:
                nbhd[v].update(vertices - {v})
        for ord in range(order-1):
            prev_nbhd = deepcopy(nbhd)
            for k, nb in prev_nbhd.items():
                for v in nb:
                    nbhd[k].update(prev_nbhd[v])
        return nbhd

    def compute_connected_components(self, value):
        '''
        Find connected components associated with the given value on the mesh.
        '''
        nbhd = self.get_neighborhood(value)
        visited = set()
        components = []
        for start in nbhd:
            if start not in visited:
                # Depth First Search
                stack = [start]
                component = []
                while stack:
                    vertex = stack.pop()
                    if vertex not in visited:
                        visited.add(vertex)
                        if self.scalars[vertex] == value:
                            component.append(vertex)
                            stack.extend(nbhd[vertex] - visited)
                # Save component
                components.append(component)
        return components

    def compute_nbhd_dist(self, value):
        '''
        Compute the shortest path between vertices and their (immediate) neighbors.
        '''
        nbhd = self.get_neighborhood(value)
        dist = {}
        for k, nb in nbhd.items():
            # dist[k,k] = 0
            for v in nb: # (v,k) will also be covered
                dist[k,v] = dist.get((v,k), np.linalg.norm(self.verts[k] - self.verts[v]))
        return dist

    def compute_dist_in_components(self, n_jobs=None, gpu=False):
        '''
        Compute the shortest path between any two vertices within each connected component.
        '''
        if gpu and not dependency.has('cupy', raise_error=False):
            print(f"Use cpu multiprocessing instead. Can be slow, though...")
            gpu = False
        dists = []
        for value in self.values:
            # Get connected components corresponding to the scalar value
            if value not in self._cc:
                self._cc[value] = self.compute_connected_components(value)
            # Initialize distance between immediate neighbors
            if value not in self._dist:
                self._dist[value] = self.compute_nbhd_dist(value)
            # Run Floyd-Warshall for each component
            for cid, component in enumerate(self._cc[value]):
                N = len(component) # if N <= 2: continue
                # Construct mapping between vertex in the mesh and index in the component
                node2idx = {component[idx]: idx for idx in range(N)} # idx2node = component
                # Compute all-pairs shortest path within a component, time complexity = O(N^3)
                graph = np.zeros([N,N])
                for k in component:
                    for v in self._nbhd[value][k]: # (v,k) will also be covered
                        graph[node2idx[k],node2idx[v]] = self._dist[value][k,v]
                if not gpu:
                    # dist = floyd_warshall(csr_matrix(graph), directed=False)
                    myjob = lambda value, cid, *args, **kwargs: [value, cid, floyd_warshall(*args, **kwargs)]
                    dist = delayed(myjob)(value, cid, csr_matrix(graph), directed=False)
                else:
                    # Enjoyed a 100x acceleration on 3080 Ti over i9-10980XE @3GHz for a 8000 nodes graph
                    dist = [value, cid, gpu_floyd_warshall(graph, directed=False)]
                dists.append(dist)
        if not gpu:
            dists = Parallel(n_jobs=n_jobs)(dists)
        # Store distance matricx for each component
        cdist = {}
        for value, cid, dist in dists:
            cdist[value,cid] = dist # NxN
        # Cache the result
        self._cdist = cdist
        return cdist

    def gaussian_smooth_in_components(self, surf_data, sigma):
        '''
        Gaussian smoothing on the surface within each connected component.

        Parameters
        ----------
        surf_data : N or NxT array
        sigma : float in mm
        '''
        # Get geodesic distance matrix for each component
        if self._cdist is None:
            self.compute_dist_in_components()
        # Gaussian smoothing using matrix multiplication
        kernel = lambda d: np.exp(-0.5*d**2/sigma**2)
        smoothed = np.zeros_like(surf_data)
        for value in self.values:
            for cid, component in enumerate(self._cc[value]):
                dist = self._cdist[value,cid]
                W = kernel(dist)
                W /= W.sum(axis=1, keepdims=True)
                smoothed[component] = W @ surf_data[component]
        return smoothed


def gpu_floyd_warshall(dist, directed=False, grid=(1024,), block=(1024,)):
    # Move data to GPU
    dist = cp.array(dist, dtype=cp.float32)
    # Determine the number of vertices in the graph
    N = dist.shape[0]
    assert(dist.shape[1]==N)
    # Initialize distance matrix for Floyd-Warshall algorithm
    INF = 1e10 # Not usng np.inf to avoid 0*inf=nan; not using 3.4e38 to allow INF+INF
    dist[dist==0] = INF
    dist[np.diag_indices(N)] = 0
    # Enforce symmetry for undirected graph
    if not directed:
        dist = np.minimum(dist, dist.T)
    # Define RawKernel for parallel inner loops
    gpu_inner_loops = cp.RawKernel(r'''
        extern "C" __global__
        void gpu_inner_loops(float* dist, int N, int k) {
            // Calculates unique thread ID in the block
            int t = (blockDim.x * blockDim.y) * threadIdx.z + (threadIdx.y * blockDim.x) + threadIdx.x; 
           
            // Calculates unique block ID in the grid
            int b = (gridDim.x * gridDim.y) * blockIdx.z + (blockIdx.y * gridDim.x) + blockIdx.x;
           
            // Block size
            int T = blockDim.x * blockDim.y * blockDim.z;
           
            // Grid size
            int B = gridDim.x * gridDim.y * gridDim.z;
    
            // Independent update for each element in the distance matrix
            for (int i = b; i < N; i += B) {
                for (int j = t; j < N; j += T) {
                    float d = dist[i * N + k] + dist[k * N + j];
                    // GPUs hate branching
                    dist[i * N + j] = d * (d < dist[i * N + j]) + dist[i * N + j] * (d >= dist[i * N + j]);
                }
            }
        }
        ''', 'gpu_inner_loops')
    # Sequential outer loop
    for k in range(N):
        gpu_inner_loops(grid, block, (dist, N, k))
        cp.cuda.Device().synchronize()
    # Move data back to CPU
    return cp.asnumpy(dist)
    

class ValueSpecificSmoother(utils.Savable):
    def __init__(self, surf_mesh_file, surf_value_file, values=None, transform=None, n_jobs=None, gpu=False):
        print('>> Loading surface mesh and values...', end='')
        verts, faces = io.read_surf_mesh(surf_mesh_file)
        nodes, scalars = io.read_surf_data(surf_value_file)
        verts_patch, faces_patch = create_surf_patch(verts, faces, nodes)
        if transform is not None:
            scalars = eval(transform)
        print(f"\b\b\b: ({len(verts)} vertices, {len(faces)} faces) -> {len(nodes)} nodes in considering")
        print('>> Computing value specific surface components...')
        self.mesh = ScalarValuedMesh(verts_patch, faces_patch, scalars, values=values)
        print('>> Computing geodesic distance within components...')
        self.mesh.compute_dist_in_components(n_jobs=n_jobs, gpu=gpu)
        print('>> Done! Ready for component specific Gaussian smoothing.')

    def smooth(self, in_file, out_file, sigma):
        nodes, data = io.read_surf_data(in_file)
        smoothed = self.mesh.gaussian_smooth_in_components(data, sigma)
        io.write_surf_data(out_file, nodes, smoothed)
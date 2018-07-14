    #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import subprocess, ctypes, multiprocessing
from itertools import chain
from scipy import spatial
from os import path
import numpy as np
from . import six, afni, io, utils


def map_sequence(seq1, seq2):
    '''
    Generate a mapper from seq1 to seq2, so that seq1[mapper]==seq2,
    i.e, the mapper[k]'s node of seq1 is the k's node of seq2.
    mapper = map_sequence(v2, v1)
    mapper_inv = map_sequence(v1, v2)
    v2m = v2[mapper]
    f2m = mapper_inv[f2]
    assert(all(v2m==v1))
    assert(all(f2m[lexsort(f2m.T)]==f1[lexsort(f1.T)]))
    '''
    assert(seq1.shape[0]==seq2.shape[0])
    indexer1 = np.lexsort(seq1.T) # Don't use `indexer1 = np.argsort(seq1, axis=0)`
    indexer2 = np.lexsort(seq2.T)
    mapper = np.zeros(len(seq1), dtype=int)
    mapper[indexer2] = indexer1
    return mapper


def quadruple_mesh(verts, faces, power=1, values=[]):
    for _ in range(power):
        nv = [v for v in verts]
        nv_parent = {}
        nf = []
        def get_new_vert(n1, n2):
            n_idx = nv_parent.setdefault(tuple(sorted([n1, n2])), len(nv))
            if n_idx == len(nv):
                nv.append((verts[n1]+verts[n2])/2)
            return n_idx
        for f in faces:
            nv0 = get_new_vert(f[1], f[2])
            nv1 = get_new_vert(f[2], f[0])
            nv2 = get_new_vert(f[0], f[1])
            # nf.extend([(f[0], nv1, nv2), (nv1, f[2], nv0), (nv2, nv0, f[1]), (nv2, nv1, nv0)])
            # The triangle should always list the vertices in a counter-clockwise direction 
            # with respect to an outward pointing surface normal vector [Noah Benson]
            nf.extend([(f[0], nv2, nv1), (nv2, f[1], nv0), (nv1, nv0, f[2]), (nv0, nv1, nv2)])
        verts = nv
        faces = nf
    return np.array(verts), np.array(faces)


def immediate_neighbors(verts, faces, return_array=False):
    '''
    By default, neighbors are represented as a list of sets:
        [set([n00, n01, ...]), set([n10, n11, ...]), ...]
    If return_array=True, a numpy array with a special schema is returned
    to facilitate inter-process memory sharing in multiprocessing:
        [[n_nb0+1, n00, n01, ..., 0, 0], [n_nb1+1, n10, n11, ..., 0, 0], ...] 
    '''
    # neighbors = [np.unique(np.r_[faces[faces[:,0]==idx,1:].ravel(), 
    #     faces[faces[:,1]==idx,::2].ravel(), 
    #     faces[faces[:,2]==idx,:2].ravel()]) for idx in range(verts.shape[0])]
    # The above code is unreasonably slow...
    n_verts = verts.shape[0]
    neighbors = [set() for _ in range(n_verts)] # Don't use [set()]*n_verts
    for face in faces:
        neighbors[face[0]].update(face[1:])
        neighbors[face[1]].update(face[::2])
        neighbors[face[2]].update(face[:2])
    if return_array: # For shared memory parallelism
        n_nb = [len(nbhd) for nbhd in neighbors]
        shared_arr = multiprocessing.Array(ctypes.c_int32, n_verts*(max(n_nb)+1), lock=False)
        arr = np.frombuffer(shared_arr, dtype=np.int32).reshape(n_verts,-1)
        for idx in range(n_verts):
            arr[idx,0] = n_nb[idx] + 1
            arr[idx,1:arr[idx,0]] = list(neighbors[idx])
        neighbors = shared_arr
    return neighbors


def interp_over_mesh(verts, faces, indices, values, radius=3, neighbors=None, n_verts=None):
    '''
    By reusing precomputed neighbors (multiprocessing.Array) and n_verts, 
    one can use inter-process memory sharing to efficiently interpolate multiple dsets in parallel.
    verts and faces are unused in this case.
    '''
    if neighbors is None:
        neighbors = immediate_neighbors(verts, faces)
    if n_verts is None:
        n_verts = verts.shape[0]
    indices_set = set(indices)
    new_val = np.zeros(n_verts)
    if isinstance(neighbors, ctypes.Array): # For shared memory parallelism
        neighbors = np.frombuffer(neighbors, dtype=np.int32).reshape(n_verts,-1)
        for idx in range(n_verts):
            nbhd = set(neighbors[idx,1:neighbors[idx,0]])
            new_nbhd = nbhd
            for _ in range(1, radius):
                ext_nbhd = set()
                for nb in new_nbhd:
                    ext_nbhd.update(neighbors[nb,1:neighbors[nb,0]])
                new_nbhd = ext_nbhd.difference(nbhd)
                nbhd.update(ext_nbhd)
            new_val[idx] = np.mean([values[nb] for nb in nbhd if nb in indices_set])
    else: # neighbors is a list of sets
        for idx in range(n_verts):
            nbhd = neighbors[idx].copy()
            new_nbhd = nbhd
            for _ in range(1, radius):
                ext_nbhd = set()
                for nb in new_nbhd:
                    ext_nbhd.update(neighbors[nb])
                new_nbhd = ext_nbhd.difference(nbhd)
                nbhd.update(ext_nbhd)
            # new_val[idx] = np.mean(values[np.isin(indices, list(nbhd))])
            # The above code is slow again...
            new_val[idx] = np.mean([values[nb] for nb in nbhd if nb in indices_set])
    return new_val


def interp_dset(fdset, fmesh, prefix):
    '''
    `fmesh` can be either:
    - fname for a high density surface mesh
    - (neighbors, n_verts) for using shared memory parallelism
    '''
    indices, values = io.read_niml_bin_nodes(fdset)
    if isinstance(fmesh, six.string_types):
        verts, faces = io.read_asc(fmesh)
        new_val = interp_over_mesh(verts, faces, indices, values)
    else: # fmesh = (neighbors, n_verts), for shared memory parallelism
        new_val = interp_over_mesh(None, None, indices, values, neighbors=fmesh[0], n_verts=fmesh[1])
    io.write_niml_bin_nodes(utils.fname_with_ext(prefix, '.niml.dset'), np.arange(new_val.shape[0]), new_val)


def compute_verts_area(verts, faces):
    '''
    Compute element area for each vertex.
    '''
    # Compute area for each face triangle
    A = verts[faces[:,0],:]
    B = verts[faces[:,1],:]
    C = verts[faces[:,2],:]
    face_areas = 0.5*np.linalg.norm(np.cross(B-A, C-A), axis=1)
    vert_areas = np.zeros(verts.shape[0])
    # Attribtue the face area to each of its three vertices
    # for k, f in enumerate(faces):
    #     vert_areas[f] += face_areas[k]
    # vert_areas /= 3.0
    # The above code with advanced indexing is 2x slower...
    for (a, b, c), x in zip(faces, face_areas/3.0):
        vert_areas[a] += x
        vert_areas[b] += x
        vert_areas[c] += x
    return vert_areas


def smooth_verts_data(verts, faces, data, factor=0.1, n_iters=1):
    for _ in range(n_iters):
        smooth_data = np.zeros(len(data))
        counts = np.zeros(len(data), dtype=int)
        for a, b, c in faces:
            smooth_data[a] += data[b] + data[c]
            smooth_data[b] += data[c] + data[a]
            smooth_data[c] += data[a] + data[b]
            counts[a] += 2
            counts[b] += 2
            counts[c] += 2
        smooth_data = factor * smooth_data/counts + (1-factor) * data
        data = smooth_data
    return smooth_data


def compute_intermediate_mesh(inner, outer, alpha, method='equivolume'):
    alpha = np.array(alpha).reshape(-1, 1)
    vin, fin = io.read_asc(inner) if isinstance(inner, six.string_types) else inner
    vout, fout = io.read_asc(outer) if isinstance(outer, six.string_types) else outer
    Ain = compute_verts_area(vin, fin)
    Aout = compute_verts_area(vout, fout)
    if method in ['equivolume', 'equivolume_inside']:
        smooth_factor = 0.1
        smooth_iters = 2
        Ain = smooth_verts_data(vin, fin, Ain, factor=smooth_factor, n_iters=smooth_iters)
        Aout = smooth_verts_data(vout, fout, Aout, factor=smooth_factor, n_iters=smooth_iters)
        if method == 'equivolume':
            rho = 1 / (Aout - Ain) * (-Ain + np.sqrt(alpha * Aout**2 + (1-alpha) * Ain**2))
        elif method == 'equivolume_inside':
            rho = alpha + np.zeros(len(Ain))
            inside = ((0 <= alpha) & (alpha <= 1)).ravel()
            rho[inside,:] = 1 / (Aout - Ain) * (-Ain + np.sqrt(alpha[inside,:] * Aout**2 + (1-alpha[inside,:]) * Ain**2))
    elif method == 'equidistance':
        rho = alpha
    verts = (1-rho[...,np.newaxis]) * vin + rho[...,np.newaxis] * vout
    faces = fin
    return verts.squeeze(), faces


def compute_voxel_depth(xyz, inner, outer, S2E_mat, method='equivolume'):
    if isinstance(xyz, six.string_types):
        xyz = io.Mask(xyz, kind='full').xyz
    if isinstance(S2E_mat, six.string_types):
        S2E_mat = afni.get_S2E_mat(S2E_mat, mat='S2B')
    alphas = np.linspace(0, 1, 11)
    verts, faces = compute_intermediate_mesh(inner, outer, alphas, method=method)
    verts = np.dot(S2E_mat[:,:3], (verts*[-1,-1,1]).transpose(0,2,1)).transpose(1,2,0) + S2E_mat[:,3]
    face_xyz = (verts[:,faces[:,0],:] + verts[:,faces[:,1],:] + verts[:,faces[:,2],:]) / 3
    kdt = spatial.cKDTree(face_xyz.reshape(-1,3), )
    depth = np.zeros(xyz.shape[0])
    from mypy import mypy
    pp = mypy.ProgressPrinter(len(depth), delta=0.0001)
    for k, p in enumerate(xyz):
        # idx = np.argmin(np.linalg.norm(p - face_xyz, axis=-1)) // faces.shape[0]
        idx = kdt.query(p)[1] // faces.shape[0]
        depth[k] = alphas[idx]
        pp.step()
    return depth


def intermediate_asc(fname, inner, outer, alpha, method='equivolume'):
    verts, faces = compute_intermediate_mesh(inner, outer, alpha, method=method)
    io.write_asc(fname, verts, faces)


class Surface(object):
    def __init__(self, suma_dir, surf_vol=None):
        self.surf_dir = suma_dir
        self.subj = afni.get_suma_subj(self.surf_dir)
        if surf_vol is None:
            self.surf_vol = 'SurfVol_Alnd_Exp+orig.HEAD'
            # self.surf_vol = path.join(self.surf_dir, self.subj + '_SurfVol+orig.HEAD')
        else:
            self.surf_vol = surf_vol
        self.surfs = ['pial', 'smoothwm', 'inflated', 'sphere.reg']
        self.hemis = ['lh', 'rh']
        self.specs = [path.join(self.surf_dir, '{0}_{1}.spec'.format(self.subj, hemi)) for hemi in self.hemis]

    def _get_surf2exp_transform(self, exp_anat):
        pass

    def _get_spherical_coordinates(self, hemi):
        verts = io.read_asc(path.join(self.surf_dir, '{hemi}.sphere.reg.asc'.format(hemi=hemi)))[0]
        theta = np.arccos(verts[:,2]/100) # Polar (inclination) angle, [0,pi]
        phi = np.arctan2(verts[:,1], verts[:,0]) # Azimuth angle, (-pi,pi]
        return theta, phi

    def to_1D_dset(self, prefix, node_values, hemi):
        np.savetxt('{0}.{1}.1D.dset'.format(prefix, hemi), np.c_[np.arange(len(node_values)), node_values])

    def to_vol_dset(self, prefix, surf_dset, grid_parent=None):
        if grid_parent is None:
            grid_parent = self.surf_vol
        tmp_out = []
        pc = utils.ParallelCaller()
        for k, dset in enumerate(surf_dset):
            tmp_out.append(f'tmp.{self.hemis[k]}.{prefix}+orig.HEAD')
            cmd = f'3dSurf2Vol \
                -spec {self.specs[k]} \
                -surf_A smoothwm \
                -surf_B pial \
                -sv {self.surf_vol} \
                -grid_parent {grid_parent} \
                -sdata_1D {dset} \
                -f_steps 13 \
                -f_p1_fr -0.0 \
                -f_pn_fr 0.0 \
                -map_func ave \
                -prefix {tmp_out[k]} -overwrite'
            pc.check_call(cmd)
        pc.wait()
        subprocess.check_call(['3dcalc', '-l', tmp_out[0], '-r', tmp_out[1], '-expr', 'max(l,r)', 
            '-prefix', prefix, '-overwrite'])


if __name__ == '__main__':
    pass

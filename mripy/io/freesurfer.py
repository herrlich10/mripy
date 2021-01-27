#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
from collections import OrderedDict
import numpy as np


def read_fs_uint24(fi):
    '''
    Read big endian 3-byte unsigned integer from opened binary file.
    '''
    return np.sum(np.fromfile(fi, dtype='uint8', count=3) * 256**np.array([2,1,0]))


def write_fs_uint24(fo, x):
    '''
    Write big endian 3-byte unsigned integer to opened binary file.
    '''
    fo.write(np.array(x).astype('>u4').tobytes()[1:])


def read_fs_int32(fi):
    '''
    Read big endian 4-byte integer from opened binary file.
    '''
    return np.fromfile(fi, dtype='>i4', count=1)[0]


def write_fs_int32(fo, x):
    '''
    Write big endian 4-byte integer to opened binary file.
    '''
    np.array(x).astype('>i4').tofile(fo)


def read_fs_str(fi):
    L = read_fs_int32(fi)
    return fi.read(L)[:-1].decode()


def read_fs_surf(fname):
    '''
    Read FreeSurfer surface mesh binary file (big endian).

    Returns
    -------
    verts : Nx3 float array, [x, y, z]
    faces : Nx3 int array, [v1, v2, v3]

    References
    ----------
    https://github.com/fieldtrip/fieldtrip/blob/master/external/freesurfer/read_surf.m
    http://www.grahamwideman.com/gw/brain/fs/surfacefileformats.htm
    '''
    TRIANGLE_FILE_MAGIC_NUMBER = 16777214   # 0xfffffe
    QUAD_FILE_MAGIC_NUMBER = 16777215       # 0xffffff
    NEW_QUAD_FILE_MAGIC_NUMBER = 16777213   # 0xfffffd
    with open(fname, 'rb') as fi:
        magic = read_fs_uint24(fi)
        # Mesh format with triangle faces (three vertices) 
        # Most mesh files are in this format
        if magic == TRIANGLE_FILE_MAGIC_NUMBER:
            fi.readline() # There is a comment line followed by one or two \n's after the magic number
            if fi.read(1) != b'\n': # There is only one \n. Go back.
                fi.seek(-1, whence=1)
            vnum = read_fs_int32(fi)
            fnum = read_fs_int32(fi)
            verts = np.fromfile(fi, dtype='>f4', count=vnum*3).reshape(-1,3) # float32
            faces = np.fromfile(fi, dtype='>i4', count=fnum*3).reshape(-1,3) # int32
        # Mesh format with quadrangle faces (four vertices)
        elif magic in [QUAD_FILE_MAGIC_NUMBER, NEW_QUAD_FILE_MAGIC_NUMBER]:
            vnum = read_fs_uint24(fi)
            fnum = read_fs_uint24(fi)
            verts = np.fromfile(fi, dtype='>i2', count=vnum*3).reshape(-1,3) / 100 # int16/100
            faces = np.sum(np.fromfile(fi, dtype='uint8', count=fnum*4*3).reshape(-1,4,3)
                * 256**np.array([2,1,0]), axis=-1) # uint24
        else:
            raise ValueError(f"Unknown magic number: {magic}.")
        # Sanity check
        assert(len(verts)==vnum)
        assert(len(faces)==fnum)
        return verts, faces


def read_fs_patch(fname):
    '''
    Read FreeSurfer surface patch binary file (big endian).

    Returns
    -------
    vtx : int array
        The stored vtx value encoding both nodes and border information.
    verts : Nx3 float array, [x, y, z]
    nodes : int array
        The node index of the patch vertices on the original inflated surface.
        The numbers are obviously non-contiguous.
    border : bool array
        Whether the vertex is on the border of the patch.

    References
    ----------
    https://rdrr.io/cran/freesurferformats/src/R/read_fs_patch.R
    http://www.grahamwideman.com/gw/brain/fs/surfacefileformats.htm (obsoleted)
    '''
    with open(fname, 'rb') as fi:
        magic = read_fs_int32(fi)
        if magic != -1:     # 0xffffffff, the first 3-byte (uint24) equals 16777215
            raise ValueError(f"Unknown magic number: {magic}.")
        vnum = read_fs_int32(fi)
        dtype = np.dtype([('vtx', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
        verts = np.fromfile(fi, dtype=dtype, count=vnum)
        # Sanity check
        assert(len(verts)==vnum)
        vtx = verts['vtx']
        nodes = np.abs(verts['vtx']) - 1
        border = (verts['vtx']<0)
        verts = np.c_[verts['x'], verts['y'], verts['z']]
        return vtx, verts, nodes, border


def read_fs_curv(fname):
    '''
    Read FreeSurfer surface data binary file (big endian).

    Returns
    -------
    curv : float array
        Per vertex data value (e.g., curvature, thickness, sulc, etc.).
    
    References
    ----------
    https://github.com/fieldtrip/fieldtrip/blob/master/external/freesurfer/read_curv.m
    http://www.grahamwideman.com/gw/brain/fs/surfacefileformats.htm
    '''
    NEW_VERSION_MAGIC_NUMBER = 16777215
    with open(fname, 'rb') as fi:
        magic = read_fs_uint24(fi)
        # Most data files are in this format
        if magic == NEW_VERSION_MAGIC_NUMBER:
            vnum = read_fs_int32(fi)
            fnum = read_fs_int32(fi) # Not used
            vals_per_vertex = read_fs_int32(fi) # Only 1 is supported by FreeSurfer as of 20210104
            curv = np.fromfile(fi, dtype='>f4', count=vnum*vals_per_vertex).reshape(-1,vals_per_vertex).squeeze()
        else:
            vnum = magic
            fnum = read_fs_uint24(fi)
            curv = np.fromfile(fi, dtype='>i2', count=vnum) / 100
        # Sanity check
        assert(len(curv)==vnum)
        return curv


def write_fs_curv(fname, curv):
    '''
    Write FreeSurfer surface data binary file (big endian).

    Parameters
    ----------
    curv : 1D array like
        Surface data, each vertex must have one and only one value.

    Only support writing in the new binary version.
    '''
    vnum = curv.shape[0]
    fnum = 0
    vals_per_vertex = 1
    assert(curv.shape==(vnum,))
    magic = 16777215
    with open(fname, 'wb') as fo:
        write_fs_uint24(fo, magic)
        write_fs_int32(fo, vnum)
        write_fs_int32(fo, fnum)
        write_fs_int32(fo, vals_per_vertex)
        curv.astype('>f4').tofile(fo)


def read_fs_annotation(fname, return_df=False):
    '''
    Returns
    -------
    nodes : int array

    References
    ----------
    https://github.com/fieldtrip/fieldtrip/blob/master/external/freesurfer/read_annotation.m
    '''
    with open(fname, 'rb') as fi:
        vnum = read_fs_int32(fi)
        nodes, labels = np.fromfile(fi, dtype='>i4', count=vnum*2).reshape(-1,2).T
        has_table = (np.fromfile(fi, dtype='>i4', count=1).size > 0)
        if has_table:
            enum = read_fs_int32(fi) # Number of entries
            if enum > 0: # Reading from the original version (FSread_annot -FSversion 2005)
                raise NotImplementedError
            else: # Reading from later versions
                version = -enum
                if version == 2: # This is the only other version as of 20210104 (FSread_annot -FSversion 2009)
                    enum = read_fs_int32(fi) # Not really used. Not always equals `enum_to_read`
                    orig_table_fname = read_fs_str(fi)
                    enum_to_read = read_fs_int32(fi) # Number of entries
                    color_table = OrderedDict()
                    for k in range(enum_to_read):
                        struct_id = read_fs_int32(fi) # Not used
                        struct_name = read_fs_str(fi)
                        R, G, B, T = np.fromfile(fi, dtype='>i4', count=4) # T for transparency (i.e., 255 - alpha)
                        label = R + G*2**8 + B*2**16 # This is what stored in labels
                        color_table[struct_name] = (R, G, B, T, label)
                    print(f'The annotation file contains a color table with {enum_to_read} entries (originally "{orig_table_fname}")')
                else:
                    raise ValueError(f"Unknown annotation version: {version}.")
        else: # The file does not contain name and color information
            color_table = None
        if return_df:
            import pandas as pd
            color_table = pd.DataFrame(color_table, index=['R', 'G', 'B', 'transparency', 'label']).T
        return nodes, labels, color_table


def write_fs_color_table(fname, color_table):
    '''
    Write FreeSurfer color table file (*.ctab), similar to {subject}/label/aparc.annot.a2009s.ctab, 
    or https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

    This file will be useful to generate a valid annot.niml.dset containing correct color table, 
    which can then be used in *.spec to display custom anatomical labels at SUMA crosshair location. E.g., 
    FSread_annot -input lh.HCP-MMP1.annot \
        -FSversion 2009 -FScmap lh.HCP-MMP1.ctab -FScmaprange 0 180 \
        -dset lh.HCP-MMP1.annot.niml.dset -overwrite
    '''
    try: # If `color_table` is a pd.DataFrame, convert it to OrderedDict
        color_table = OrderedDict(zip(color_table.index.values, color_table.values))
    except AttributeError:
        pass
    idx_w = int(np.ceil(np.log10(len(color_table))))
    name_w = int(np.ceil((max([len(name) for name in color_table.keys()])+2)/10)*10)
    with open(fname, 'w') as fo:
        for idx, (name, (R, G, B, T, label)) in enumerate(color_table.items()):
            fo.write(f"{idx:{idx_w}d}  {name:<{name_w}}{R:4d}{G:4d}{B:4d} {T:4d}\n")


if __name__ == '__main__':
    pass

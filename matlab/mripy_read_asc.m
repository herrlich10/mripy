function [verts, faces] = mripy_read_asc(fname)
%MRIPY_READ_ASC Read FS/SUMA surface (vertices and faces) in *.asc format.
%   2017-08-12: Created by qcc
    n = dlmread(fname, '', [1, 0, 1, 1]); % Undocumented feature: \s as delimiter
    n_verts = n(1);
    n_faces = n(2);
    verts = dlmread(fname, '', [2, 0, 1+n_verts, 3]);
    faces = dlmread(fname, '', [2+n_verts, 0, 1+n_verts+n_faces, 3]);
end


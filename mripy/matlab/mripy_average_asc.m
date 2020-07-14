function [verts, faces] = mripy_average_asc(fnames, output)
%MRIPY_AVERAGE_ASC Average the volumetric geometry of surface meshes.
%   2017-08-17: Created by qcc
    N = length(fnames);
    [verts, faces] = mripy_read_asc(fnames{1});
    for k = 2:N
        [v, f] = mripy_read_asc(fnames{k});
        assert(all(size(v)==size(verts)));
        assert(all(f(:)==faces(:)));
        verts = verts + v;
    end
    verts = verts / N;
    if nargin >= 2
        mripy_write_asc(output, verts, faces);
    end
end


function mripy_write_asc(fname, verts, faces)
%MRIPY_WRITE_ASC Write FS/SUMA surface (vertices and faces) as *.asc format.
%   2017-08-13: Created by qcc
    n_verts = size(verts, 1);
    n_faces = size(faces, 1);
    assert(n_verts == max(reshape(faces(:,1:3),[],1))+1); % Sanity check
    dlmwrite(fname, ['# Created ', datestr(now(), 'yyyy-mm-dd HH:MM:SS.FFF'), ' with mripy'], '');
    dlmwrite(fname, [n_verts, n_faces], 'delimiter', ' ', 'precision', '%d', '-append');
    dlmwrite(fname, verts, 'delimiter', ' ', 'precision', '%.6f', '-append');
    dlmwrite(fname, faces, 'delimiter', ' ', '-append');
end


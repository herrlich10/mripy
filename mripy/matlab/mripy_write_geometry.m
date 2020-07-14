function data = mripy_write_geometry(fname, verts, faces)
%MRIPY_WRITE_GEOMETRY Write FS/SUMA surface (vertices and faces) as
%   three.js BufferGeometry.
%
%   References
%   ----------
%   https://threejs.org/docs/#api/core/BufferGeometry
% 
%   2017-08-17: Created by qcc
    data.position = reshape(verts(:,1:3)', [], 1);
    data.index = reshape(faces(:,1:3)', [], 1);
    if ~isempty(fname)
        text = jsonencode(data); % Require R2016b
        fid = fopen(fname, 'w');
        fprintf(fid, '%s\n', text);
        fclose(fid);
    end
end


function mripy_align_asc(fname, transform, output)
%MRIPY_ALIGN_ASC Apply surface-to-experiment alignment transform to
%   vertices coordinates.
%   2017-08-14: Created by qcc
%   2017-08-16: Get coordinates mapping done right
    if nargin < 3
        [pth, name, ext] = fileparts(fname);
        output = fullfile(pth, [name, '_al', ext]);
    end
    fprintf('>> Aligning "%s" according to "%s"...\n', fname, transform);
    [verts, faces] = mripy_read_asc(fname);
    xform = mripy_read_transform(transform);
    % SPM (*.asc vertices coordinates) to DICOM (afni internal coordinates) before matrix multiplication
    xyz1 = (xform * [bsxfun(@times, verts(:,1:3), [-1, -1, 1]), ones(size(verts,1),1)]')';
    verts(:,1:3) = bsxfun(@times, xyz1(:,1:3), [-1, -1, 1]);
    mripy_write_asc(output, verts, faces);
end


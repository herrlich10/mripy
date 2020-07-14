function xform = mripy_read_transform(fname)
%MRIPY_READ_TRANSFORM Read AFNI coordinates transform matrix in *.1D format.
%   2017-08-14: Created by qcc
    fid = fopen(fname);
    c = textscan(fid, '%f', 'Delimiter', ' ', 'MultipleDelimsAsOne', true, 'CommentStyle', '#');
    fclose(fid);
    xform = reshape([c{1}], 4, 3)';
end


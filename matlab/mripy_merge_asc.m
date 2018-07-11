function mripy_merge_asc(lh, rh, output)
%MRIPY_MERGE_ASC Merge FS/SUMA surfaces (left and right hemisphere) in *.asc format.
%   Note that vertices indexing is zero-based and continuous.
%
%   References
%   1. https://github.com/PyMVPA/PyMVPA/blob/master/mvpa2/tests/test_surfing_afni.py
%   2. http://www.pymvpa.org/generated/cmd_prep-afni-surf.html
%   3. https://afni.nimh.nih.gov/afni/community/board/read.php?1,149391,149394
%   4. https://afni.nimh.nih.gov/afni/community/board/read.php?1,146337,146339
%   5. http://cosmomvpa.org/datadb-v0.3.zip/digit/README
% 
%   2017-08-13: Created by qcc
    if nargin < 3
        [pth, name, ext] = fileparts(lh);
        out_name = regexprep(name, '\.(lh)\.', '.mh.');
        if strcmp(out_name, name) % No match...
            error('Failed to auto generate output fname. You should specify it explicitly~');
        end
        output = fullfile(pth, [out_name, ext]);
    end
    [lv, lf] = mripy_read_asc(lh);
    [rv, rf] = mripy_read_asc(rh);
    rf(:,1:3) = rf(:,1:3) + size(lv,1);
    verts = [lv; rv];
    faces = [lf; rf];
    mripy_write_asc(output, verts, faces);
end


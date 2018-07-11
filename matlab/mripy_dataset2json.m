function data = mripy_dataset2json(asc, anat, func)
%MRIPY_DATASET2JSON Save surface dataset as three.js data in *.json.
%   2017-08-18: Created by qcc
    [verts, faces] = mripy_read_asc(asc);
    data = mripy_write_geometry([], verts, faces);
    ds = cosmo_surface_dataset(anat);
    data.underlay = ds.samples';
    ds2 = cosmo_surface_dataset(func);
    data.overlay = ds2.samples(1,:)';
    data.overlay(isnan(data.overlay)) = 0;
end


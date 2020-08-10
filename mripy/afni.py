#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import sys, os, re, shlex, shutil, glob, subprocess, collections
from os import path
from datetime import datetime
import numpy as np
from scipy import interpolate
import matplotlib as mpl
from . import six


# Test afni installation
has_afni = bool(re.search('version', subprocess.check_output(['afni', '-ver']).decode('utf-8'), re.IGNORECASE))
# # Find afni path
# config_dir = path.expanduser('~/.mripy')
# if not path.exists(config_dir):
#     os.makedirs(config_dir)
# if has_afni:
#     config_file = path.join(config_dir, 'afni_path')
#     if path.exists(config_file):
#         with open(config_file, 'r') as f:
#             afni_path = f.readline()
#     else:
#         afni_path = subprocess.check_output('find ~ -iregex ".*/abin"', shell=True).decode('utf-8').split('\n')[0]
#         with open(config_file, 'w') as f:
#             f.write(afni_path)
# else:
#     afni_path = ''


def filter_output(lines, tags=None, pattern=None):
    '''
    Filter output lines according to their initial tags (++, *+, **, etc.) and/or
    a regex search pattern.

    Parameters
    ----------
    tags : list of tags
        Default is [], which means all lines will pass the filter.
    pattern : str
    '''
    if tags is None:
        tags = []
    if len(tags) > 0:
        lines = [line for line in lines if line[:2] in tags]
    if pattern is not None:
        lines = [line for line in lines if re.search(pattern, line)]
    return lines


def check_output(cmd, tags=None, pattern=None, verbose=0, **kwargs):
    '''
    The syntax of subprocess.check_output(shell=False) is tedious for long cmd.
    But for security reason, we don't want to use shell=True for external cmd.
    This helper function allows you to execute a single cmd without shell=True.

    Parameters
    ----------
    cmd : str
        A single command string packed with all options (but no wildcard)
    **kwargs :
        Go to subprocess.check_output(**kwargs)

    Returns
    -------
    lines : list of lines
        Much easier to deal with compared with subprocess.check_output()
    '''
    if isinstance(cmd, six.string_types):
        cmd = shlex.split(cmd) # Split by space, preserving quoted substrings
    lines = subprocess.check_output(cmd, stderr=subprocess.STDOUT, **kwargs).decode('utf-8').split('\n')
    lines = filter_output(lines, tags, pattern)
    if verbose:
        for line in lines:
            print(line, file=sys.stderr if line.startswith('*') else sys.stdout)
    return lines


def call(cmd):
    if isinstance(cmd, six.string_types):
        cmd = shlex.split(cmd) # Split by space, preserving quoted substrings
    cmd_str = ' '.join(cmd)
    print('>>', cmd_str)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    for line in iter(p.stdout.readline, b''): # The 2nd argument is sentinel character
        print(line.decode('utf-8'), end='')
    p.stdout.close() # Notify the child process that the PIPE has been broken
    if p.wait():
        raise RuntimeError(f'Error occurs when executing the following command (returncode={p.returncode}):\n{cmd_str}') 


def split_out_file(out_file, split_path=False, trailing_slash=False):
    '''
    Ensure that path.join(out_dir, prefix, ext) can be checked by path.exists().

    >>> split_out_file('dset.nii')
    ('dset', '.nii')
    >>> split_out_file('dset.1D')
    ('dset', '.1D')
    >>> split_out_file('folder/dset')
    ('folder/dset', '+orig.HEAD')
    >>> split_out_file('folder/dset+orig', split_path=True)
    ('folder', 'dset', '+orig.HEAD')
    >>> split_out_file('dset+orig.', split_path=True)
    ('', 'dset', '+orig.HEAD')
    >>> split_out_file('folder/dset+orig.HEAD', split_path=True, trailing_slash=True)
    ('folder/', 'dset', '+orig.HEAD')
    >>> split_out_file('dset+tlrc.BRIK', split_path=True, trailing_slash=True)
    ('', 'dset', '+tlrc.HEAD')
    '''
    out_dir, out_name = path.split(out_file)
    if trailing_slash and out_dir:
        out_dir += '/'
    match = re.match(r'(.+)(.nii|.nii.gz|.1D|.1D.dset|.1D.roi|.niml.dset|.niml.roi|.gii.dset|.csv)$', out_name)
    if match:
        prefix, ext = match.groups()
    else:
        match = re.match(r'(.+)(\+(?:orig|tlrc))(?:.|.HEAD|.BRIK)?$', out_name)
        if match:
            prefix, ext = match.groups()
            ext += '.HEAD'
        else:
            prefix = out_name
            ext = '+orig.HEAD'
    if split_path:
        return out_dir, prefix, ext
    else:
        return path.join(out_dir, prefix), ext


def insert_suffix(fname, suffix):
    prefix, ext = split_out_file(fname)
    return f"{prefix}{suffix}{ext}"


def get_prefix(fname, with_path=False):
    '''
    Return "dset" given "path/to/dset+orig.HEAD", "dset+orig.", "dset+tlrc", "dsets"
    '''
    if path.splitext(fname)[1] in ['.niml', '.1D', '.dset']: # For surface dset
        match = re.match(r'(.+)\.(?:niml|1D)(?:\.dset)?', fname)
        prefix = match.group(1)
    else: # For 3d dset
        # fstem = path.splitext(path.basename(fname))[0]
        if fname[-5:].upper() in ['.HEAD', '.BRIK']:
            fstem = fname[:-5]
        elif fname.endswith('.'):
            fstem = fname[:-1]
        else:
            fstem = fname
        prefix = fstem[:-5] if len(fstem) > 5 and fstem[-5:] in ['+orig', '+tlrc'] else fstem
    if not with_path:
        prefix = path.basename(prefix)
    return prefix


def get_surf_vol(suma_dir):
    '''
    Infer SUMA SurfVol filename with full path (agnostic about file type: .nii vs +orig.HEAD/BRIK).
    '''
    # TODO: SurfVol.depth.nii
    candidates = glob.glob(path.join(suma_dir, '*_SurfVol*'))
    candidates = [f for f in candidates if re.search(r'_SurfVol(?:\.nii|\+orig\.HEAD)', f)]
    if len(candidates) == 0:
        raise ValueError(f'>> Cannot identify SurfVol in "{suma_dir}"')
    else:
        return candidates[0]


def get_suma_subj(suma_dir):
    '''Infer SUMA subject given path to SUMA folder.'''
    match = re.match('(.+)_SurfVol.+', path.basename(get_surf_vol(suma_dir)))
    if match:
        return match.group(1)
    else:
        raise RuntimeError(f'>> Cannot infer SUMA subject from "{suma_dir}"')


def get_surf_type(suma_dir):
    '''Infer SUMA surface mesh file type (.gii vs .asc).'''
    surf_files = [f for f in os.listdir(suma_dir) if re.match('(?:lh|rh).(?:pial|smoothwm|inflated).*', f)]
    return path.splitext(surf_files[0])[1]


SPEC_HEMIS = ['lh', 'rh', 'both', 'mh', 'bh']
HEMI_PATTERN = r'(?:(?<=[^a-zA-Z0-9])|^)(?:lh|rh|both|mh|bh)(?=[^a-zA-Z0-9])'

def substitute_hemi(fname, hemi='{0}'):
    return re.sub(HEMI_PATTERN, hemi, fname)


def get_suma_spec(suma_spec):
    '''
    Infer other spec files from one spec file (either lh.spec, rh.spec, or both.spec).
    
    Parameters
    ----------
    suma_spec : str
        Either a .spec file or the suma_dir.
    '''
    if path.isdir(suma_spec): # It is actually the `suma_dir`
        subj = get_suma_subj(suma_spec)
        return {hemi: path.join(suma_spec, f"{subj}_{hemi}.spec") for hemi in SPEC_HEMIS}
    else: # It is a .spec file
        spec_fmt = re.sub(f"({'|'.join(SPEC_HEMIS)}).spec", '{0}.spec', suma_spec)
        return {hemi: spec_fmt.format(hemi) for hemi in SPEC_HEMIS}


def get_suma_info(suma_dir, suma_spec=None):
    info = {}
    info['subject'] = get_suma_subj(suma_dir)
    if suma_spec is None: # Infer spec files from suma_dir
        info['spec'] = get_suma_spec(suma_dir)
    else: # Infer other spec files from one spec file
        info['spec'] = get_suma_spec(suma_spec)
    return info


def get_hemi(fname):
    basename = path.basename(fname)
    match = re.search(HEMI_PATTERN, basename)
    if match:
        hemi = match.group(0)
    else:
        raise ValueError(f'** ERROR: Cannot infer "hemi" from "{basename}"')
    return hemi


def infer_surf_dset_variants(fname, hemis=SPEC_HEMIS):
    '''
    >>> infer_surf_dset_variants('data.niml.dset')
    {'lh': 'lh.data.niml.dset', 'rh': 'rh.data.niml.dset', 'both': 'both.data.niml.dset', mh': 'mh.data.niml.dset'}
    >>> infer_surf_dset_variants('lh.data.niml.dset')
    {'lh': 'lh.data.niml.dset'}

    Parameters
    ----------
    fname : str, list, or dict
    '''
    if isinstance(fname, six.string_types):
        match = re.search(HEMI_PATTERN, path.basename(fname))
        if match:
            fname = {match.group(0): fname}
        else:
            out_dir, prefix, ext = split_out_file(fname, split_path=True, trailing_slash=True)
            fname = {hemi: f"{out_dir}{hemi}.{prefix}{ext}" for hemi in hemis}
    if not isinstance(fname, dict):
        fdict = {}
        for f in fname:
            match = re.search(HEMI_PATTERN, path.basename(f))
            if match:
                fdict[match.group(0)] = f
            else:
                raise ValueError(f'** ERROR: Cannot infer "hemi" from "{path.basename(f)}"')
        fname = fdict
    return fname


def get_ORIENT(fname, format='str'):
    '''
    Parameters
    ----------
    format : str, {'code', 'str', 'mat', 'sorter'}

    References
    ----------
    [1] https://afni.nimh.nih.gov/pub/dist/doc/program_help/README.attributes.html
        #define ORI_R2L_TYPE  0  /* Right to Left         */
        #define ORI_L2R_TYPE  1  /* Left to Right         */
        #define ORI_P2A_TYPE  2  /* Posterior to Anterior */
        #define ORI_A2P_TYPE  3  /* Anterior to Posterior */
        #define ORI_I2S_TYPE  4  /* Inferior to Superior  */
        #define ORI_S2I_TYPE  5  /* Superior to Inferior  */

        Thus "0 3 4" is standard DICOM Reference Coordinates System, i.e., RAI.
        The AFNI convention is also that R-L, A-P, and I-S are negative-to-positive, i.e., RAI.

    [2] https://nipy.org/nibabel/nifti_images.html
        On the other hand, NIFTI images have an affine relating the voxel coordinates 
        to world coordinates in RAS+ space, or LPI in AFNI's term.
    '''
    res = check_output(['3dAttribute', 'ORIENT_SPECIFIC', fname])[-2]
    ORIENT = np.fromiter(map(int, res.split()), int)
    code2str = np.array(['R', 'L', 'P', 'A', 'I', 'S'])
    code2mat = np.array([[ 1, 0, 0],
                         [-1, 0, 0],
                         [ 0,-1, 0],
                         [ 0, 1, 0],
                         [ 0, 0, 1],
                         [ 0, 0,-1]])
    code2axis = np.array([0, 0, 1, 1, 2, 2])
    if format == 'code':
        return ORIENT
    elif format == 'str':
        return ''.join(code2str[ORIENT])
    elif format == 'mat':
        return code2mat[ORIENT]
    elif format == 'sorter':
        return np.argsort(code2axis[ORIENT])


def get_DIMENSION(fname):
    '''
    [x, y, z, t, 0]
    '''
    res = check_output(['3dAttribute', 'DATASET_DIMENSIONS', fname])[-2]
    DIMENSION = np.fromiter(map(int, res.split()), int)
    return DIMENSION


def get_ORIGIN(fname):
    res = check_output(['3dAttribute', 'ORIGIN', fname])[-2]
    ORIGIN = np.fromiter(map(float, res.split()), float)
    return ORIGIN


def get_DELTA(fname):
    res = check_output(['3dAttribute', 'DELTA', fname])[-2]
    DELTA = np.fromiter(map(float, res.split()), float)
    return DELTA


def get_affine(fname):
    ORIENT = get_ORIENT(fname, format='sorter')
    ORIGIN = get_ORIGIN(fname)
    DELTA = get_DELTA(fname)
    MAT = np.c_[np.diag(DELTA), ORIGIN][ORIENT,:]
    return MAT


def get_affine_nifti(fname):
    MAT = np.diag([-1,-1, 1]) @ get_affine(fname)
    return MAT


def get_dims(fname):
    '''
    Dimensions (number of voxels) of the data matrix.
    See also: get_head_dims
    '''
    # res = check_output(['@GetAfniDims', fname])[-2] # There can be leading warnings for oblique datasets
    res = check_output(['3dinfo', '-n4', fname])[-2] # `@GetAfniDims` may not work for things like `dset.nii'[0..10]'`
    return np.int_(res.split()) # np.fromiter(map(int, res.split()), int)


def get_head_dims(fname):
    '''
    Dimensions (number of voxels) along R-L, A-P, I-S axes.
    See also: get_dims
    '''
    res = check_output(['3dinfo', '-orient', '-n4', fname])[-2]
    res = res.split()
    orient = res[0]
    dims = np.int_(res[1:])
    ori2ax = {'R': 0, 'L': 0, 'A': 1, 'P': 1, 'I': 2, 'S': 2}
    axes = [ori2ax[ori] for ori in orient]
    return np.r_[dims[np.argsort(axes)], dims[3]]


def get_head_delta(fname):
    '''
    Resolution (voxel size) along R-L, A-P, I-S axes.
    '''
    res = check_output(['3dinfo', '-orient', '-d3', fname])[-2]
    res = res.split()
    orient = res[0]
    delta = np.abs(np.float_(res[1:]))
    ori2ax = {'R': 0, 'L': 0, 'A': 1, 'P': 1, 'I': 2, 'S': 2}
    axes = [ori2ax[ori] for ori in orient]
    return delta[np.argsort(axes)]


def get_head_extents(fname):
    '''
    Spatial extent along R, L, A, P, I and S.
    '''
    res = check_output(['3dinfo', '-extent', fname])[-2]
    return np.float_(res.split())


def get_brick_labels(fname, label2index=False):
    res = check_output(['3dAttribute', 'BRICK_LABS', fname])[-2]
    labels = res.split('~')[:-1] # Each label ends with "~"
    if label2index:
        return {label: k for k, label in enumerate(labels)}
    else:
        return np.array(labels)


def set_brick_labels(fname, labels):
    check_output(['3drefit', '-relabel_all_str', ' '.join(labels), fname])


def get_TR(fname):
    return float(check_output(['3dinfo', '-TR', fname])[-2])


def get_attribute(fname, name, type=None):
    res = check_output(['3dAttribute', name, fname])[-2]
    if type == 'int':
        return np.int_(res[:-1].split())
    elif type == 'float':
        return np.float_(res[:-1].split())
    else:
        return res[:-1]


def set_attribute(fname, name, value, type=None):
    values = np.atleast_1d(value)
    if type == 'str' or isinstance(value, str):
        check_output(['3drefit', '-atrstring', name, f"{value}", fname])
    elif type == 'int' or np.issubdtype(values.dtype, np.integer):
        check_output(['3drefit', '-atrint', name, f"{' '.join([str(v) for v in values])}", fname])
    elif type == 'float' or np.issubdtype(values.dtype, np.floating):
        check_output(['3drefit', '-atrfloat', name, f"{' '.join([str(v) for v in values])}", fname])


def get_nifti_field(fname, name, type=None):
    res = check_output(['nifti_tool', '-disp_hdr', '-field', name, '-infiles', fname])[-2]
    if type == 'int':
        return np.int_(res.split()[3:])
    elif type == 'float':
        return np.float_(res.split()[3:])
    else:
        return res[37:]


def set_nifti_field(fname, name, value, out_file=None):
    values = np.atleast_1d(value)
    check_output(['nifti_tool', '-mod_hdr', '-mod_field', name, f"{' '.join([str(v) for v in values])}", '-infiles', fname] 
        + (['-overwrite'] if out_file is None else ['-prefix', out_file]))


def get_S2E_mat(fname, mat='S2E'):
    mat = {'S2E': 'S2B', 'S2B': 'S2B', 'E2S': 'B2S', 'B2S': 'B2S'}[mat]
    res = check_output("cat_matvec -ONELINE '{0}::ALLINEATE_MATVEC_{1}_000000'".format(fname, mat))[-2]
    return np.float_(res.split()).reshape(3,4)


def generate_spec(fname, surfs, ext=None, **kwargs):
    if ext is None:
        ext = '.gii'
    defaults = dict(dict(type={'.asc': 'FS', '.gii': 'GII'}[ext], state=None, anat=None, parent=None), **kwargs)
    surfs = [dict(defaults, **({'name': surf} if isinstance(surf, six.string_types) else surf)) for surf in surfs]
    has_smoothwm = np.any([('smoothwm' in surf['name']) for surf in surfs])
    is_both = np.any([('lh' in surf['name']) for surf in surfs]) and np.any([('rh' in surf['name']) for surf in surfs])
    for surf in surfs:
        match = re.search(rf'([l|r]h)\.(.+)\.{ext[1:]}', surf['name'])
        surf['hemi'] = match.group(1)
        surf['surf'] = match.group(2)
        is_anat = surf['surf'] in ['pial', 'smoothwm', 'white']
        if surf['state'] is None:
            if not is_anat and is_both:
                surf['state'] = '_'.join([surf['surf'], surf['hemi']])
            else:
                surf['state'] = surf['surf']
        if surf['anat'] is None:
            surf['anat'] = 'Y' if is_anat else 'N'
        if surf['parent'] is None:
            if surf['name'] == 'smoothwm' or not has_smoothwm:
                surf['parent'] = 'SAME'
            else:
                surf['parent'] = '.'.join([surf['hemi'], 'smoothwm', ext[1:]])
    cmds = []
    for surf in surfs:
         cmds.extend(['-tsnad', surf['type'], surf['state'], surf['name'], surf['anat'], surf['parent']])
    subprocess.check_call(['quickspec', '-spec', fname, '-overwrite'] + cmds)


def update_afnirc(**kwargs):
    rc_file = path.expanduser('~/.afnirc')
    bak_file = path.expanduser('~/.afnirc.{0}.bak'.format(datetime.now().strftime('%Y%m%d')))
    if not path.exists(bak_file):
        shutil.copy(rc_file, bak_file)
    with open(rc_file, 'r') as fin:
        lines = fin.read().splitlines()
    updated = []
    is_managed = False
    managed_begin = '// Managed by mripy: begin'
    managed_end = '// Managed by mripy: end'
    managed = collections.OrderedDict()
    for line in lines:
        if not is_managed:
            if line == managed_begin:
                is_managed = True
            else:
                updated.append(line)
        else:
            if line == managed_end:
                is_managed = False
            else:
                match = re.search('(\S+)\s+=\s+((?:.(?!//))+)(?:\s+//\s+(.+))?', line)
                managed[match.group(1)] = (match.group(2).strip(), match.group(3)) # key, value, comment (can be None)
    for k, v in kwargs.items():
        if not isinstance(v, tuple):
            kwargs[k] = (v, None)
    managed.update(kwargs)
    n_managed = len([v for v in managed.values() if v[0] is not None])
    if n_managed > 0:
        if updated[-1] != '':
            updated.append('')
        updated.append(managed_begin)
    for key, (value, comment) in managed.items():
        if value is not None:
            updated.append('   {0: <24} = {1}'.format(key, value) +
                ('\t// {0}'.format(comment) if comment is not None else ''))
    if n_managed > 0:
        updated.append(managed_end)
    with open(rc_file, 'w') as fout:
        fout.write('\n'.join(updated))


def add_colormap(cmap, name=None, cyclic=False, index=None, categorical=False):
    '''
    cmap : list of RGB colors | matplotlib.colors.LinearSegmentedColormap
    '''
    if name is None:
        if isinstance(cmap, mpl.colors.LinearSegmentedColormap):
            name = cmap.name
        else:
            name = 'User{0:02d}'.format(index)
    if isinstance(cmap, mpl.colors.LinearSegmentedColormap):
        cmap = plots.get_color_list(cmap)
    if index is None:
        index = 1
    # Make colormap dir
    cmap_dir = path.expanduser('~/abin/colormaps')
    if not path.exists(cmap_dir):
        os.makedirs(cmap_dir)
    # Generate palette file
    temp_file = 'colors.tmp'
    with open(temp_file, 'w') as fout:
        fout.writelines(['\t'.join(map(str, color))+'\n' for color in cmap])
    cmap_file = path.join(cmap_dir, '{0}.pal'.format(name))
    with open(cmap_file, 'w') as fout:
        if categorical:
            subprocess.check_call(['MakeColorMap', '-f', temp_file, '-ah', name, '-nc', str(len(cmap))], stdout=fout)
        else:
            subprocess.check_call(['MakeColorMap', '-f', temp_file, '-ah', name] +
                (['-nc', str(128), '-sl'] if cyclic else ['-nc', str(129)]), stdout=fout)
    os.remove(temp_file)
    # Update .afnirc
    update_afnirc(**{'AFNI_COLORSCALE_{0:02d}'.format(index): path.relpath(cmap_file, path.expanduser('~'))})


def write_colorscale_file(fname, pal_name, colors, locations=None, interp=None):
    '''
    Parameters
    ----------
    fname : *.pal file name
    pal_name : palette name (or title)
    colors : a list of RGB colors within [0,1]
        first color (bottom) -> last color (top)
    locations : locations of the breakpoints where colors are defined
        0 (bottom) -> 1 (top)
    interp : 'linear'|'nearest'

    AFNI document says "There are exactly 128 color locations on an AFNI colorscale."
    For details, see https://afni.nimh.nih.gov/pub/dist/doc/OLD/afni_colorscale.html
    But in fact, if you fill the colorscale file with a lot of colors, only the first 256 colors will be used.
    '''
    if locations is None:
        locations = np.linspace(0, 1, len(colors))
    if interp is None:
        interp = 'linear'
    cmap = interpolate.interp1d(locations, colors, kind=interp, axis=0, bounds_error=False, fill_value='extrapolate')
    clist = [mpl.colors.to_hex(color) for color in cmap(np.linspace(0, 1, 256))]
    with open(fname, 'w') as fout:
        fout.write(f"{pal_name}\n")
        fout.writelines([f"{color}\n" for color in reversed(clist)])


def parse_patch(patch):
    '''
    Notes
    -----
    1. Each replacement is started with one or more comment lines. The last
       comment line is treated as replacement target, which may contain an
       optional replacement directive at the end:
        # This is an example <replace command="1"/>
       Possible directives for replacing the original scripts includes:
        1) command="n": replace n commands
        2) line="n": replace n lines
        3) until="regexp": replace until a specific line (the regexp is the
           last line to be replaced)
    2. Each replacement must end with two consecutive newlines.
    '''
    with open(patch, 'r') as fin:
        lines = fin.read().splitlines()
    replacements = []
    is_content = False
    n_newlines = 0
    for k, line in enumerate(lines):
        if is_content:
            contents.append(line)
            if line.strip() == '':
                n_newlines += 1
                if n_newlines >= 2:
                    is_content = False
            else:
                n_newlines = 0
            if not is_content or k+1 == len(lines):
                for kk in range(min(2, len(contents))):
                    if contents[-1] == '':
                        contents.pop(-1)
                    else:
                        break
                contents.append('# </patch>')
                replacements.append({'target': target, 'directives': directives, 'contents': contents})
        elif line[0] == '#':
            if k+1 < len(lines) and lines[k+1][0] != '#':
                match = re.match('((?:(?!<replace).)*)(?:<replace(.*)/>)?', line)
                target = match.group(1).rstrip()
                if match.group(2) is not None:
                    attributes = shlex.split(match.group(2).strip())
                    directives = dict([attr.split('=') for attr in attributes])
                else:
                    directives = {'command': 1}
                is_content = True
                contents = ['# <patch>']
    return replacements


def patch_afni_proc(original, patch, inplace=True):
    replacements = parse_patch(patch)
    n = 0
    with open(original, 'r') as fin:
        lines = fin.read().splitlines()
    patched = []
    n_to_replace = 0
    for k, line in enumerate(lines):
        if n == len(replacements):
            patched.append(line)
        else:
            replacement = replacements[n]
        if not n_to_replace:
            patched.append(line)
            match = re.search(replacement['target'], line)
            if match:
                replacement['indent'] = match.start()
                replacement['n_lines'] = six.MAXSIZE
                directives = replacement['directives']
                if 'command' in directives:
                    nc = 0
                    n_lines = 0
                    while nc < int(directives['command']):
                        n_lines += 1
                        x = lines[k+n_lines].strip()
                        if x != '' and x[0] != '#' and x[-1] != '\\':
                            nc += 1
                    replacement['n_lines'] = min(replacement['n_lines'], n_lines)
                if 'until' in directives:
                    n_lines = 0
                    while not re.match(directives['until'], lines[k+n_lines]):
                        n_lines += 1
                    replacement['n_lines'] = min(replacement['n_lines'], n_lines)
                if 'line' in directives:
                    replacement['n_lines'] = min(replacement['n_lines'], int(directives['line']))
                n_to_replace = replacement['n_lines']
        else:
            patched.append('# ' + line)
            n_to_replace -= 1
            if n_to_replace == 0:
                for content in replacement['contents']:
                    patched.append(' '*replacement['indent'] + content)
    if not inplace:
        p, f = path.split(original)
        fname = path.join(p, 'patched.'+f)
    else:
        shutil.copy(original, original+'.bak')
        fname = original
    with open(fname, 'w') as fout:
        fout.write('\n'.join(patched))


if __name__ == '__main__':
    pass
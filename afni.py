#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import sys, os, re, shlex, shutil, glob, subprocess, collections
from os import path
from datetime import datetime
import numpy as np
import matplotlib as mpl
from . import six


# Test afni installation
has_afni = bool(re.search('version', subprocess.check_output(['afni', '-ver']).decode('utf-8'), re.IGNORECASE))
# Find afni path
config_dir = path.expanduser('~/.mripy')
if not path.exists(config_dir):
    os.makedirs(config_dir)
if has_afni:
    config_file = path.join(config_dir, 'afni_path')
    if path.exists(config_file):
        with open(config_file, 'r') as f:
            afni_path = f.readline()
    else:
        afni_path = subprocess.check_output('find ~ -iregex ".*/abin"', shell=True).decode('utf-8').split('\n')[0]
        with open(config_file, 'w') as f:
            f.write(afni_path)
else:
    afni_path = ''


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


def get_suma_subj(suma_dir):
    '''Infer SUMA subject given path to SUMA folder'''
    try:
        # spec_file = glob.glob(path.join(suma_dir, 'std.60.*_both.spec'))[0]
        # return path.basename(spec_file)[7:-10]
        surf_vol = glob.glob(path.join(suma_dir, '*_SurfVol+orig.HEAD'))[0]
        return path.basename(surf_vol)[:-18]
    except IndexError:
        return None


HEMI_PATTERN = r'(?<=[^a-zA-Z0-9])(?:lh|rh|both)(?=[^a-zA-Z0-9])|^(?:lh|rh|both)(?=[^a-zA-Z0-9])'
SPEC_HEMIS = ['lh', 'rh', 'both']

def substitute_hemi(s, hemi='{0}'):
    return re.sub(HEMI_PATTERN, hemi, s)


def get_suma_spec(suma_spec):
    '''Infer other spec files from one spec file (either lh.spec, rh.spec, or both.spec).'''
    spec_fmt = re.sub('[a-z]+.spec', '{0}.spec', suma_spec)
    return {hemi: spec_fmt.format(hemi) for hemi in SPEC_HEMIS}


def get_suma_info(suma_dir, suma_spec=None):
    info = {}
    info['subject'] = get_suma_subj(suma_dir)
    if suma_spec is None: # Infer spec files from suma_dir
        info['spec'] = {hemi: '{0}/{1}_{2}.spec'.format(suma_dir, info['subject'], hemi) for hemi in SPEC_HEMIS}
    else: # Infer other spec files from one spec file
        info['spec'] = get_suma_spec(suma_spec)
    return info


def get_dims(fname):
    '''
    Dimensions (number of voxels) of the data matrix.
    See also: get_head_dims
    '''
    res = check_output(['@GetAfniDims', fname])[-2] # There can be leading warnings for oblique datasets
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
    labels = res.split('~')
    if label2index:
        return {label: k for k, label in enumerate(labels)}
    else:
        return labels


def get_S2E_mat(fname, mat='S2E'):
    mat = {'S2E': 'S2B', 'S2B': 'S2B', 'E2S': 'B2S', 'B2S': 'B2S'}[mat]
    res = check_output("cat_matvec -ONELINE '{0}::ALLINEATE_MATVEC_{1}_000000'".format(fname, mat))[-2]
    return np.float_(res.split()).reshape(3,4)


def generate_spec(spec_file, surfs, **kwargs):
    kwargs = dict(dict(type='FS', state=None), **kwargs)
    surfs = [dict(kwargs, **({'name': surf} if isinstance(surf, six.string_types) else surf)) for surf in surfs]
    for surf in surfs:
        if surf['state'] is None:
            surf['state'] = re.search(r'[l|r]h\.(.+)\.asc', surf['name']).group(1)
    cmds = []
    for surf in surfs:
         cmds.extend(['-tsn', surf['type'], surf['state'], surf['name']])
    subprocess.check_call(['quickspec', '-spec', spec_file, '-overwrite'] + cmds)


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


def add_colormap(cmap, name=None, cyclic=False, index=None):
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
        subprocess.check_call(['MakeColorMap', '-f', temp_file, '-ah', name] +
            (['-nc', str(128), '-sl'] if cyclic else ['-nc', str(129)]), stdout=fout)
    os.remove(temp_file)
    # Update .afnirc
    update_afnirc(**{'AFNI_COLORSCALE_{0:02d}'.format(index): path.relpath(cmap_file, path.expanduser('~'))})


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
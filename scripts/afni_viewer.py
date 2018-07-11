#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import argparse, textwrap, subprocess, glob, re, os
from os import path

# 1. Clusterize
# https://afni.nimh.nih.gov/afni/community/board/read.php?1,137288,137299
# You can't drive the clusterize plugin directly yet...
# 2. Layout
# `afni -layout -` is required to suppress the default layout (open axi/sag/cor image)
# 3. DriveSuma
# `-switch_surf lh.inflated` is required to switch state

def parse_window_mini_language(desc):
    info = {}
    for item in desc.split():
        key, value = item.split('=')
        if key == 'geom':
            match = re.match('(?:(.+)x(.+))?([+|-].+)([+|-].+)', value)
            if match:
                if match.group(1) is not None:
                    info['w'] = int(match.group(1))
                    info['h'] = int(match.group(2))
                info['x'] = int(match.group(3))
                info['y'] = int(match.group(4))
            match = re.match('(.+)x(.+)', value)
            if match:
                info['w'] = int(match.group(1))
                info['h'] = int(match.group(2))
        else:
            info[key] = value
    return info


if __name__ == '__main__':
    import script_utils # Append mripy to Python path
    from mripy import afni

    CONTROLLER_HEIGHT = 412
    WINDOW_OFFSET = 426

    parser = argparse.ArgumentParser(description='Open afni/suma to view results with least possible clicks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            examples:
              1) ULay = SurfVol_Alnd_Exp+orig
                 OLay = stats.*REML+orig'[RetinoL-R#0_Coef]'
                 Thr = stats.*REML+orig'[RetinoL-R#0_Tstat]'
                 p = 0.05, clim = 1,
                 view axialimage and sagittalgraph, move Xhairs to I J K
                 press t to link suma.
                $ afni_viewer.py -l RetinoL-R -a -S -ijk 82 180 167 --link
              2) ULay = T1+orig
                 OLay = stats+orig'[L-R#0_Tstat]'
                 Thr = stats+orig'[L+R#0_Tstat]'
                 view all three image, no suma.
                $ afni_viewer.py -u T1+orig -o stats+orig -v L-R#t -t L+R#t -suma off\
            '''))
    parser.add_argument('-u', '--ulay', default='infer', help='underlay dset, default SurfVol_Alnd_Exp+orig > T1_al+orig')
    parser.add_argument('-o', '--olay', default='infer', help='overlay dset, default stats.*REML+orig')
    parser.add_argument('-l', '--label', help='sub-brick label for OLay/Thr')
    parser.add_argument('-v', '--value', default='infer', help='sub-brick index/label for OLay')
    parser.add_argument('-t', '--threshold', default='infer', help='sub-brick index/label for Thr')
    parser.add_argument('-p', default='0.05', help='p value corresponding to overlay threshold')
    parser.add_argument('-r', '--range', default='1', help='color range')
    parser.add_argument('-m', '--resample', default='NN', help='resampling mode for OLay/Thr, default NN: {NN|Li|Cu|Bk}[.{NN|Li|Cu|Bk}]')
    parser.add_argument('-no', '--olay_off', action='store_true', help='hide overlay')
    parser.add_argument('-a', '--axialimage', nargs='?', const='', help='show axial image: -a geom=800x500+900+0 crop=260:330,240:310')
    parser.add_argument('-s', '--sagittalimage', nargs='?', const='', help='show sagittal image')
    parser.add_argument('-c', '--coronalimage', nargs='?', const='', help='show coronal image')
    parser.add_argument('-A', '--axialgraph', nargs='?', const='', help='show axial graph')
    parser.add_argument('-S', '--sagittalgraph', nargs='?', const='', help='show sagittal graph')
    parser.add_argument('-C', '--coronalgraph', nargs='?', const='', help='show coronal graph')
    parser.add_argument('-ijk', '--ijk', nargs=3, metavar=('I', 'J', 'K'), help='crosshairs coordinates in voxel indices')
    parser.add_argument('-xyz', '--xyz', nargs=3, metavar=('X', 'Y', 'Z'), help='crosshairs coordinates in DICOM order')
    parser.add_argument('-suma', '--suma', nargs='?', const='', help='show suma, with "geom=800x500+900+0 state=inflated"')
    parser.add_argument('--suma_dir', default='../SUMA', help='path to SUMA dir')
    parser.add_argument('--spec', default='', help='spec file')
    parser.add_argument('--surf_vol', default='infer', help='spec file')
    parser.add_argument('--viewer_cont', default='', help='additional viewer control sub-commands. --viewer_cont="..."')
    parser.add_argument('--camera', default='-key:r7 up -key:r36 right', help='rotate suma camera with arrow keys')
    parser.add_argument('-link', '--link', action='store_const', const='-key t', default='', help='link suma with afni with t key')
    parser.add_argument('--surf_cont', default='', help='additional surface control sub-commands. --surf_cont="..."')
    parser.add_argument('--controller', action='store_true', help='show object controller')
    parser.add_argument('-d', '--dset', help='surface dset')
    parser.add_argument('--dry', action='store_true', help='print cmd without actual execution')
    args = parser.parse_args()

    if args.ulay == 'infer':
        if path.exists('SurfVol_Alnd_Exp+orig.HEAD'):
            args.ulay = 'SurfVol_Alnd_Exp+orig'
            if args.suma is None:
                args.suma = ''
        elif path.exists('T1_al+orig.HEAD'):
            args.ulay = 'T1_al+orig'
        else:
            raise ValueException('-u ULAY is required.')
    if args.olay == 'infer':
        dsets = glob.glob('stats.*REML+orig.HEAD')
        if dsets:
            args.olay = dsets[0]
        elif args.olay_off:
            args.olay = args.ulay
        else:
            raise ValueException('-o OLAY is required.')
    label2index = afni.get_brick_labels(args.olay, label2index=True)
    n_bricks = len(label2index)
    if args.value == 'infer':
        if args.label is not None:
            args.value = args.label + '#0_Coef'
        else:
            args.value = '{0}'.format(n_bricks-3) # Presumably beta value
    args.value = re.sub('#b$', '#0_Coef', args.value)
    if args.value in label2index:
        args.value = label2index[args.value]
    if args.threshold == 'infer':
        if args.label is not None:
            args.threshold = args.label + '#0_Tstat'
        else:
            args.threshold = '{0}'.format(n_bricks-2) # Presumably t value
    args.threshold = re.sub('#t', '#0_Tstat', args.threshold)
    if args.threshold in label2index:
        args.threshold = label2index[args.threshold]
    if (args.axialimage is None and args.sagittalimage is None and args.coronalimage is None and
        args.axialgraph is None and args.sagittalgraph is None and args.coronalgraph is None):
        args.axialimage = ''
        args.sagittalimage = ''
        args.coronalimage = ''

    # print(args)
    environ = {'AFNI_NOSPLASH': 'YES'}
    if args.olay_off:
        olay_cmd = 'SET_FUNC_VISIBLE A.-'
    else:
        olay_cmd = 'SET_FUNC_VISIBLE A.+; \
            OPEN_PANEL A.Define_Overlay; \
            SET_OVERLAY A.{olay} {value} {threshold}; \
            SET_THRESHNEW A {p} *p; \
            SET_FUNC_RANGE A.{range}; \
            SET_FUNC_RESAM A.{resample}'.format(
            olay=afni.get_prefix(args.olay),
            value=args.value,
            threshold=args.threshold,
            p=args.p,
            range=args.range,
            resample=args.resample)
    win_names = ['axialimage', 'sagittalimage', 'coronalimage', 'axialgraph', 'sagittalgraph', 'coronalgraph']
    win_cmds = []
    win_x = 0
    for name in win_names:
        if args.__getattribute__(name) is not None:
            cmd = 'OPEN_WINDOW A.{0} {1}'.format(name, args.__getattribute__(name))
            if 'geom' not in cmd:
                cmd += 'geom=+{0}+{1}'.format(win_x, CONTROLLER_HEIGHT)
            win_x += WINDOW_OFFSET
            win_cmds.append(cmd)
    win_cmd = '; '.join(win_cmds)
    if args.xyz is not None:
        xyz_cmd = 'SET_DICOM_XYZ A {0}'.format(' '.join(args.xyz))
    elif args.ijk is None:
        xyz_cmd = ''
    elif args.ijk == ['q', 'c', 'c']:
        xyz_cmd = ''
    else:
        xyz_cmd = 'SET_IJK A {0}'.format(' '.join(args.ijk))
    # `afni -layout -` is required to suppress the default layout (open axi/sag/cor image)
    afni_cmd = 'afni -layout - -niml -com "\
        {win_cmd}; \
        SET_UNDERLAY A.{ulay} 0; \
        {olay_cmd}; \
        {xyz_cmd}" &'.format(
        ulay=afni.get_prefix(args.ulay),
        olay_cmd=olay_cmd,
        win_cmd=win_cmd,
        xyz_cmd=xyz_cmd)
    print(afni_cmd)
    if not args.dry:
        subprocess.Popen(afni_cmd, env=dict(os.environ, **environ), shell=True) # subprocess.call(cmd, shell=True)

    if args.suma is not None and args.suma.lower() not in ['none', 'off'] :
        if not args.spec or not path.exists(path.join(args.suma_dir, args.spec)):
            suma_subj = afni.get_suma_subj(args.suma_dir)
            args.spec = '{0}/{2}{1}_both.spec'.format(args.suma_dir, suma_subj, args.spec)
        if args.surf_vol == 'infer':
            args.surf_vol = args.ulay
        suma_info = dict(w=800, h=500, x=900, y=0, state='inflated')
        suma_info.update(parse_window_mini_language(args.suma))
        if not re.search('\b[lrm]h\b', suma_info['state']):
            suma_info['state'] = 'lh.' + suma_info['state'] # Only `-switch_surf lh.inflated`
        if not args.dset is None:
            args.surf_cont = '-load_dset {0} '.format(args.dset) + args.surf_cont
            args.controller = True
        if args.controller:
            args.surf_cont = '-view_object_cont y ' + args.surf_cont
        suma_cmd = 'suma \
            -spec {spec_file} \
            -sv {surf_vol} \
            -drive_com "\
            -com viewer_cont -viewer_size {viewer_size} -viewer_position {viewer_pos} \
            {viewer_cont} {camera} {link} \
            -com surf_cont -switch_surf {state} {surf_cont} \
            " &'.format(
            spec_file=args.spec,
            surf_vol=args.surf_vol,
            viewer_size='{0} {1}'.format(suma_info['w'], suma_info['h']),
            viewer_pos='{0} {1}'.format(suma_info['x'], suma_info['y']),
            state=suma_info['state'],
            viewer_cont=args.viewer_cont,
            camera=args.camera,
            link=args.link,
            surf_cont=args.surf_cont)
        # suma -spec ../SUMA/YangY_both.spec -sv SurfVol_Alnd_Exp+orig -drive_com "-com viewer_cont -key:r10 right" &
        print(suma_cmd)
        if not args.dry:
            subprocess.call(suma_cmd, shell=True)

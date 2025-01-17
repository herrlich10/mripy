#!/usr/bin/env python
import sys, os, shutil, os.path as path, fnmatch, subprocess

package = 'mripy'

def clean():
    for folder in ['dist', 'build', f'{package}.egg-info']:
        if path.exists(folder):
            shutil.rmtree(folder)

def build(force=False): # Build release
    if not path.exists('build') or force:
        clean()
        subprocess.run('python setup.py sdist bdist_wheel', shell=True)

def ignored(fname, ignore):
    for pattern in ignore:
        if fnmatch.fnmatch(fname, pattern):
            return True
        if fnmatch.fnmatch(path.basename(fname), pattern):
            return True
    return False

def find_missing_files(src_dir, dst_dir):
    root = path.basename(src_dir)
    ignore = ['__pycache__', '*.pyc', '.DS_Store']
    missing = []
    for curr, dirs, files in os.walk(src_dir):
        curr = path.relpath(curr, src_dir)
        if not ignored(curr, ignore):
            for fname in files:
                if not ignored(fname, ignore):
                    if not path.exists(path.join(dst_dir, curr, fname)):
                        missing.append(path.join(root, curr, fname))
    return missing

def increase_version():
    version_file = f"{package}/__version__"
    with open(version_file) as f:
        version = f.readline().strip().split('.')
    version[-1] = str(int(version[-1])+1)
    with open(version_file, 'w') as f:
        f.write(f"{'.'.join(version)}\n")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''

    if mode == 'release':
        build()
        print(f">> Uploading to PyPI ...")
        subprocess.run('python -m twine upload dist/*', shell=True, check=True)
        increase_version()
        clean()
    elif mode == 'test':
        build()
        print(f">> Uploading to test.pypi.org ...")
        subprocess.run('python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*', shell=True)
        clean()
    elif mode == 'clean':
        clean()
    else:
        build(force=True)
        missing = find_missing_files(f'{package}', f'build/lib/{package}')
        if missing:
            print(f"** The following files are not properly packaged:")
            for fname in missing:
                print(f"\t{fname}")
        else:
            print(f">> Build finished.")

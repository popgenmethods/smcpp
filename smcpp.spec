# -*- mode: python -*-
from tempfile import NamedTemporaryFile

block_cipher = None

def Entrypoint(dgns,
               scripts=None, pathex=None, hiddenimports=[],
               hookspath=None, excludes=None, runtime_hooks=None):
    import pkg_resources

    # get toplevel packages of distribution from metadata
    def get_toplevel(dist):
        distribution = pkg_resources.get_distribution(dist)
        if distribution.has_metadata('top_level.txt'):
            return list(distribution.get_metadata('top_level.txt').split())
        else:
            return []

    packages = hiddenimports
    for distribution in hiddenimports:
        packages += get_toplevel(distribution)

    scripts = scripts or []
    pathex = pathex or []
    for dist, group, name in dgns:
        # get the entry point
        ep = pkg_resources.get_entry_info(dist, group, name)
        # insert path of the egg at the verify front of the search path
        pathex = [ep.dist.location] + pathex
        # script name must not be a valid module name to avoid name clashes on import
        fh = NamedTemporaryFile(delete=False)
        print "creating script for entry point", dist, group, name
        fh.write("import {0}\n".format(ep.module_name))
        fh.write("{0}.{1}()\n".format(ep.module_name, '.'.join(ep.attrs)))
        for package in packages:
            fh.write("import {0}\n".format(package))
        scripts.append(fh.name)
    print(scripts)
    return Analysis(scripts, pathex, hiddenimports, hookspath, excludes, runtime_hooks)

a = Entrypoint([['smcpp', 'gui_scripts', 'smc++-gui']],
             pathex=['/export/home/terhorst/Dropbox/Berkeley/Research/psmc++/psmcpp'],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[])
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='smc++',
          debug=False,
          strip=False,
          upx=True,
          console=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='smc++')

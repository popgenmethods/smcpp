# -*- mode: python -*-
import sys
import gooey
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'), prefix = 'gooey/languages')
gooey_images = Tree(os.path.join(gooey_root, 'images'), prefix = 'gooey/images')

block_cipher = None

a = Analysis([os.path.join('scripts', 'smc++-gui-script.py')],
             pathex=[os.getcwd()],
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          gooey_languages,
          gooey_images,
          name='smc++',
          debug=False,
          strip=False,
          upx=True,
          icon=os.path.join(gooey_root, 'images', 'program_icon.ico'),
          console=False)

if sys.platform == 'darwin':
    app = BUNDLE(exe, name='SMC++.app', icon=None)

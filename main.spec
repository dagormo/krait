# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\david.moore\\OneDrive - Thermo Fisher Scientific\\Desktop\\Cluster\\MordredVersion\\UI\\resources', 'resources')]
binaries = []
hiddenimports = ["sklearn","sklearn.ensemble._hist_gradient_boosting.gradient_boosting","sklearn.ensemble._hist_gradient_boosting.predictor","sklearn._loss"]
tmp_ret = collect_all('mordred')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\david.moore\\OneDrive - Thermo Fisher Scientific\\Desktop\\Cluster\\MordredVersion\\UI\\main_with_multistep.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['C:\\Users\\david.moore\\OneDrive - Thermo Fisher Scientific\\Desktop\\Cluster\\MordredVersion\\UI\\hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)

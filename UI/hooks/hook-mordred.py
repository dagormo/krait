from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# include every Python submodule under mordred
hiddenimports = collect_submodules("mordred")

# include any non-.py data files that Mordred needs
datas = collect_data_files("mordred")

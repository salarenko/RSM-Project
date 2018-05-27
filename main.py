from modules.components.file_loader import load_file
from modules.normal_module import run_normal_module
from modules.pca_module import run_pca_module
from setup import run_normal_mode, run_pca_mode

if __name__ == '__main__':
    loadedData = load_file()

    if run_pca_mode:
        run_pca_module(loadedData)
    if run_normal_mode:
        run_normal_module(loadedData)
conda create -n virtues_2 python=3.12

conda activate virtues_2

conda run -n virtues_2 pip install numpy
conda run -n virtues_2 pip install pandas
conda run -n virtues_2 pip install einops
conda run -n virtues_2 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda run -n virtues_2 pip install biopython
conda run -n virtues_2 pip3 install -U scikit-learn
conda run -n virtues_2 pip install -U matplotlib
conda run -n virtues_2 pip install seaborn
conda run -n virtues_2 pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
conda run -n virtues_2 pip install wandb
conda run -n virtues_2 pip install pillow
conda run -n virtues_2 pip install umap-learn
conda run -n virtues_2 pip install POT
conda run -n virtues_2 pip install loguru
conda run -n virtues_2 pip install omegaconf imblearn
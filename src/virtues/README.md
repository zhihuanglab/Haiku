# VirTues
## AI-powered virtual tissues from spatial proteomics for clinical diagnostics and biomedical discovery

*[[Preprint]](https://arxiv.org/pdf/2501.06039), [[Supplement]](https://nbviewer.org/github/bunnelab/virtues/blob/main/.github/supplement.pdf), 2025* <img src=".github/logo_virtues.png" alt="VirTues Logo" width="40%" align="right" />

**Authors:** Johann Wenckstern*, Eeshaan Jain*, Kiril Vasilev, Matteo Pariset, Andreas Wicki, Gabriele Gut, Charlotte Bunne

Spatial proteomics technologies have transformed our understanding of complex tissue architectures by enabling simultaneous analysis of multiple molecular markers and their spatial organization. The high dimensionality of these data, varying marker combinations across experiments and heterogeneous study designs pose unique challenges for computational analysis. Here, we present Virtual Tissues (VirTues), a foundation model framework for biological tissues that operates across the molecular, cellular and tissue scale. VirTues introduces innovations in transformer architecture design, including a novel tokenization scheme that captures both spatial and marker dimensions and attention mechanisms that scale to high-dimensional multiplex data while maintaining interpretability. Trained on diverse cancer and non-cancer tissue datasets, VirTues demonstrates strong generalization capabilities without task-specific fine-tuning, enabling cross-study analysis and novel marker integration. As a generalist model, VirTues outperforms existing approaches across clinical diagnostics, biological discovery and patient case retrieval tasks, while providing insights into tissue function and disease mechanisms.

<br>
<p align='center'>
<img src=".github/abstract_virtues.png" alt="VirTues Graphical Abstract" width="80%" />
</p>

## Installation
To create a new conda environment `virtues` with Python 3.12 and install all requirements run:
```
source setup.sh
```

## Getting Started

### Configuration
Before running VirTues, please ensure that your base configuration found in `configs/base_config` is properly setup for your system. 
This includes setting the following fields:
```yaml
experiment.disable_wandb: 'disabled' | 'online' | 'offline' # set to 'disabled' to disable wandb logging
experiment.wandb_entity: <entity-name> # your wandb entity name, leave empty for default
experiment.wandb_project: <project-name> # your project name
dataset.path: /path/to/dataset # directory containing individual dataset folders
esm.encoding_dir: /path/to/esm_embeddings # directory containing protein embeddings as [UNIPROT-ID].pt files
```

### Datasets 
Datasets used in the paper will be made available upon publication.

To setup a new dataset, we recommend following our file structure:
```z
dataset.path/
├──[CUSTOM]/
│   ├──images/ # multiplexed images without processing, names must match 'image_name' columns in annotations
│   │  ├──A0001.npy
│   │  ├──A0002.npy
│   │  ├──...
│   ├──masks/ # cell segmentations masks, names must match 'image_name' columns in annotations
│   │  ├──A0001.npy
│   │  ├──A0002.npy
│   │  ├──...
│   ├──clinical.csv # image-wise annotations, must contain column 'image_name'
│   ├──sce_annotations.csv # cell-wise annotations, must contain columns 'image_name' and 'cell_id'
...
```
Further protein embeddings need to be added to `esm.encoding_dir` and a table `gene_dict_[CUSTOM].csv` containing for each  channel (in the correct order of measurement) a name and a UniProt ID needs to be added to `./metadata/[CUSTOM]/`.

### Training 
After setting up the datasets, VirTues can be pretrained via the `src/train_virtues.py` script. For example, to train an instance of VirTues on the Danenberg et al. dataset run: 
```bash
python -m src.train_virtues experiment.name=[NAME] --dataset.name=danenberg
```

VirTues can also be pretrained on multiple datasets at once. For instance, the following command executes training on Danenberg et al. and Jackson et al.:
```bash
python -m src.train_virtues experiment.name=[NAME] --dataset.union_list=[danenberg,jacksonfischer]
```
All the training results are stored in the `expt/` directory.

### Evaluation
Once VirTues has been pretrained, with `experiment.name` as [NAME], it will be stored in `expt/[NAME]`. To run downstream task on a dataset, run 
```bash
python -m src.train_downstream experiment.name=[NAME] dataset.name=danenberg downstream.task_level=image
```
The above will run all image level downstream tasks sequentially on the danenberg dataset. Tasks can be image level (tissue), crop level (niche) or patch level (cellular). Image and crop level tasks use ABMIL, while patch level tasks use linear probing.

The datasets have the following tasks:
```
lung: image | patch
danenberg: image | crop | patch
jacksonfischer: image
hochschulz: image | patch
damond: image | patch
```

All the downstream evaluation results are stored in the `downstream_expt/` directory.


## Acknowledgements
The project was built on top of amazing repositories such as [PyTorch](https://github.com/pytorch/pytorch) (v2.5.1, CUDA 12.1), [xformers](https://github.com/facebookresearch/xformers) (v0.0.28) and [scikit-learn](https://github.com/scikit-learn/scikit-learn) (v1.5.2). We thank the authors and developers for their contribution.

## License and Terms of Use

ⓒ AIMM Lab. This model and associated code are released under the [CC-BY-NC-ND 4.0]((https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en)) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the VirTues platform and its derivatives, which include models trained on outputs from the VirTues platform, is prohibited and requires prior approval.

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://arxiv.org/abs/2501.06039):

```
@article{wenckstern2025ai,
  title={{AI-powered virtual tissues from spatial proteomics for clinical diagnostics and biomedical discovery}},
  author={Wenckstern, Johann and Jain, Eeshaan and Vasilev, Kiril and Pariset, Matteo and Wicki, Andreas and Gut, Gabriele and Bunne, Charlotte},
  journal={arXiv preprint arXiv:2501.06039},
  year={2025},
  url={https://arxiv.org/abs/2501.06039}, 
}
```

import numpy as np
from utils.downstream_utils import load_virtues, load_train_test_tokens, clean_data, setup_data_for_abmil, \
                                         set_seed, SubSamplingMILDataset, setup_and_return_abmil_wandb, print_all_metrics, \
                                         dump_train_test_tokens, modify_sv_dict, load_patch_tokens, add_report_to_dict
from dataset.imc_dataset import get_imc_dataset
from dataset.imc_base import CoordinateDumper, PatchEvalDataset
from models.ABMIL.gated_abmil import GatedABMILClassifierWithValidation
from models.LinearProbe.linear_probe import run_linear_probe
from torch.utils.data import DataLoader
import torch
import wandb 
import sys
import os
from loguru import logger
import warnings
from models.virtues.mae import VirTuesMAE
from utils.esm_utils import load_esm_embeddings
from omegaconf import OmegaConf
import pickle
warnings.filterwarnings("ignore")
import json
from tqdm import tqdm
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


if __name__ == '__main__':

    conf = OmegaConf.load("configs/base_config.yaml")

    cli_conf = OmegaConf.from_cli()

    if hasattr(cli_conf, 'base') and cli_conf.base.additional_config is not None:
        additional_conf = OmegaConf.load(cli_conf.base.additional_config)
        conf = OmegaConf.merge(conf, additional_conf)

    if os.path.exists(f'{conf.experiment.dir}/{conf.experiment.name}/config.pkl'):
        expt_config = pickle.load(open(f'{conf.experiment.dir}/{conf.experiment.name}/config.pkl', 'rb'))
        conf = OmegaConf.merge(conf, expt_config)

    conf = OmegaConf.merge(conf, cli_conf)

    assert conf.downstream.task_level in {'image', 'crop', 'patch'}
    assert conf.downstream.task_level == 'image' if conf.dataset.name == 'jacksonfischer' else True
    logger.info(f'Running downstream task with config: {conf}')


    imc_dataset = get_imc_dataset(conf, filter_channel_names=conf.dataset.filter_channels)

    conf.downstream.dir = f'{conf.downstream.dir}/{conf.experiment.name}' 

    os.makedirs(f'{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}', exist_ok=True)
    handler_id = logger.add(f'{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}/training.log')
    mae_model = load_virtues(conf, run_name=conf.experiment.name)

    if conf.downstream.task_level == 'image' or conf.downstream.task_level == 'crop':
        try:
            logger.info('Loading train test tokens')
            sv_dict = load_train_test_tokens(save_path=f'{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}', mode='cls', imc_dataset_name=conf.dataset.name)
            logger.info('Loaded train test tokens')
        except:
            logger.info('Did not find train test tokens')
            logger.info(f'Generating train test tokens for {conf.dataset.name}')

            sv_dict = dump_train_test_tokens(
                conf, imc_dataset, mae_model, channel_names=None, save_path=f'{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}', mode='cls',
                task_level=conf.downstream.task_level)
            
        logger.info(f'Keys: {sv_dict.keys()}')
        logger.info(f'Metadata Keys: {sv_dict["test_metadata"][0].keys()}')

        if conf.downstream.task_level == 'image':
            if conf.dataset.name == 'lung':
                all_task_names = ['cancer_type', 'grade', 'Relapse']
            elif conf.dataset.name == 'danenberg':
                all_task_names = ['ERStatus', 'PAM50', 'grade']
            elif conf.dataset.name == 'jacksonfischer':
                all_task_names = ['tumor_grade', 'tumor_type', 'tumor_clinical_type']
            elif conf.dataset.name == 'hochschulz':
                all_task_names = ['patient_cancer_stage', 'patient_relapse', 'patient_mutation']
            elif conf.dataset.name == 'damond':
                all_task_names = ['stage', 'aab_status']


        else:
            if conf.dataset.name == 'danenberg':
                all_task_names = [
                    'Suppressed expansion',
                    'TLS-like',
                    'PDPN^{+} active stroma'
                ]
            else:
                raise ValueError('Niche level tasks are only available for danenberg dataset')
            
        
            sv_dict = modify_sv_dict(sv_dict, all_task_names)


        all_nan_pool = {
            'lung': {
                'cancer_type': {'nan', 'Adeno squamous cell carcinoma',
                    'Large cell carcinoma',
                    'NSCLC',
                    'Mesotheliom',
                    'Basaloides Ca',
                    'Large cell neuroendocrine carcinoma'},
                
                'grade': {'nan'},

                'Relapse': {'nan'},
            },

            'jacksonfischer': {
                'tumor_grade': {'nan'},
                'tumor_type': {'nan', '[]'},
                'tumor_clinical_type': {'nan'},
            },
            'hochschulz': {
                'patient_cancer_stage': {'nan', 'III or IV', 'unknown'},
                'patient_relapse': {'nan', 'untreated/lost'},
                'patient_mutation': {'nan', 'unknown'},
            },
            'damond': {
                'stage': {'nan', 'None'},
                'aab_status': {'nan', 'None'}
            },
            'danenberg': {
                'ERStatus': {'nan', 'unknown', 'None'},
                'PAM50': {'nan', 'unknown', 'None'},
                'grade': {'nan', 'unknown', 'None'},
                'Suppressed expansion': {'nan', 'None'},
                'TLS-like': {'nan', 'None'},
                'PDPN': {'nan', 'None'},
            }

        }

        nan_pool = all_nan_pool[conf.dataset.name]
        filter_functions = {
            k: lambda _: True for k in all_task_names
        }

        logger.info(f'Running downstream task for {conf.dataset.name}')
        logger.info(f'Nan pool: {nan_pool}')
        logger.info(f'Filter functions: {filter_functions}')


        for task_name in all_task_names:
            if 'PDPN' in task_name:
                task_name = 'PDPN'

            logger.info(f'Running task: {task_name}')
            logger.info('Cleaning data')
            clean_sv_dict = clean_data(task_name, sv_dict, nan_pool[task_name], filter_fn=filter_functions[task_name])
            logger.info('Setting up data for ABMIL')
            train_X, train_y, test_X, test_y, all_y, le = setup_data_for_abmil(clean_sv_dict, task_name, conf.dataset.name)
            for seed in range(conf.downstream.num_seeds):
                run_name = f'{conf.experiment.name}_{conf.dataset.name}_{task_name}_seed={seed}'
            
                group_name = f'{conf.dataset.name}_{task_name}'
   
                os.makedirs(f'{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}/{task_name}', exist_ok=True)

                logger.info(f'Run name: {run_name}\tGroup name: {group_name}')
   
                run = setup_and_return_abmil_wandb(conf, f'VirTues_{conf.downstream.task_level}_level', run_name, group_name)
                extra_configs = {
                    'num_epochs': conf.downstream.epochs,
                    'task_name': task_name,
                    'seed': seed,
                    'model_name': conf.experiment.name,
                }
                wandb.config.update(extra_configs)
                
                set_seed(seed)

                abmil_model = GatedABMILClassifierWithValidation(input_dim=train_X[0].shape[1],
                                                                    hidden_dim=256, num_heads=4,
                                                                        num_classes=len(all_y),
                                                                        wandb_log=True,
                                                                        save_path=f'{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}/{task_name}',
                                                                        name=run_name)
            
                train_dataset = SubSamplingMILDataset(train_X, train_y)
                val_dataset = SubSamplingMILDataset(train_X, train_y, val=True)
                test_dataset = SubSamplingMILDataset(test_X, test_y, val=True)

                train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
                val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=val_dataset.collate_fn)
                test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=test_dataset.collate_fn)

                loss_fn = torch.nn.CrossEntropyLoss() if len(all_y) > 2 else torch.nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(abmil_model.parameters(), lr=1e-4)
                logger.info('Training ABMIL model')
                abmil_model.train_model(train_dl, val_dl, conf.downstream.epochs, optimizer, loss_fn, test_dl=test_dl,
                                        monitor="valid_loss")

                logger.info('Getting predictions on test set')
                result_dict = abmil_model.get_outputs(test_dl, load_best=True)
                final_metrics = print_all_metrics(result_dict['predictions'], result_dict['ground_truth'])
                final_metrics = {f'Final_test_{k}': v for k, v in final_metrics.items()}
                wandb.log(final_metrics)
                logger.info('Finished training')
                logger.info(f'Saving metrics to {conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}/{task_name}/{run_name}.json')
                with open(f'{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}/{task_name}/{run_name}.json', 'w') as f:
                    json.dump(final_metrics, f)

                run.finish()


    else:
        if conf.dataset.name == 'hochschulz' or conf.dataset.name == 'damond':
            if not os.path.exists(os.path.join(f'dumps/{conf.dataset.name}/patch_coordinates_train.pt')) and not os.path.exists(os.path.join(f'dumps/{conf.dataset.name}/patch_coordinates_test.pt')):
                logger.info(f'Generating patch coordinates for {conf.dataset.name}')
                train_coordinate_dumper = CoordinateDumper(conf, imc_dataset, image_section_size=conf.image_info.image_section_size, patch_size=conf.image_info.patch_size, split='train')
                test_coordinate_dumper = CoordinateDumper(conf, imc_dataset, image_section_size=conf.image_info.image_section_size, patch_size=conf.image_info.patch_size, split='test')

                coordinates = []
                for i in tqdm(range(len(train_coordinate_dumper))):
                    coordinates.extend(train_coordinate_dumper[i])

                torch.save(coordinates, os.path.join(f'dumps/{conf.dataset.name}/patch_coordinates_train.pt'))

                coordinates = []
                for i in tqdm(range(len(test_coordinate_dumper))):
                    coordinates.extend(test_coordinate_dumper[i])

                torch.save(coordinates, os.path.join(f'dumps/{conf.dataset.name}/patch_coordinates_test.pt'))

        train_patch_eval_dataset = PatchEvalDataset(conf, imcdataset=imc_dataset, image_section_size=conf.image_info.image_section_size, patch_size=conf.image_info.patch_size, split='train')
        test_patch_eval_dataset = PatchEvalDataset(conf, imcdataset=imc_dataset, image_section_size=conf.image_info.image_section_size, patch_size=conf.image_info.patch_size, split='test')

            
        train_patches, train_patch_metadata, test_patches, test_patch_metadata = load_patch_tokens(conf, imc_dataset, train_patch_eval_dataset, test_patch_eval_dataset, mae_model) 


        print(train_patch_metadata.keys())

        if conf.dataset.name == 'lung':
            all_task_names = ['predominant_cell_category', 'predominant_cell_type']
            all_column_names = ['cell_category', 'cell_type']
            labels_to_filter = [
                ['None', 'Other'],
                ['None', 'nan']
            ]
            samplers = [None, None]

        elif conf.dataset.name == 'danenberg':
            all_task_names = ['predominant_cell_category']
            all_column_names = ['cell_category']
            labels_to_filter = [
                ['None', 'nan', 'Antigen-Presenting Cell']
            ]
            samplers = [None]

        elif conf.dataset.name == 'hochschulz':
            all_task_names = ['cell_category']
            all_column_names = ['cell_category']
            labels_to_filter = [
                ['None', 'Other']
            ]
            samplers = [None]

        elif conf.dataset.name == 'damond':
            all_task_names = ['cell_category']
            all_column_names = ['cell_category']
            labels_to_filter = [
                ['None', 'other', 'unknown']
            ]

            samplers = [
                RandomUnderSampler(
                    sampling_strategy={
                        imc_dataset.cell_type_to_id(column_name='cell_category')['immune']: 33403,
                        imc_dataset.cell_type_to_id(column_name='cell_category')['islet']: 50000,
                        imc_dataset.cell_type_to_id(column_name='cell_category')['exocrine']: 100000
                    },
                    random_state=0
                )
            ]

            

        result_dicts = []
        for task_name, column_name, filters, sampler in zip(all_task_names, all_column_names, labels_to_filter, samplers):
            FILTERS = [imc_dataset.cell_type_to_id(column_name=column_name)[x] for x in filters]
            train_labels = np.array(train_patch_metadata[task_name])
            test_labels = np.array(test_patch_metadata[task_name])
            id_to_label_dict = imc_dataset.id_to_cell_type(column_name=column_name)

            lr_model, train_report, test_report, test_confusion_matrix = run_linear_probe(train_patches, train_labels, test_patches, test_labels, filter_labels=FILTERS, id_to_name_dict=id_to_label_dict, return_confusion_matrix=True,
                                                                                          sampler=sampler, sample_test_set=False)
            add_report_to_dict(test_report, result_dicts, 'virtues+linear_probe', task_name, 0)


        results_df = pd.DataFrame(result_dicts)
        results_df.to_csv(f'{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}/results.csv')







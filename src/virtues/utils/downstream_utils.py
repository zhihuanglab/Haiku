import random
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from models.virtues.mae import VirTuesMAE
from utils.esm_utils import load_esm_embeddings
import wandb
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import os
from tqdm import tqdm
from dataset.imc_base import ImageEvalDataset, CropEvalDataset, PatchEvalDataset, PatchEvalDatasetFixedCoordinates, CoordinateDumper
    
class SubSamplingMILDataset(Dataset):

    def __init__(self, bags, labels, val=False, max_bag_size=500):
        self.bags = bags
        self.labels = labels

        self.val = val
        self.max_bag_size = max_bag_size

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, index):

        if self.val:
            return self.get_item_val(index)

        bag = self.bags[index]
        label = self.labels[index]

        max_bag_size = min(self.max_bag_size, bag.size(0))

        indices = torch.randperm(bag.size(0))[:max_bag_size]

        return bag[indices], label
    

    def get_item_val(self, index):
        return self.bags[index], self.labels[index]
    
    def collate_fn(self, batch):
        bags, labels = list(map(list, zip(*batch)))
        max_len = max([b.size(0) for b in bags])
        masks = []
        for i in range(len(bags)):
            seq_len = bags[i].size(0)
            token_dim = bags[i].size(1)
            if seq_len < max_len:
                padding = torch.zeros(max_len - seq_len, token_dim)
                bags[i] = torch.cat([bags[i], padding], dim=0)
            mask = torch.concat([torch.zeros(seq_len, dtype=torch.bool), torch.ones(max_len - seq_len, dtype=torch.bool)])
            masks.append(mask)
        bags = torch.stack(bags, dim=0) # B x S x D
        masks = torch.stack(masks, dim=0) # B x S
        labels = torch.tensor(labels) # B
        return bags, masks, labels
    
def load_virtues(conf, run_name, device='cuda', eval_mode=True):
    model_ckpt = f'{conf.experiment.dir}/{run_name}/checkpoints/{run_name}.pt'
    esm_embeddings = load_esm_embeddings(conf)
    mae_model = mae_model = VirTuesMAE(
        protein_emb=esm_embeddings,
        patch_size=conf.image_info.patch_size,
        model_dim=conf.model.dim,
        feedforward_dim=conf.model.feedforward_dim,
        encoder_pattern=conf.model.encoder_pattern,
        num_encoder_heads=conf.model.num_encoder_heads,
        mae_decoder_pattern=conf.model.decoder_pattern,
        mae_num_decoder_heads=conf.model.num_decoder_heads,
        mae_num_hidden_layers_head=conf.model.num_decoder_hidden_layers,
        dropout=conf.training.dropout,
        pos_emb=conf.model.pos_emb
    )
    mae_model.load_state_dict(torch.load(model_ckpt))
    mae_model.eval()
    mae_model.to(device)

    if eval_mode:
        conf.image_info.use_rnd_crop_dir = False

    return mae_model



def dump_train_test_tokens(conf, imc_dataset, model, channel_names, save_path='tmp/', mode='channel', ckpt_path=None,
                           task_level='image'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def channel_tokens_and_metadata(conf, image_eval_dataset, model, device):
        dataloader = DataLoader(image_eval_dataset, batch_size=None, shuffle=False, num_workers=conf.training.num_workers)
        max_chunk_size = conf.training.batch_size * 2

        cls_tokens = []
        metadata = []
        for batch in tqdm(dataloader):

            all_results = []
            crops, channels, md = batch # crops is B x C x GH x GW x D tensor

            for crops_chunk, channels_chunk in zip(torch.split(crops, max_chunk_size, dim=0), torch.split(channels, max_chunk_size, dim=0)):
                crops_chunk = crops_chunk.to(device)
                channels_chunk = channels_chunk.to(device)
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=conf.training.fp16):
                        # results = model.embed(crops_chunk, channels_chunk, return_dict=True, place_on_cpu=True)
                        results = model.encoder(crops_chunk, channels_chunk, mask=None)[0]
                results = results.cpu() 
                all_results.append(results)


            all_patches = torch.cat(all_results, dim=0)
            all_crops = all_patches.mean(dim=(-2,-3))
            cls_tokens.append(all_crops)
            metadata.append(md)
        return cls_tokens, metadata
    

    def compute_cls_tokens_and_metadata(conf, image_eval_dataset, model, device):

        dataloader = DataLoader(image_eval_dataset, batch_size=None, shuffle=False, num_workers=conf.training.num_workers)

        max_chunk_size = conf.training.batch_size * 2

        cls_tokens = []
        metadata = []
        for batch in tqdm(dataloader):

            batch_cls_tokens = []

            crops, channels, md = batch # crops is B x C x GH x GW x D tensor

            for crops_chunk, channels_chunk in zip(torch.split(crops, max_chunk_size, dim=0), torch.split(channels, max_chunk_size, dim=0)):
                crops_chunk = crops_chunk.to(device)
                channels_chunk = channels_chunk.to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=conf.training.fp16):
                        results = model.embed(crops_chunk, channels_chunk, return_dict=True, place_on_cpu=True)
                for res in results: # this should be only one result as input to model is not a list
                    chunk_cls_token = res["patch_summary_token"].float()
                    # print(chunk_cls_token.shape)
                    batch_cls_tokens.append(chunk_cls_token)
            
            batch_cls_tokens = torch.cat(batch_cls_tokens, dim=0)
            cls_tokens.append(batch_cls_tokens)
            metadata.append(md)
        return cls_tokens, metadata


    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))

    model.eval()
    model.to(device)

    train_image_eval = ImageEvalDataset(conf, imcdataset=imc_dataset, split='train', image_section_size=conf.image_info.image_section_size, patch_size=conf.image_info.patch_size)
    test_image_eval = ImageEvalDataset(conf, imcdataset=imc_dataset, split='test', image_section_size=conf.image_info.image_section_size, patch_size=conf.image_info.patch_size)

    if mode == 'channel':
        token_metadata_fn = channel_tokens_and_metadata
    elif mode == 'cls':
        token_metadata_fn = compute_cls_tokens_and_metadata
    else:
        raise ValueError('mode must be one of "channel" or "cls"')

    train_cls_tokens, train_metadata = token_metadata_fn(conf, train_image_eval, model, device)
    test_cls_tokens, test_metadata =   token_metadata_fn(conf, test_image_eval, model, device)

    save_dict = {
        'train_metadata': train_metadata,
        'test_metadata': test_metadata,
        # 'channel_dict': channel_dict
    }

    if mode == 'channel':
        channel_dict = {channel: i for i, channel in enumerate(channel_names)}
        save_dict['channel_dict'] = channel_dict

        for _c in channel_names:
            save_dict[f'train_{_c}_tokens'] = list(map(lambda x: x[:,channel_dict[_c]], train_cls_tokens))
            save_dict[f'test_{_c}_tokens'] = list(map(lambda x: x[:,channel_dict[_c]], test_cls_tokens))

    else:
        save_dict['train_cls_tokens'] = train_cls_tokens
        save_dict['test_cls_tokens'] = test_cls_tokens
    torch.save(save_dict, f'{save_path}/{mode}_{imc_dataset.name}_tokens.pt')

    return save_dict




def load_train_test_tokens(save_path='tmp/', mode='cls', imc_dataset_name='lung'):
    save_dict = torch.load(f'{save_path}/{mode}_{imc_dataset_name}_tokens.pt')
    return save_dict


def filter_nan_tokens(task_name, metadata, nan_pool={'nan'}, filter_function=lambda x: True):
    indices_to_keep = []
    for i in range(len(metadata)):
        if metadata[i][task_name] not in nan_pool and filter_function(metadata[i][task_name]):
            indices_to_keep.append(i)
    return indices_to_keep



def clean_data(task_name, sv_dict, nan_pool, filter_fn=lambda x: True):
    
    train_indices = filter_nan_tokens(task_name, sv_dict['train_metadata'], nan_pool, filter_fn)
    test_indices = filter_nan_tokens(task_name, sv_dict['test_metadata'], nan_pool, filter_fn)

    filtered_train_cls_tokens = [sv_dict['train_cls_tokens'][i] for i in train_indices]
    filtered_test_cls_tokens = [sv_dict['test_cls_tokens'][i] for i in test_indices]

    filtered_train_metadata = [sv_dict['train_metadata'][i] for i in train_indices]
    filtered_test_metadata = [sv_dict['test_metadata'][i] for i in test_indices]

    return {
        'train_cls_tokens': filtered_train_cls_tokens,
        'test_cls_tokens': filtered_test_cls_tokens,
        'train_metadata': filtered_train_metadata,
        'test_metadata': filtered_test_metadata
    }


def reshape_tokens(tokens):
    dim = tokens[0].shape[-1]
    return list(map(lambda x: x.view(-1, dim), tokens))

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def label_convert_function(train_labels, test_labels, task_name, dataset_name='lung'):

    if dataset_name == 'jacksonfischer':
        if task_name == 'tumor_type':
            invasive_ductal_names = {'INVASIVE_DUCTAL', 'INVASIVE_DUCTAL_LOBULAR', 'INVASIV_DUCTAL'}
            rename = lambda x: 'INVASIVE_DUCTAL' if x in invasive_ductal_names else 'NOT_INVASIVE_DUCTAL'
            train_labels = [rename(x) for x in train_labels]
            test_labels = [rename(x) for x in test_labels]
            return train_labels, test_labels
        else:
            return train_labels, test_labels
    
    else:
        return train_labels, test_labels

def get_labels(train_metadata, test_metadata, task_name, dataset_name):
    train_labels = [md[task_name] for md in train_metadata]
    test_labels = [md[task_name] for md in test_metadata]

    return label_convert_function(train_labels, test_labels, task_name, dataset_name)



def setup_data_for_abmil(sv_dict, task_name, dataset_name):
    le = LabelEncoder()
    train_labels, test_labels = get_labels(sv_dict['train_metadata'], sv_dict['test_metadata'], task_name, dataset_name)
    all_classes = set(train_labels + test_labels)
    le.fit(list(all_classes))
    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)

    train_tokens = reshape_tokens(sv_dict['train_cls_tokens'])
    test_tokens = reshape_tokens(sv_dict['test_cls_tokens'])

    return train_tokens, train_labels, test_tokens, test_labels, all_classes, le


def setup_abmil_wandb(conf, project_name, run_name, group_name):
    assert conf.experiment.disable_wandb in ["online", "offline", "disabled"], f"Received {conf.experiment.disable_wandb} for disable_wandb. Must be one of ['online', 'offline', 'disabled']"
    wandb.init(project=project_name, name=run_name, entity=conf.experiment.wandb_entity, group=group_name, mode=conf.experiment.disable_wandb)

def setup_and_return_abmil_wandb(conf, project_name, run_name, group_name):
    assert conf.experiment.disable_wandb in ["online", "offline", "disabled"], f"Received {conf.experiment.disable_wandb} for disable_wandb. Must be one of ['online', 'offline', 'disabled']"
    run = wandb.init(project=project_name, name=run_name, entity=conf.experiment.wandb_entity, group=group_name, reinit=True, mode=conf.experiment.disable_wandb)
    return run


def print_all_metrics(preds, gt):
    accuracy = accuracy_score(gt, preds)
    f1_macro = f1_score(gt, preds, average='macro')

    # precision, recall, f1, roc_auc, average_precision
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1_micro = f1_score(gt, preds, average='micro')
    f1_weighted = f1_score(gt, preds, average='weighted')
    # roc_auc = roc_auc_score(gt, preds)
    # average_precision = average_precision_score(gt, preds)

    # tabulate print
    headers = ['Accuracy', 'F1 Macro', 'Precision', 'Recall', 'F1 Micro', 'F1 Weighted']
    table = [[round(accuracy,3), round(f1_macro,3), round(precision,3), round(recall,3), round(f1_micro,3), round(f1_weighted,3)]]
    print(tabulate(table, headers=headers, tablefmt='pretty'))

    vals = table[0]

    return {k: v for k, v in zip(headers, vals)}



def modify_sv_dict(sv_dict, task_names):
    new_sv_dict = {}
    new_sv_dict['train_metadata'] = []
    new_sv_dict['train_cls_tokens'] = []
    new_sv_dict['test_metadata'] = []
    new_sv_dict['test_cls_tokens'] = []
    for i in range(len(sv_dict['train_metadata'])):
        num_crops = sv_dict['train_metadata'][i]['num_subimages']
        for j in range(num_crops):
            md = {}
            md['image_name'] = sv_dict['train_metadata'][i]['image_name']
            md['crop_num'] = j
            for task_name in task_names:
                md[task_name] = sv_dict['train_metadata'][i][task_name][j]

            new_sv_dict['train_metadata'].append(md)

            new_sv_dict['train_cls_tokens'].append(sv_dict['train_cls_tokens'][i][j])

    for i in range(len(sv_dict['test_metadata'])):
        num_crops = sv_dict['test_metadata'][i]['num_subimages']
        for j in range(num_crops):
            md = {}
            md['image_name'] = sv_dict['test_metadata'][i]['image_name']
            md['crop_num'] = j
            for task_name in task_names:
                md[task_name] = sv_dict['test_metadata'][i][task_name][j]

            new_sv_dict['test_metadata'].append(md)

            new_sv_dict['test_cls_tokens'].append(sv_dict['test_cls_tokens'][i][j])

    return new_sv_dict
        

def load_patches_and_metadata_from_coordinates(conf, model, patch_eval_dataset_fixedcoords, tasks):
    dataloader = DataLoader(patch_eval_dataset_fixedcoords, batch_size=conf.training.batch_size, pin_memory=True, shuffle=False, num_workers=conf.training.num_workers)

    patches = []
    patch_metadata = dict()
    for task in tasks:
        patch_metadata[task.task_name] = []
    patch_metadata["image_name"] = []

    for i, batch in enumerate(tqdm(dataloader)):
        imgs, channels, metadata, patch_indices = batch
        imgs = imgs.cuda()
        channels = channels.cuda()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=conf.training.fp16):
                results = model.embed(imgs, channels, return_dict=True, place_on_cpu=True)
        
        for res in results:
            patch_summary = res["patch_summary_token"] # B x GH x GW x D
            for i in range(patch_summary.shape[0]):
                layer = patch_summary[i] # GH x GW x D
                N = layer.shape[0] * layer.shape[1]
                patch_indices_layer = patch_indices[i]
                patches += list(layer.flatten(start_dim=0, end_dim=-2)[patch_indices_layer])
                patch_metadata["image_name"] += [metadata["image_name"][i]] * len(patch_indices_layer)
                for task in tasks:
                    grid = metadata[task.task_name][i]
                    patch_metadata[task.task_name] += grid.flatten(start_dim=0, end_dim=-1)[patch_indices_layer].tolist()

    patches = torch.stack(patches, dim=0).float()
    return patches, patch_metadata



def load_patch_tokens(conf, imc_dataset, train_patch_eval_dataset, test_patch_eval_dataset, model, force_recompute=False,
                      num_patches_per_crop=100, num_crops_per_img=5):
    name = imc_dataset.name
    embedding_path = f'{conf.downstream.dir}/{conf.dataset.name}/{conf.downstream.task_level}/embeddings'
    if os.path.exists(os.path.join(embedding_path, f"patch_embeddings_train.pt")):
        train_data = torch.load(os.path.join(embedding_path, f"patch_embeddings_train.pt"))
        test_data = torch.load(os.path.join(embedding_path, f"patch_embeddings_test.pt"))
    else:
        os.makedirs(embedding_path, exist_ok=True)
        force_recompute = True
        train_data = {"patch_summary_tokens": [], "reduced_patch_summary_tokens": [], "metadata": []}
        test_data = {"patch_summary_tokens": [], "reduced_patch_summary_tokens": [], "metadata": []}    

    train_tokens = train_data["patch_summary_tokens"]
    train_metadata = train_data["metadata"]

    test_tokens = test_data["patch_summary_tokens"]
    test_metadata = test_data["metadata"]

    if force_recompute:
        if os.path.exists(os.path.join(f'dumps/{conf.dataset.name}/patch_coordinates_train.pt')) and os.path.exists(os.path.join(f'dumps/{conf.dataset.name}/patch_coordinates_test.pt')):
            print('Found coordinates for patches. Using these to compute further with given model.')
            coordinates_train = torch.load(f'dumps/{conf.dataset.name}/patch_coordinates_train.pt')
            coordinates_test = torch.load(f'dumps/{conf.dataset.name}/patch_coordinates_test.pt')

            print('Loading patch embeddings from coordinates.')
            print(f'Number of training patches: {len(coordinates_train)}')
            print(f'Number of testing patches: {len(coordinates_test)}')

            train_patch_eval_dataset_fixedcoords = PatchEvalDatasetFixedCoordinates(conf, coordinates_train, imc_dataset, image_section_size=conf.image_info.image_section_size, patch_size=conf.image_info.patch_size, split='train')
            test_patch_eval_dataset_fixedcoords = PatchEvalDatasetFixedCoordinates(conf, coordinates_test, imc_dataset, image_section_size=conf.image_info.image_section_size, patch_size=conf.image_info.patch_size, split='test')

            train_tokens, train_metadata = load_patches_and_metadata_from_coordinates(conf, model, train_patch_eval_dataset_fixedcoords, imc_dataset.patch_level_tasks)
            test_tokens, test_metadata = load_patches_and_metadata_from_coordinates(conf, model, test_patch_eval_dataset_fixedcoords, imc_dataset.patch_level_tasks)

        else:
            print('Did not find coordinates for patches. Computing new coordinates.')


        
            train_tokens, train_metadata, train_crop_coordinates = sample_patches_and_metadata(conf, model, train_patch_eval_dataset, num_patches_per_crop, num_crops_per_img, imc_dataset.patch_level_tasks)
            test_tokens, test_metadata, test_crop_coordinates = sample_patches_and_metadata(conf, model, test_patch_eval_dataset, num_patches_per_crop, num_crops_per_img, imc_dataset.patch_level_tasks)

            torch.save(train_crop_coordinates, f'dumps/{conf.dataset.name}/patch_coordinates_train.pt')
            torch.save(test_crop_coordinates, f'dumps/{conf.dataset.name}/patch_coordinates_test.pt')

        train_data = {"patch_summary_tokens": train_tokens, "reduced_patch_summary_tokens": train_data["reduced_patch_summary_tokens"], "metadata": train_metadata}
        test_data = {"patch_summary_tokens": test_tokens, "reduced_patch_summary_tokens": test_data["reduced_patch_summary_tokens"], "metadata": test_metadata}

        torch.save(train_data, os.path.join(embedding_path, f"patch_embeddings_train.pt"))
        torch.save(test_data, os.path.join(embedding_path, f"patch_embeddings_test.pt"))
            
    return train_tokens, train_metadata, test_tokens, test_metadata


def sample_patches_and_metadata(conf, model, patch_eval_dataset, samples_per_crop, crop_per_img, tasks, max_num_batches=None):

    set_seed(conf.downstream.seed)

    dataloader = DataLoader(patch_eval_dataset, batch_size=conf.training.batch_size, pin_memory=True, shuffle=False, num_workers=conf.training.num_workers)

    patches = []
    patch_metadata = dict()
    for task in tasks:
        patch_metadata[task.task_name] = []
    patch_metadata["image_name"] = []

    crop_coordinates = []

    for _ in range(crop_per_img):
        for i, batch in enumerate(tqdm(dataloader)):
            if max_num_batches is not None and i >= max_num_batches:
                break
            imgs, channels, metadata = batch
            imgs = imgs.cuda()
            channels = channels.cuda()
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=conf.training.fp16):
                    results = model.embed(imgs, channels, return_dict=True, place_on_cpu=True)
            
            for res in results:
                patch_summary = res["patch_summary_token"] # B x GH x GW x D
                for i in range(patch_summary.shape[0]):
                    layer = patch_summary[i] # GH x GW x D
                    N = layer.shape[0] * layer.shape[1]
                    patch_indices = np.random.choice(N, samples_per_crop)
                    patches += list(layer.flatten(start_dim=0, end_dim=-2)[patch_indices])
                    patch_metadata["image_name"] += [metadata["image_name"][i]] * samples_per_crop
                    for task in tasks:
                        grid = metadata[task.task_name][i]
                        patch_metadata[task.task_name] += grid.flatten(start_dim=0, end_dim=-1)[patch_indices].tolist()
                    crop_coordinates.append(
                        {
                            'image_name' : metadata["image_name"][i],
                            'row' : metadata['row'][i],
                            'col' : metadata['col'][i],
                            'patch_indices' : patch_indices
                        }
                    )
    
    patches = torch.stack(patches, dim=0).float()
    return patches, patch_metadata, crop_coordinates



def add_report_to_dict(report, results_dicts, method, task, run_id):
    for key in report.keys():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            classname = key
            for metric in report[classname].keys():
                results_dicts.append({
                    'method' : method,
                    'task' : task,
                    'metric' : metric,
                    'score' : report[classname][metric],
                    'class' : classname,
                    'run_id' : run_id,
                })

    results_dicts.append({
        'method' : method,
        'task' : task,
        'metric' : 'macro avg f1-score',
        'score' : report['macro avg']['f1-score'],
        'class' : 'global',
        'run_id' : run_id,
    })

    results_dicts.append({
        'method' : method,
        'task' : task,
        'metric' : 'weighted avg f1-score',
        'score' : report['weighted avg']['f1-score'],
        'class' : 'global',
        'run_id' : run_id,
    })
    
    results_dicts.append({
        'method' : method,
        'task' : task,
        'metric' : 'accuracy',
        'score' : report['accuracy'],
        'class' : 'global',
        'run_id' : run_id,
    })
    return 
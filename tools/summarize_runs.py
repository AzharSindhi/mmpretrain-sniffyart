import os
import sys
import numpy as np
import torch
import traceback
from concurrent.futures import ProcessPoolExecutor
from mmengine.config import Config
from mmpretrain.apis import init_model
from mmpretrain.datasets import build_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import multiprocessing as mp
from tqdm import tqdm
import json
import glob
from pathlib import Path
# Import ClsDataPreprocessor directly
from mmpretrain.models.utils.data_preprocessor import ClsDataPreprocessor

# Define model groups
model_groups = {
    "hrnet": ["hrnet_1", "hrnet_2", "hrnet_3", "hrnet_4", "hrnet_5"],
    "rn101": ["rn101_1", "rn101_2", "rn101_3", "rn101_4", "rn101_5"],
    "rn50": ["rn50_1", "rn50_2", "rn50_3", "rn50_4", "rn50_5"],
    "swinv2": ["swinv2_1", "swinv2_2", "swinv2_3", "swinv2_4", "swinv2_5"]
}

DEVICE = "cuda:0"
REDUCED_BATCH_SIZE = 32  # Reduced batch size

def custom_collate(batch):
    inputs = torch.stack([item['inputs'] for item in batch])
    data_samples = [item['data_samples'] for item in batch]  # Assuming data_samples contains complex structures
    return {'inputs': inputs, 'data_samples': data_samples}

def custom_inference_model(model, input_data):
    with torch.no_grad():
        output = model(input_data).detach().cpu()
        pred_label = torch.argmax(output, dim=1).numpy().tolist()
    return pred_label

def evaluate_model(model, dataloader, preprocessor):
    gt_labels = []
    pred_labels = []

    for data in dataloader:
        data = preprocessor(data)
        inputs = data['inputs'].to(DEVICE)
        gts = [d.gt_label.item() for d in data["data_samples"]]
        preds = custom_inference_model(model, inputs)
        gt_labels.extend(gts)
        pred_labels.extend(preds)

    accuracy = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels, average="macro")
    return accuracy, f1

def load_config_and_preprocessor(model_path):
    config_file = [f for f in os.listdir(model_path) if f.endswith('.py')][0]
    config_path = os.path.join(model_path, config_file)
    cfg = Config.fromfile(config_path)
    cfg.test_dataloader.dataset.test_mode = True

    preprocessor_cfg = cfg.get('data_preprocessor')
    preprocessor = ClsDataPreprocessor(**preprocessor_cfg)
    return cfg, preprocessor

def build_dataloader(dataset_cfg):
    dataset = build_dataset(dataset_cfg)
    dataloader = DataLoader(
        dataset, 
        batch_size=REDUCED_BATCH_SIZE, 
        num_workers=5, 
        shuffle=False,
        collate_fn=custom_collate)
    return dataloader

def locate_results_path(model_path, type):
    json_path, log_path = None, None
    # get the directories in the model path
    json_paths = sorted(glob.glob(os.path.join(model_path, '*', '*.json'), recursive=True), key=lambda x: x.split('/')[-2])
    if len(json_paths) > 0:
        json_path = json_paths[-1]
    
    log_paths = sorted(glob.glob(os.path.join(model_path, '*', 'vis_data', '*.json'), recursive=True), key=lambda x: x.split('/')[-2])
    if len(log_paths) > 0:
        log_path = log_paths[-1] # get the latest log file
        
    return json_path, log_path

def read_log_results(log_path):
    val_accuracy, val_f1 = None, None
    with open(log_path, 'r') as f:
        lines = f.readlines()[-20:]
        metrics_line = [line for line in lines if 'Epoch(val)' in line]
        if len(metrics_line) > 0:        
            metrics = metrics_line[0].split("Epoch(val)")[1].split("  ")
            val_accuracy = float(metrics[2].split(": ")[1])
            val_f1 = float(metrics[6].split(": ")[1])
    
    return val_accuracy, val_f1

def read_json_results(json_path):
    with open(json_path, 'r') as f:
        results = json.loads(f.read().strip())
    return results

def read_val_results(json_path):
    with open(json_path, 'r') as f:
        result = json.loads(f.read().strip().splitlines()[-1])
    return result

def run_evaluation(model_dir, group, subdir_path, results):

    model_path = os.path.join(subdir_path, model_dir)
    checkpoint_path = os.path.join(model_path, 'epoch_100.pth')

    # Locate paths
    test_json_path, val_json_path = locate_results_path(model_path, 'test')

    if test_json_path:
        test_results = read_json_results(test_json_path)
        test_accuracy = test_results['accuracy/top1']
        test_f1 = test_results['single-label/f1-score']
        results['test'][group]['accuracies'].append(test_accuracy)
        results['test'][group]['f1_scores'].append(test_f1)
    else:
        return results # ignore
        cfg, preprocessor = load_config_and_preprocessor(model_path)
        test_dataloader = build_dataloader(cfg.test_dataloader.dataset)
        model = init_model(cfg, checkpoint_path, device=DEVICE)
        model.eval()
        test_accuracy, test_f1 = evaluate_model(model, test_dataloader, preprocessor)
        results['test'][group]['accuracies'].append(test_accuracy)
        results['test'][group]['f1_scores'].append(test_f1)

    if val_json_path:
        val_result = read_val_results(val_json_path)
        if val_result["step"] > 20: # lineant
            val_accuracy = val_result['accuracy/top1']
            val_f1 = val_result['single-label/f1-score']
            results['val'][group]['accuracies'].append(val_accuracy)
            results['val'][group]['f1_scores'].append(val_f1)
        else:
            raise Exception(f"should be available, check inside  {val_json_path}")
            cfg, preprocessor = load_config_and_preprocessor(model_path)
            val_dataloader = build_dataloader(cfg.val_dataloader.dataset)
            model = init_model(cfg, checkpoint_path, device=DEVICE)
            model.eval()
            val_accuracy, val_f1 = evaluate_model(model, val_dataloader, preprocessor)
            results['val'][group]['accuracies'].append(val_accuracy)
            results['val'][group]['f1_scores'].append(val_f1)
    else:
        raise Exception(f"should be available, check {val_json_path}")
        cfg, preprocessor = load_config_and_preprocessor(model_path)
        val_dataloader = build_dataloader(cfg.val_dataloader.dataset)
        model = init_model(cfg, checkpoint_path, device=DEVICE)
        model.eval()
        val_accuracy, val_f1 = evaluate_model(model, val_dataloader, preprocessor)
        results['val'][group]['accuracies'].append(val_accuracy)
        results['val'][group]['f1_scores'].append(val_f1)
    
    return results


def process_setting(base_dir, experiment_name, neck, file):
    subdir_path = os.path.join(base_dir, experiment_name, neck) # default or nonlinear
    results = {
        'test': {group: {'accuracies': [], 'f1_scores': []} for group in model_groups},
        'val': {group: {'accuracies': [], 'f1_scores': []} for group in model_groups}
    }
    modelnames = os.listdir(subdir_path)
    for i, model_dir in tqdm(enumerate(modelnames)): #hrnet, rn101, rn50, swinv2, etc
        print(f"Processing {os.path.join(subdir_path, model_dir)} ({i+1}/{len(modelnames)}):")
        group = model_dir.split('_')[0]
        if group in model_groups.keys():
            results = run_evaluation(model_dir, group, subdir_path, results)
    file.write("----------------------------------------------------\n")
    write_results(base_dir, experiment_name, neck, results, file)

def write_results(base_dir, context_type, subdir, results, file):
    file.write(f"directory: {base_dir}/{context_type}/{subdir}\n")
    file.write(f"type: {subdir}\n")
    print(f"directory: {base_dir}/{context_type}/{subdir}")
    print(f"type: {subdir}")

    file.write("Test Results:\n")
    print("Test Results:")
    for group, metrics in results['test'].items():
        mean_accuracy = np.mean(metrics['accuracies'])
        std_accuracy = np.std(metrics['accuracies'])
        mean_f1 = np.mean(metrics['f1_scores'])
        std_f1 = np.std(metrics['f1_scores'])

        result_str = (f"{group} -> mean accuracy: {mean_accuracy:.4f}, std accuracy: {std_accuracy:.4f}, "
                      f"mean f1: {mean_f1:.4f}, std f1: {std_f1:.4f}")
        file.write(result_str + "\n")
        print(result_str)

    file.write("Validation Results:\n")
    print("Validation Results:")
    for group, metrics in results['val'].items():
        mean_accuracy = np.mean(metrics['accuracies'])
        std_accuracy = np.std(metrics['accuracies'])
        mean_f1 = np.mean(metrics['f1_scores'])
        std_f1 = np.std(metrics['f1_scores'])

        result_str = (f"{group} -> mean accuracy: {mean_accuracy:.4f}, std accuracy: {std_accuracy:.4f}, "
                      f"mean f1: {mean_f1:.4f}, std f1: {std_f1:.4f}")
        file.write(result_str + "\n")
        print(result_str)

def main(base_dir):
    with open(os.path.join(base_dir, 'summary_results.txt'), 'w') as file:
        experiment_types = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        context_experiments = [d for d in experiment_types if "with_context" in d or "without_context" in d]
        for experiment_name in context_experiments:
            for neck in ['default']:
                process_setting(base_dir, experiment_name, neck, file)
        if not "with_context_ablations" in experiment_types:
            return
        
        experiment_name = "with_context_ablations"
        for neck in ['default']:
            ablations = os.listdir(os.path.join(base_dir, experiment_name, neck))
            for ablation in ablations:
                thisneck = os.path.join(neck, ablation)
                process_setting(base_dir, experiment_name, thisneck, file)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    if len(sys.argv) != 2:
        print("Usage: python summarize_results.py <base_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]
    main(base_dir)

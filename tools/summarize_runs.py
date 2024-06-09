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

# Import ClsDataPreprocessor directly
sys.path.append('/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart')
from mmpretrain.models.utils.data_preprocessor import ClsDataPreprocessor

# Define model groups
model_groups = {
    "hrnet": ["hrnet_1", "hrnet_2", "hrnet_3", "hrnet_4", "hrnet_5"],
    "rn101": ["rn101_1", "rn101_2", "rn101_3", "rn101_4", "rn101_5"],
    "rn50": ["rn50_1", "rn50_2", "rn50_3", "rn50_4", "rn50_5"],
    "swinv2": ["swinv2_1", "swinv2_2", "swinv2_3", "swinv2_4", "swinv2_5"]
}

DEVICE = "cuda:0"
REDUCED_BATCH_SIZE = 2  # Reduced batch size

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
    f1 = f1_score(gt_labels, pred_labels, average='weighted')
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

def run_evaluation(model_dir, group, models, subdir_path, results):
    try:
        if model_dir in models:
            model_path = os.path.join(subdir_path, model_dir)
            checkpoint_path = os.path.join(model_path, 'epoch_100.pth')

            cfg, preprocessor = load_config_and_preprocessor(model_path)

            test_dataloader = build_dataloader(cfg.test_dataloader.dataset)
            val_dataloader = build_dataloader(cfg.val_dataloader.dataset)

            model = init_model(cfg, checkpoint_path, device=DEVICE)
            model.eval()

            test_accuracy, test_f1 = evaluate_model(model, test_dataloader, preprocessor)
            results['test'][group]['accuracies'].append(test_accuracy)
            results['test'][group]['f1_scores'].append(test_f1)

            val_accuracy, val_f1 = evaluate_model(model, val_dataloader, preprocessor)
            results['val'][group]['accuracies'].append(val_accuracy)
            results['val'][group]['f1_scores'].append(val_f1)
            
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}
    return results

def process_setting(base_dir, context_type, subdir, file):
    subdir_path = os.path.join(base_dir, context_type, subdir)
    results = {
        'test': {group: {'accuracies': [], 'f1_scores': []} for group in model_groups},
        'val': {group: {'accuracies': [], 'f1_scores': []} for group in model_groups}
    }

    errors = []

    with ProcessPoolExecutor(mp_context=mp.get_context('spawn')) as executor:
        futures = []
        for model_dir in os.listdir(subdir_path):
            for group, models in model_groups.items():
                futures.append(executor.submit(run_evaluation, model_dir, group, models, subdir_path, results))

        for future in futures:
            result = future.result()
            if 'error' in result:
                errors.append(result)

    if errors:
        print("Errors occurred during processing:")
        for error in errors:
            print(f"Error: {error['error']}")
            print(f"Traceback: {error['traceback']}")

    write_results(base_dir, context_type, subdir, results, file)

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
        for context_type in ['with_context', 'without_context']:
            for subdir in ['default', 'nonlinear']:
                process_setting(base_dir, context_type, subdir, file)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    if len(sys.argv) != 2:
        print("Usage: python summarize_results.py <base_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]
    main(base_dir)

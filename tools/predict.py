import os
import json
import shutil
from mmcls.apis import init_model, inference_model
from mmcv import Config
from mmcls.datasets import build_dataloader, build_dataset

# Initialize the model
config_file = 'path/to/your/config.py'
checkpoint_file = 'path/to/your/checkpoint.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Load configuration
cfg = Config.fromfile(config_file)

# Modify the config to set the test mode
cfg.data.test.test_mode = True

# Build the test dataset and dataloader
dataset = build_dataset(cfg.data.test)
dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

# Output directory
output_dir = 'path/to/save/classified_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to save images
def save_image(img_path, gt_label, pred_label, output_dir):
    if pred_label == gt_label:
        # Correctly classified, store in GT label directory
        correct_label_dir = os.path.join(output_dir, str(gt_label))
        if not os.path.exists(correct_label_dir):
            os.makedirs(correct_label_dir)
        shutil.copy(img_path, os.path.join(correct_label_dir, os.path.basename(img_path)))
    else:
        # Misclassified, store in GT -> Pred directory
        misclassified_dir = os.path.join(output_dir, f'{gt_label}->{pred_label}')
        if not os.path.exists(misclassified_dir):
            os.makedirs(misclassified_dir)
        shutil.copy(img_path, os.path.join(misclassified_dir, os.path.basename(img_path)))

# Iterate through the test dataloader
for data in dataloader:
    img_path = data['img_metas'][0].data[0][0]['filename']
    gt_label = data['gt_labels'].data[0].item()
    result = inference_model(model, img_path)
    pred_label = result['pred_label']

    save_image(img_path, gt_label, pred_label, output_dir)

print("Classified images have been saved.")

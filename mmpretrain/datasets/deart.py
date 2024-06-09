from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any
from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from PIL import Image 
import numpy as np
from pycocotools.coco import COCO
import os
import torch

@DATASETS.register_module()
class DeArt(BaseDataset):
    def __init__(self,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 with_label=True,
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 use_context = True,
                 random_context = False,
                 random_context_prob = 0.0,
                 mask_context_box = False,
                 metainfo: Optional[dict] = None,
                 lazy_init: bool = False,
                 **kwargs):
        assert (ann_file or data_prefix or data_root), \
            'One of `ann_file`, `data_root` and `data_prefix` must '\
            'be specified.'

        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.with_label = with_label
        classes = ["partial", "ride", "bend", "sit", "fall", "stand", "other", "walk", "pray", "kneel", "lie", "squats", "push"]
        self.class_mapping = self.get_class_mapping(classes)
        self.use_context = use_context
        self.random_context = random_context
        self.mask_context_box = mask_context_box
        self.random_context_prob = random_context_prob
        super().__init__(
            # The base class requires string ann_file but this class doesn't
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            # Force to lazy_init for some modification before loading data.
            lazy_init=True,
            **kwargs)

        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    def get_class_mapping(self, class_names):
        class_mapping = {}
        for idx, c in enumerate(class_names):
            class_mapping[c] = idx
        return class_mapping

    # def save_processed_image(
    #     self,
    #     image,
    #     filename,
    #     outdir="./outputs",
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # ):
    #     mean = np.array(mean)
    #     std = np.array(std)

    #     os.makedirs(outdir, exist_ok=True)
    #     image = image.detach().cpu().numpy().transpose(1, 2, 0)
    #     image = image.astype(np.uint8)

    #     print("image shape:", image.shape, image.min(), image.max())
    #     image_path_context = os.path.join(outdir, filename)
    #     cv2.imwrite(image_path_context, image)

    def read_image(self, path):
        return np.array(Image.open(path).convert("RGB"))

    def crop_image(self, img, box):
        x, y, w, h = np.asarray(box, dtype=int)
        return np.copy(img[y: y + h, x: x + w])

    def mask_context_image(self, image, bbox):

        if self.mask_context_box:
            x, y, w, h = bbox
            image[y : y + h, x : x + w] = (0, 0, 0)

        return image
    
    def load_data_list(self):
        assert isinstance(self.ann_file, str)

        coco = COCO(self.ann_file)
        ann_ids = coco.getAnnIds()
        img_ids = coco.getImgIds()
        annotations = coco.loadAnns(ids=ann_ids)
        data_list = []
        for annotation in annotations:
            img_id = annotation["image_id"]
            gt_label = int(annotation["category_id"])
            x, y, w, h = annotation["bbox"]
            
            img_id_random = np.random.choice(img_ids)
            image_dict = coco.loadImgs(ids=[img_id_random])[0]
            image_path_random = os.path.join(self.data_root, image_dict["file_name"])
        
            image_dict = coco.loadImgs(ids=[img_id])[0]
            image_path = os.path.join(self.data_root, image_dict["file_name"])
            info = {'img_path': image_path, 'img_path_random': image_path_random, 'bbox': [x,y,w,h], 'gt_label': int(gt_label)}
            data_list.append(info)
        
        return data_list
        
    def apply_transform(self, img, data_info):
        """
        Apply transformations to the image
        """
        data_info = data_info.copy()
        data_info["img"] = img
        data_info['img_shape'] = img.shape[:2]
        data_info['ori_shape'] = img.shape[:2]
        return self.pipeline(data_info)
    
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        # print(data_info)
        context_image = self.read_image(data_info["img_path"])
        box_image = self.crop_image(context_image, data_info["bbox"])
        out = self.apply_transform(box_image, data_info)
        
        if self.use_context:
            if self.mask_context_box:
                context_image = self.mask_context_image(context_image, data_info["bbox"])
            elif self.random_context and np.random.rand() < self.random_context_prob:
                context_image = self.read_image(data_info["img_path_random"])
            
            context_out = self.apply_transform(context_image, data_info)
            box_tensor = out["inputs"]
            context_tensor = context_out["inputs"]
            out["inputs"] = torch.cat((box_tensor, context_tensor), dim=0)
        
        return out
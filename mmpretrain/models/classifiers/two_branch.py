# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseClassifier
import copy


@MODELS.register_module()
class TwoBranchModel(BaseClassifier):
    """Two-branch image classifier for supervised classification task.

    Args:
        backbone1 (dict): The first backbone module.
        backbone2 (dict): The second backbone module.
        neck (dict, optional): The neck module to process features from
            the backbone. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path.
            Defaults to None.
        train_cfg (dict, optional): The training setting. Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. Defaults to None.
        init_cfg (dict, optional): The config to control initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 use_context: Optional[bool] = False,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'ClsDataPreprocessor')
            data_preprocessor.setdefault('batch_augments', train_cfg)
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        super(TwoBranchModel, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone_crop = backbone
        self.neck = neck
        self.head = head

        # If the model needs to load pretrain weights from a third party,
        # the key can be modified with this hook
        if hasattr(self.backbone_crop, '_checkpoint_filter'):
            self._register_load_state_dict_pre_hook(
                self.backbone_crop._checkpoint_filter)
        
        if use_context:
            self.backbone_context = copy.deepcopy(self.backbone_crop)
        else:
            self.backbone_context = None
        
        self.use_context = use_context
        self.transform = None
        if use_context:
            # size doubled due to context concatenation, have to transform bakc
            self.transform = nn.Linear(head.in_channels *2, head.in_channels) 
    

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor(s) without any
          post-processing, same as a common PyTorch Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def unpack_inputs(self, inputs: torch.Tensor):
        """
        Unpacks a tensor of shape (B, 3 or 6, H, W) into two tensors of shape (B, 3, H, W)
        and ensures they are deep copies.

        Args:
            inputs (torch.Tensor): The input tensor of shape (B, 3 or 6, H, W).

        Returns:
            crop_tensor (torch.Tensor): The first output tensor of shape (B, 3, H, W).
            box_tensor (torch.Tensor): The second output tensor of shape (B, 3, H, W) or None.
        """
        channels = inputs.size(1)
        if channels == 6:
            assert self.backbone_context is not None, "Input contains context, but the context backbone not provided"
            crop_tensor = inputs[:, :3, :, :].clone()
            context_tensor = inputs[:, 3:, :, :].clone()
            return crop_tensor, context_tensor

        elif channels == 3:
            assert self.backbone_context is None, "Input does not contain context information, but the context backbone has been provided"
            return inputs, None
        else:
            raise ValueError(f"Expected input with 3 or 6 channels, but got {channels} channels.")
        
    
    def append_context_out(self, xcontext:torch.Tensor, x: tuple):
        if xcontext is None:
            return x
        out_context = self.backbone_context(xcontext)
        # out = tuple([torch.cat((x1, x2), dim=1) for x1, x2 in zip(x, out_context)])
        return x + out_context

        
    def extract_feat(self, inputs, stage='neck'):
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from:

                - "backbone": The output of backbone network. Returns a tuple
                  including multiple stages features.
                - "neck": The output of neck module. Returns a tuple including
                  multiple stages features.
                - "pre_logits": The feature before the final classification
                  linear layer. Usually returns a tensor.

                Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the
            output of backbone and neck is a tuple and the output of
            pre_logits is a tensor.

        Examples:
            1. Backbone output

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64, 56, 56])
            torch.Size([1, 128, 28, 28])
            torch.Size([1, 256, 14, 14])
            torch.Size([1, 512, 7, 7])

            2. Neck output

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>>
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64])
            torch.Size([1, 128])
            torch.Size([1, 256])
            torch.Size([1, 512])

            3. Pre-logits output (without the final linear classifier head)

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/vision_transformer/vit-base-p16_pt-64xb64_in1k-224.py').model
            >>> model = build_classifier(cfg)
            >>>
            >>> out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
            >>> print(out.shape)  # The hidden dims in head is 3072
            torch.Size([1, 3072])
        """  # noqa: E501
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')
        
        xcrop, xcontext = self.unpack_inputs(inputs)
        x = self.backbone_crop(xcrop)
        x = self.append_context_out(xcontext, x)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
            x = torch.cat(x, dim=1)
            if self.transform:
                x = (self.transform(x),)
        if stage == 'neck':
            return x

        assert self.with_head and hasattr(self.head, 'pre_logits'), \
            "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples, **kwargs)

    def get_layer_depth(self, param_name: str):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            Tuple[int, int]: The layer-wise depth and the max depth.
        """
        if hasattr(self.backbone, 'get_layer_depth'):
            return self.backbone.get_layer_depth(param_name, 'backbone.')
        else:
            raise NotImplementedError(
                f"The backbone {type(self.backbone)} doesn't "
                'support `get_layer_depth` by now.')
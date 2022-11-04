# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LeafDataset(CustomDataset):
    """Leaf dataset.

    In segmentation map annotation for HRF, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    CLASSES = ('background', 'leaf')

    PALETTE = [[0, 0, 0], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(LeafDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
        assert osp.exists(self.img_dir)

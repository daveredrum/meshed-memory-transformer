# from .field import RawField, Merge, ImageDetectionsField, TextField
from .field import RawField, Merge, ImageDetectionsField, TextField, ScanNetDetectionsField
# from .dataset import COCO
from .dataset import ScanNet
from torch.utils.data import DataLoader as TorchDataLoader

class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)

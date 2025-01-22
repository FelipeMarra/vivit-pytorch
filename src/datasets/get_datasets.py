from enum import Enum
from typing import Tuple
from torch.utils.data import Dataset
from torchvision.transforms import v2
from datasets.kinetics import KineticsDataset
from datasets.games import GamesDataset

class DatasetsEnum(Enum):
    KINETICS400 = 1
    GAMES = 2

def get_datasets(
        option:DatasetsEnum, 
        root:str, 
        n_frames:int, 
        train_transforms:v2.Compose, 
        test_transforms:v2.Compose,
        stride:int = 2,
        transpose:bool=True) -> Tuple[Dataset, Dataset, Dataset]:
    """
        Args:
            options: datset option from DatasetsEnum
            root: root path for the dataset
            n_frames: n_frames to retreive
            train_transforms: train transforms Compose
            test_transforms: eval and test transforms Compose

        Returns:
            train, eval and test datasets if available and None otherwise
    """
    if option == DatasetsEnum.KINETICS400:
        train_dataset = KineticsDataset(root, 'train', n_frames, train_transforms, stride=stride, transpose=transpose)
        eval_dataset = KineticsDataset(root, 'val', n_frames, test_transforms, stride=stride, transpose=transpose)
        test_dataset = KineticsDataset(root, 'test', n_frames, test_transforms, stride=stride, transpose=transpose)

        return train_dataset, eval_dataset, test_dataset

    if option == DatasetsEnum.GAMES:
        train_dataset = GamesDataset(root, 'train', n_frames, train_transforms, stride=stride, transpose=transpose)

        #TODO there is no split yet, so return the same thing for now
        eval_dataset = GamesDataset(root, 'val', n_frames, train_transforms, stride=stride, transpose=transpose)
        test_dataset = GamesDataset(root, 'test', n_frames, train_transforms, stride=stride, transpose=transpose)

        return train_dataset, eval_dataset, test_dataset

from .ucf101 import UCF101  # type: ignore
from .sun397 import SUN397  # type: ignore
from .eurosat import EuroSAT  # type: ignore
from .food101 import Food101  # type: ignore
from .imagenet import ImageNet  # type: ignore
from .sipakmed import SipakMed  # type: ignore
from .fgvc import FGVCAircraft  # type: ignore
from .hicervix import HiCervix  # type: ignore
from .caltech101 import Caltech101  # type: ignore
from .oxford_pets import OxfordPets  # type: ignore
from .dtd import DescribableTextures  # type: ignore
from .stanford_cars import StanfordCars  # type: ignore
from .oxford_flowers import OxfordFlowers  # type: ignore
from .dataset_kaggle import Dataset_kaggle1, Dataset_kaggle2  # type: ignore
from .cu_us import USDataset_C1, USDataset_C2, USDataset_C3  # type: ignore


dataset_list = {
    "oxford_pets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvc": FGVCAircraft,
    "food101": Food101,
    "oxford_flowers": OxfordFlowers,
    "stanford_cars": StanfordCars,
    "imagenet": ImageNet,
    "sipakmed": SipakMed,
    "kaggle1": Dataset_kaggle1,
    "kaggle2": Dataset_kaggle2,
    "hicervix": HiCervix,
    "cu_us1": USDataset_C1,
    "cu_us2": USDataset_C2,
    "cu_us3": USDataset_C3,
}


def build_dataset(
    dataset, root_path, shots, level="level_1", pourcentage=0.0, preprocess=None
):
    if dataset == "imagenet":
        return dataset_list[dataset](root_path, shots, preprocess)
    elif dataset == "hicervix":
        return dataset_list[dataset](root_path, shots, level, pourcentage)
    else:
        return dataset_list[dataset](root_path, shots)

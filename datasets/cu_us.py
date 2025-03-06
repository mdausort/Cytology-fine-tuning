import os
import glob
import pandas as pd
from PIL import Image
from .utils import Datum, DatasetBase  # type: ignore


class USDataset_C1(DatasetBase):

    dataset_dir = "Dataset/US_CU/"
    classes = ["C11", "C12"]

    def __init__(self, root, num_shots, preprocess=None):
        self.root = root
        self.dataset_dir = os.path.join(root, self.dataset_dir + "split_data/")
        self.image_dir = os.path.join(self.dataset_dir, "step3_c1")

        train = self.create_list_of_datum("train")
        val = self.create_list_of_datum("val")
        test = self.create_list_of_datum("test")

        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    def __getitem__(self, im_files, idx):

        image = Image.open(im_files[idx]).convert("RGB")
        image_name = im_files[idx].split("/")[-1].split(".")[0]

        df = pd.read_excel(os.path.join(self.root, "Dataset/US_CU/Labels.xlsx"))
        self.labels_dict = dict(zip(df["Cyto"], df["C1"]))

        class_name = self.labels_dict[image_name]
        class_ = self.classes.index(class_name)
        return image, class_, class_name, im_files[idx]

    def create_list_of_datum(self, set):
        """
        Create a list of Datum objects, each containing the image and label.
        """

        datum_list = []

        im_files = glob.glob(os.path.join(self.image_dir, set, "*.jpg"))
        for i in range(len(im_files)):
            # Get the image and the class
            image, class_, class_name, impath = self.__getitem__(im_files, i)

            # Create a Datum object
            datum = Datum(impath=impath, label=class_, classname=class_name)

            # Append the datum to the list
            datum_list.append(datum)

        return datum_list


class USDataset_C2(DatasetBase):

    dataset_dir = "Dataset/US_CU/"
    classes = ["C21", "C22", "C23"]

    def __init__(self, root, num_shots, preprocess=None):
        self.root = root
        self.dataset_dir = os.path.join(root, self.dataset_dir + "split_data/")
        self.image_dir = os.path.join(self.dataset_dir, "step3_c2")

        train = self.create_list_of_datum("train")
        val = self.create_list_of_datum("val")
        test = self.create_list_of_datum("test")

        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    def __getitem__(self, im_files, idx):

        image = Image.open(im_files[idx]).convert("RGB")
        image_name = im_files[idx].split("/")[-1].split(".")[0]

        df = pd.read_excel(os.path.join(self.root, "Dataset/US_CU/Labels.xlsx"))
        self.labels_dict = dict(zip(df["Cyto"], df["C2"]))

        class_name = self.labels_dict[image_name]

        class_ = self.classes.index(class_name)

        return image, class_, class_name, im_files[idx]

    def create_list_of_datum(self, set):
        """
        Create a list of Datum objects, each containing the image and label.
        """

        datum_list = []

        im_files = glob.glob(os.path.join(self.image_dir, set, "*.jpg"))
        for i in range(len(im_files)):
            # Get the image and the class
            image, class_, class_name, impath = self.__getitem__(im_files, i)

            # Create a Datum object
            datum = Datum(impath=impath, label=class_, classname=class_name)

            # Append the datum to the list
            datum_list.append(datum)

        return datum_list


class USDataset_C3(DatasetBase):

    dataset_dir = "Dataset/US_CU/"
    classes = ["C31", "C32", "C33", "C34", "C35"]

    def __init__(self, root, num_shots, preprocess=None):
        self.root = root
        self.dataset_dir = os.path.join(root, self.dataset_dir + "split_data/")
        self.image_dir = os.path.join(self.dataset_dir, "step3_c3")

        train = self.create_list_of_datum("train")
        val = self.create_list_of_datum("val")
        test = self.create_list_of_datum("test")

        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

    def __getitem__(self, im_files, idx):

        image = Image.open(im_files[idx]).convert("RGB")
        image_name = im_files[idx].split("/")[-1].split(".")[0]

        df = pd.read_excel(os.path.join(self.root, "Dataset/US_CU/Labels.xlsx"))
        self.labels_dict = dict(zip(df["Cyto"], df["C3"]))

        class_name = self.labels_dict[image_name]

        class_ = self.classes.index(class_name)

        return image, class_, class_name, im_files[idx]

    def create_list_of_datum(self, set):
        """
        Create a list of Datum objects, each containing the image and label.
        """

        datum_list = []

        im_files = glob.glob(os.path.join(self.image_dir, set, "*.jpg"))
        for i in range(len(im_files)):
            # Get the image and the class
            image, class_, class_name, impath = self.__getitem__(im_files, i)

            # Create a Datum object
            datum = Datum(impath=impath, label=class_, classname=class_name)

            # Append the datum to the list
            datum_list.append(datum)

        return datum_list

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import Datum, DatasetBase

template = ["A pap smear slide showing a {} cervical cells."]


class HiCervix(DatasetBase):

    dataset_dir = "HiCervix"

    def __init__(self, root, num_shots, level, pourcentage):
        self.dataset_dir = os.path.join(root)
        self.image_dir = os.path.join(self.dataset_dir)

        self.level = level
        self.pourcentage = pourcentage

        self.train_csv = os.path.join(self.image_dir, "train.csv")
        self.test_csv = os.path.join(self.image_dir, "test.csv")
        self.val_csv = os.path.join(self.image_dir, "val.csv")

        self.template = template

        train = self.create_list_of_datum("train")
        val = self.create_list_of_datum("val")
        test = self.create_list_of_datum("test")

        n_shots_val = min(num_shots, 4)

        # To take the same percentage of each class (experience 3)
        if self.pourcentage > 0:
            print('Percentage of the dataset considered :', self.pourcentage)
            train = self.generate_pourcent_dataset(train, pourcentage=self.pourcentage)
            val = self.generate_pourcent_dataset(val, pourcentage=self.pourcentage)
        else:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)

        super().__init__(train_x=train, val=val, test=test)

    def __getitem__(self, im_files, idx, set):

        name = im_files[idx].split("/")[-1]

        if set == "train":
            df = pd.read_csv(self.train_csv)
        elif set == "test":
            df = pd.read_csv(self.test_csv)
        elif set == "val":
            df = pd.read_csv(self.val_csv)

        interm = df[df['image_name'] == name]
        class_name = (interm.loc[:, self.level].values)[0]

        # Creation of a list of cases considered according to classification level.
        if self.level == "level_1":
            class_name = (interm.loc[:, "level_1"].values)[0]
            class_list = sorted(np.unique(df.loc[:, "level_1"].dropna().tolist()))

            class_ = class_list.index(class_name)

        elif self.level == "level_2":
            class_name = (interm.loc[:, "level_2"].values)[0]
            class_list = sorted(np.unique(df.loc[:, "level_2"].dropna().tolist()))

            if pd.isna(class_name):
                class_ = -1
            else:
                class_ = class_list.index(class_name)

        elif self.level == "level_3":
            class_name_3 = (interm.loc[:, self.level].values)[0]
            class_list_2 = sorted(np.unique(df.loc[:, "level_2"].dropna().tolist()))
            class_list_3 = sorted(np.unique(df.loc[:, "level_3"].dropna().tolist()))

            combined_class_list = np.append(class_list_2, class_list_3)

            cleaned_class_list = pd.Series(combined_class_list).dropna()

            class_list = sorted(np.unique(cleaned_class_list).tolist())

            if pd.isna(class_name_3):
                class_name = (interm.loc[:, "level_2"].values)[0]
            else:
                class_name = class_name_3

            if pd.isna(class_name):
                class_ = -1
            else:
                class_ = class_list.index(class_name)

        return class_, class_name, im_files[idx]

    def create_list_of_datum(self, set):
        """Create a list of Datum objects, each containing the image and label."""
        datum_list = []

        im_files = glob.glob(os.path.join(self.image_dir, set, "*.jpg"))

        for i in tqdm(range(len(im_files))):

            class_, class_name, impath = self.__getitem__(im_files, i, set)
            if class_name is None or class_name == "" or class_ == -1:
                continue

            print(class_, class_name, impath)

            # Create a Datum object
            datum = Datum(impath=impath, label=class_, classname=class_name)

            # Append the datum to the list
            datum_list.append(datum)

        return datum_list

import os
import torch
import random
import numpy as np
from tqdm import tqdm
from utils import get_function
from torch.utils.data import Dataset
from transformers.modeling_outputs import ImageClassifierOutput


class FeaturesDataset(Dataset):
    def __init__(self, features_path):
        self.data = np.load(features_path)
        self.features = self.data["features"]
        self.labels = self.data["labels"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return feature, label


def generate_few_shot(args, data_loader, val=False):

    shot = args.shots

    path = os.path.join(
        args.root_path,
        args.dataset
        + "_"
        + str(args.seed)
        + "_"
        + str(shot)
        + "_"
        + args.model_name
        + "_features_train.npz",
    )  # +args.shots
    if val:
        shot = 4
        path = os.path.join(
            args.root_path,
            args.dataset
            + "_"
            + str(args.seed)
            + "_"
            + str(shot)
            + "_"
            + args.model_name
            + "_features_val.npz",
        )

    dico_all = []

    for i in range(args.num_classes):
        dico = []
        for image, label in data_loader:
            for img, lab in zip(image, label):
                if lab.numpy() == i:
                    dico.append(img)

        dico_all.append(dico)

    features = []
    labels = []

    for classes, type_classe in enumerate(dico_all):
        random.shuffle(type_classe)

        new_list = type_classe[:shot]

        features.extend(new_list)
        labels.extend(np.ones(shot) * classes)

    features = np.array(features)
    labels = np.array(labels)
    np.savez(path, features=features, labels=labels)
    return


def features_extractor(args, model, train_loader, val_loader, test_loader):

    if args.model_name == "vit_google":
        setattr(model, "classifier", torch.nn.Identity())

    features_csv_train = os.path.join(
        args.root_path, args.dataset + "_" + args.model_name + "_features_train.npz"
    )
    features_csv_val = os.path.join(
        args.root_path, args.dataset + "_" + args.model_name + "_features_val.npz"
    )
    features_csv_test = os.path.join(
        args.root_path, args.dataset + "_" + args.model_name + "_features_test.npz"
    )

    list_dataloader = []
    features_path = []
    if not os.path.exists(features_csv_train):
        features_path.append(features_csv_train)
        list_dataloader.append(train_loader)
    if not os.path.exists(features_csv_train):
        features_path.append(features_csv_val)
        list_dataloader.append(val_loader)
    if not os.path.exists(features_csv_train):
        features_path.append(features_csv_test)
        list_dataloader.append(test_loader)
    if len(features_path) == 0:
        print(
            f"All features have been extracted for {args.model_name} and {args.dataset}"
        )
    else:
        encode_image, _, __ = get_function(args.model_name, model)
        with torch.no_grad():
            for dataloader, path in zip(list_dataloader, features_path):

                features = []
                labels = []

                for image, label in tqdm(dataloader):
                    img = encode_image(image.cuda())
                    if isinstance(img, ImageClassifierOutput):
                        img = img.logits
                    features.append(img.cpu().numpy())
                    labels.append(label)

                features = np.concatenate(features)
                labels = np.concatenate(labels)
                np.savez(path, features=features, labels=labels)

    return


def textual_extractor(args, dataset, model, tokenizer):

    textual_csv = os.path.join(
        args.root_path, args.dataset + "_" + args.model_name + "_textual_train.npz"
    )

    if os.path.exists(textual_csv):
        print(
            f"All textual features have been extracted for {args.model_name} and {args.dataset}"
        )
    else:

        if args.dataset in ["sipakmed", "hicervix"]:
            template = "A cytological slide showing a {} cell"
        else:
            template = "A cytological slide showing {} cells"

        texts = [
            template.format(classname.replace("_", " "))
            for classname in dataset.classnames
        ]
        _, text, token = get_function(args.model_name, model, tokenizer)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = token(texts).cuda()
                class_embeddings = text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

        np.savez(textual_csv, textuals=text_features)

    return

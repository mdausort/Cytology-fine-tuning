import os
import torch
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


def features_extractor(args, model, train_loader, val_loader, test_loader):

    device = torch.device("cuda")
    model.to(device)

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
    if not os.path.exists(features_csv_val):
        features_path.append(features_csv_val)
        list_dataloader.append(val_loader)
    if not os.path.exists(features_csv_test):
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

                    image = image.to(device)

                    img = encode_image(image)

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

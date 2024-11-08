import os
import timm
import clip
import torch
import open_clip  # https://github.com/wisdomikezogwo/quilt1m
from datasets import build_dataset
import torchvision.transforms as transforms
from datasets.utils import build_data_loader
from lora import run_uni
from run_utils import set_random_seed, get_arguments
from transformers import AutoModelForImageClassification, AutoImageProcessor
from features import (
    features_extractor,
    FeaturesDataset,
    textual_extractor,
)


def main():

    args = get_arguments()

    set_random_seed(args.seed)

    # Models
    _, preprocess = clip.load(args.backbone)
    tokenizer = None

    if args.model_name == "clip":
        model_clip, preprocess = clip.load(args.backbone)

    elif args.model_name == "quilt":
        model_clip, preprocess, _ = open_clip.create_model_and_transforms(
            "hf-hub:wisdomik/QuiltNet-B-32"
        )

    elif args.model_name == "uni":
        model_clip = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )

    elif args.model_name == "vit_google":
        _ = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model_clip = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )

    elif args.model_name == "biomedclip":
        model_clip, preprocess, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

    else:
        raise RuntimeError(
            "Wrong model name used. Try clip, uni, biomedclip, vit_google or quilt."
        )

    model_clip.eval()
    model_clip.cuda()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")

    features_csv_train = os.path.join(
        args.root_path, args.dataset + "_" + args.model_name + "_features_train.npz"
    )
    features_csv_val = os.path.join(
        args.root_path, args.dataset + "_" + args.model_name + "_features_val.npz"
    )
    features_csv_test = os.path.join(
        args.root_path, args.dataset + "_" + args.model_name + "_features_test.npz"
    )

    if (
        not os.path.exists(features_csv_train)
        or not os.path.exists(features_csv_val)
        or not os.path.exists(features_csv_test)
    ):

        dataset = build_dataset(
            args.dataset, args.root_path, -1, args.level, preprocess
        )

        val_loader = build_data_loader(
            data_source=dataset.val,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        train_loader = None
        if not args.eval_only:
            train_tranform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size=224,
                        scale=(0.08, 1),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

            train_loader = build_data_loader(
                data_source=dataset.train_x,
                batch_size=args.batch_size,
                tfm=train_tranform,
                is_train=True,
                shuffle=True,
                num_workers=5,
            )

        features_extractor(args, model_clip, train_loader, val_loader, test_loader)

    textual_csv_train = os.path.join(
        args.root_path, args.dataset + "_" + args.model_name + "_textual_train.npz"
    )

    if not os.path.exists(textual_csv_train) and args.textual == "True":
        dataset = build_dataset(
            args.dataset, args.root_path, -1, args.level, preprocess
        )

        textual_extractor(args, dataset, model_clip, tokenizer)

    train_dataset = FeaturesDataset(
        os.path.join(
            args.root_path, args.dataset + "_" + args.model_name + "_features_train.npz"
        )
    )
    val_dataset = FeaturesDataset(
        os.path.join(
            args.root_path, args.dataset + "_" + args.model_name + "_features_val.npz"
        )
    )
    test_dataset = FeaturesDataset(
        os.path.join(
            args.root_path, args.dataset + "_" + args.model_name + "_features_test.npz"
        )
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        pin_memory=True,
    )

    # Commente or uncommente the line needed

    # Classifier experience
    run_uni(args, model_clip, logit_scale, train_loader, val_loader, test_loader)
    #


if __name__ == "__main__":
    main()

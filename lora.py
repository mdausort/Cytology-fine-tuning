import os
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import cls_acc
import pytorch_warmup as warmup
import torch.nn.functional as F
from transformers.modeling_outputs import ImageClassifierOutput
from loralib.utils import (
    get_lora_parameters
)


class ClipWrapper(nn.Module):
    def __init__(self, clip_model, model_linear):
        super(ClipWrapper, self).__init__()
        self.clip_model = clip_model
        self.model_linear = model_linear

    def forward(self, x):
        # clip_model returns a tuple, we unpack it
        clip_output = self.clip_model(x)
        # Select only the first element of the tuple
        image_features = clip_output[0]
        # Apply the linear layer on the image features
        output = self.model_linear(image_features)
        return output


def get_number_trainable_parameters(model_name, clip_model):
    n_param = np.sum([p.numel() for p in get_lora_parameters(clip_model)])
    return n_param


def get_feature_size(model, input_size):
    model.eval()

    # Move model to GPU if available
    device = next(model.parameters()).device

    with torch.no_grad():
        # Create a sample input tensor and move it to the same device as the model
        sample_input = torch.randn(1, *input_size).to(device)
        features = model(sample_input)

        if isinstance(features, ImageClassifierOutput):
            features = features.logits

        return np.prod(features.size()[1:])


def evaluate_uni(args, clip_model, loader):

    clip_model.eval()

    acc = 0.0
    loss_epoch = 0.0
    tot_samples = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model(images)

            if isinstance(image_features, ImageClassifierOutput):
                image_features = image_features.logits

            loss = F.cross_entropy(image_features, target)
            loss_epoch += loss.item() * target.shape[0]
            acc += cls_acc(image_features, target) * target.shape[0]
            tot_samples += target.shape[0]

    acc /= tot_samples
    loss_epoch /= tot_samples
    return acc, loss_epoch


def run_uni(args, clip_model, logit_scale, train_loader, val_loader, test_loader):
    """Classifier experiment - backbone freezed and classification layer added on the top of it"""

    VALIDATION = True

    if args.model_name in ["vit_google"]:
        num_features = 768
    elif args.model_name in ["clip", "quilt", "biomedclip", "conch"]:
        num_features = 512
    elif args.model_name in ["uni"]:
        num_features = get_feature_size(clip_model, (3, 224, 224))

    clip_model_ = nn.Sequential(
        nn.Flatten(start_dim=1), nn.Linear(num_features, args.num_classes)
    ).cuda()

    if args.textual == "True":

        textual_features = np.load(
            os.path.join(
                args.root_path,
                args.dataset + "_" + args.model_name + "_textual_train.npz",
            )
        )["textuals"]
        clip_model_[1].weight.data = torch.tensor(textual_features, dtype=torch.float32)

    clip_model_ = clip_model_.cuda()
    trainable_parameters_ = []
    for _, param in clip_model_.named_parameters():
        trainable_parameters_.append(param)

    optimizer = torch.optim.AdamW(
        trainable_parameters_,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=args.lr,
    )

    num_steps = args.n_iters
    warmup_period = 50
    total_iters = warmup_period + num_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_steps, eta_min=1e-6
    )
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0

    while count_iters < total_iters:
        clip_model_.train()

        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.0

        for i, (images, target) in enumerate(tqdm(train_loader)):

            images, target = images.cuda(), target.cuda()
            output = clip_model_(images)
            if isinstance(output, ImageClassifierOutput):
                output = output.logits

            loss = F.cross_entropy(output, target)
            acc_train += cls_acc(output, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    scheduler.step()

        count_iters += 1

        if count_iters == total_iters:
            break

        acc_train /= tot_samples
        loss_epoch /= tot_samples

        current_lr = scheduler.get_last_lr()[0]
        for param_group in optimizer.param_groups:
            optimizer_lr = param_group["lr"]
        print(
            " OptLR: {:.6f}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}".format(
                optimizer_lr, current_lr, acc_train, loss_epoch
            )
        )

        # Eval
        if VALIDATION:
            acc_val, loss_val = evaluate_uni(args, clip_model_, val_loader)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))

    acc_test, _ = evaluate_uni(args, clip_model_, test_loader)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    with open(
        "/gpfs/home/acad/ucl-elen/elkhoury/manon/CLIP-LoRA/Results/classif1_"
        + str(args.dataset)
        + "_"
        + str(args.model_name)
        + "_"
        + str(args.seed)
        + "_"
        + str(args.shots)
        + "_"
        + str(args.lr)
        + "_"
        + str(args.textual)
        + "_results.json",
        "w",
    ) as f:
        json.dump({"val_acc": acc_val, "test_acc": acc_test}, f)

    return

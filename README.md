# Exploring Foundation Models Fine-Tuning for Cytology Tasks [Submitted to ISBI25]

Implementation of **"Exploring Foundation Models Fine-Tuning for Cytology Tasks"**.

In this paper, we explore the application of existing foundation models to cytological classification tasks, focusing on low-rank adaptation (LoRA), a parameter-efficient fine-tuning method well-suited to few-shot learning scenarios. We evaluate five foundation models across four cytological classification datasets. Our results demonstrate that fine-tuning the pre-trained backbones with LoRA significantly enhances model performance compared to merely fine-tuning the classifier head, achieving state-of-the-art results on both simple and complex classification tasks while requiring fewer data samples.

Authors: **M. Dausort, [T. Godelaine](https://scholar.google.com/citations?user=xKcPd0oAAAAJ&hl=en&oi=ao), [M. Zanella](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao), [K. El Khoury](https://scholar.google.be/citations?user=UU_keGAAAAAJ&hl=fr), [I. Salmon](https://scholar.google.be/citations?user=S1dmusUAAAAJ&hl=en), [B. Macq](https://scholar.google.be/citations?user=H9pGN70AAAAJ&hl=fr)**

**NB:** This GitHub repository is based on the implementation of [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA). 

## Contents 

- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Installation 

**NB:** The Python version used is 3.9.13.

1. Create a virtual environment, clone the GitHub repository, and install the required packages:
   ```bash
   python3 -m venv cyto_ft_venv
   source cyto_ft_venv/bin/activate
   pip3 install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2
   git clone https://github.com/mdausort/Cytology-fine-tuning.git
   cd Cytology-fine-tuning
   pip3 install -r requirements.txt
   ```

2. Download datasets:

   - **Body Cavity Fluid Cytology (BCFC, kaggle2)**
     [Download BCFC](https://www.kaggle.com/datasets/cmacus/body-cavity-fluid-cytology-images)

   - **Mendeley LBC Cervical Cancer (MLCC, kaggle1)**
     [Download MLCC](https://www.kaggle.com/datasets/blank1508/mendeley-lbc-cervical-cancer-)

   - **SIPaKMeD**
     [Download the five folders](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed), decompress and place them into a folder named 'sipakmed'.

   - **HiCervix**
     [Download HiCervix](https://zenodo.org/records/11087263)

**Important**: All file paths in scripts are set with the placeholder "TO CHANGE". You will need to search for this placeholder in the cloned repository's files and replace it with the appropriate path ```/root/path/``` as specified for your system.

## Usage 

### Experience 1: Linear Classifier

```bash
python3 main.py --root_path ./data/ \
                --dataset {dataset} \
                --seed {seed} \
                --shots 0 \
                --lr {lr} \
                --n_iters 50 \
                --position None \
                --encoder None \
                --params None \
                --r 0 \
                --model_name {model_name} \
                --num_classes {num_classes} \
                --level {level} \
                --backbone {backbone} \
                --textual False
```

### Experience 2: LoRA Few-Shot Adaptation

```bash
python3 main_lora.py --root_path ./data/ \
                     --dataset {dataset} \
                     --seed {seed} \
                     --shots {shots} \
                     --lr {lr} \
                     --n_iters 50 \
                     --position "all" \
                     --encoder "vision" \
                     --pourcentage 0 \
                     --params "q v" \
                     --r 2 \
                     --model_name {model_name} \
                     --num_classes {num_classes} \
                     --level {level} \
                     --backbone {backbone} \
                     --dropout_rate 0.25
```

### Experience 3: Pushing Model Fine-Tuning Limits

```bash
python3 main_lora.py --root_path ./data/ \
                     --dataset hicervix \
                     --seed {seed} \
                     --shots 0 \
                     --lr 1e-3 \
                     --n_iters 100 \
                     --position "all" \
                     --encoder "vision" \
                     --pourcentage {pourcentage} \
                     --params "q k v o" \
                     --r 16 \
                     --model_name clip \
                     --num_classes 25 \
                     --level 3 \
                     --backbone ViT-L/14 \
                     --dropout_rate 0.25
```

## Contact 

If you have any questions, you can contact us by email: manon.dausort@uclouvain.be, tiffanie.godelaine@uclouvain.be

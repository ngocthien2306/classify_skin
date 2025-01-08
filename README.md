# Skin Lesion Classification using Deep Learning

Deep learning approach for skin lesion classification using the HAM10000 dataset. This project implements a robust classification system for 7 different types of skin lesions:
- Melanoma (MEL)
- Melanocytic nevus (NV) 
- Basal cell carcinoma (BCC)
- Actinic keratosis/Bowen's disease (AKIEC)
- Benign keratosis (BKL)
- Dermatofibroma (DF)
- Vascular lesion (VASC)

## Features

- Data augmentation to handle class imbalance

## Project Structure

```
classify_skin/
│
├── dataset/                      # Data directory
│   ├── raw/                  # Raw dataset
│   ├── processed/            # Processed data
│   └── augmented/           # Augmented data
│
├── src/                      # Source code
│   ├── data/                # Data processing utilities
│   ├── models/              # Model architectures
│   ├── training/            # Training code
│   ├── visualization/       # Visualization utilities
│   └── utils/               # Helper utilities
│
├── scripts/                  # Training/evaluation scripts
├── config/                   # Configuration files
├── notebooks/               # Jupyter notebooks
└── requirements.txt         # Dependencies
```


## Usage

#### Training
```
python scripts/train.py
```

#### Evaluation
```
python scripts/evaluate.py
python scripts/evaluate_new.py
```

#### Augmentation and Split data
```
python scripts/data/prepate_data.py
python scripts/data/split_data.py
```


## Citation

If you use this code in your research, please cite:

```bibtex
@software{classify_skin,
  author = {Thien Nguyen, Mai Duy, Cao Cong Danh},
  title = {Skin Lesion Classification using Deep Learning},
  year = {2024},
  url = {https://github.com/ngocthien2306/classify_skin}
}
```

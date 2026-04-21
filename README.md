# rPPG Heart Rate Estimation — Cross-Dataset Domain Adaptation

**Built on top of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) (ubicomplab/UW)**

This fork extends the original toolbox with a domain-adaptive augmentation strategy for cross-dataset generalisation, developed as part of my MSc dissertation at UCL (Data Science and Machine Learning, 2023).

---

## Key Result

| Setup | Model | Pearson r (PURE) |
|-------|-------|-----------------|
| Train UBFC-rPPG → Test PURE (baseline) | DeepPhys | 0.67 |
| Train UBFC-rPPG → Test PURE + **domain adaptation** | DeepPhys | **0.96** |
| Train UBFC-rPPG → Test PURE + domain adaptation | EfficientPhys | 0.94 |
| Train UBFC-rPPG → Test PURE + domain adaptation | TS-CAN | 0.93 |

Metric: Pearson correlation between estimated and ground-truth heart rate on the [PURE dataset](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profil/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure).

---

## What I Added

The upstream toolbox provides model architectures, training loops, and baseline configs. My additions:

### 1. Domain-Adaptive Brightness Augmentation (`dataset/data_loader/PURELoader.py`)

Cross-dataset performance often degrades due to distribution shift in lighting and colour. PURE is a controlled indoor dataset (brightness range ~28–108); UBFC-rPPG captures different lighting conditions (~95–181). To close this gap:

- Computed empirical brightness statistics for both datasets
- For each training video, generated a second brightness-jittered copy with transforms sampled uniformly between the two distributions
- Concatenated original + augmented frames, doubling effective training set size
- Labels duplicated and resampled to match

```python
# PURELoader.py — generate_constant_transform()
brightness_const = random.uniform(
    smartphone_brightness_min / pure_brightness_max,   # lower bound
    smartphone_brightness_max / pure_brightness_min    # upper bound
)
constant_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=brightness_const),
    transforms.ToTensor()
])
```

### 2. Face-Box Crop Preprocessing Configs

Added cross-dataset configs with tuned face-box extraction:

- `USE_LARGE_FACE_BOX: True` with `LARGE_BOX_COEF: 1.5` (tuned from default 2.0)
- Resolutions: 72×72 and 96×96
- Evaluation methods: FFT and peak detection
- Models: DeepPhys, EfficientPhys, TS-CAN, PhysNet

Config naming convention: `{train_dataset}_{train_dataset}_{test_dataset}_{model}_BASIC_{eval_method}_{preprocess}.yaml`

Example: `UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC_FFT_facebox.yaml`

---

## Architectures

This repo includes the following neural rPPG methods from the upstream toolbox:

| Model | Paper |
|-------|-------|
| DeepPhys | [Chen et al., ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Weixuan_Chen_DeepPhys_Video-Based_Physiological_ECCV_2018_paper.pdf) |
| EfficientPhys | [Liu et al., WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf) |
| TS-CAN | [Liu et al., NeurIPS 2020](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf) |
| PhysNet | [Yu et al., BMVC 2019](https://bmvc2019.org/wp-content/uploads/papers/0186-paper.pdf) |

---

## Setup

```bash
# Clone
git clone https://github.com/qpingWIN/Disser_final.git
cd Disser_final

# Install dependencies
pip install -r requirements.txt
```

**Datasets:** Download [PURE](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profil/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure) and [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg) and update `DATA_PATH` / `CACHED_PATH` in the relevant YAML config.

**Run cross-dataset experiment (train UBFC-rPPG → test PURE with domain adaptation):**

```bash
python main.py \
  --config_file configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC_FFT_facebox.yaml
```

**Hardware:** Configs include `DEVICE: mps` for Apple Silicon and `DEVICE: cpu` for CPU-only runs. Change to `cuda:0` for NVIDIA GPU.

---

## Repository Structure

```
├── configs/
│   ├── train_configs/          # Training + evaluation configs (includes my facebox variants)
│   └── infer_configs/          # Inference-only configs
├── dataset/
│   └── data_loader/
│       ├── PURELoader.py       # Modified: domain-adaptive augmentation
│       └── UBFCrPPGLoader.py   # Modified: additional preprocessing
├── neural_methods/             # Model architectures (upstream)
├── Pure_groundtruth_df copy.ipynb  # Benchmark analysis notebook
└── main.py
```

---

## Attribution

Base codebase: [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) by the UbiComp Lab, University of Washington.  
Domain adaptation extensions and cross-dataset configs: Pavlo Petrashko, UCL MSc Dissertation, 2023.

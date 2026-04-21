# Remote Heart Rate Estimation for Mobile Video: Augmentation-Based Domain Adaptation

Internship project at [Klarity Health](https://www.klarity.co.uk) · UCL MSc Dissertation, Department of Computer Science, 2023.

Neural rPPG models trained on controlled lab datasets degrade significantly when tested on unconstrained smartphone video. This project systematically evaluates three augmentation strategies to close that gap, benchmarked across TS-CAN, DeepPhys, and EfficientPhys on both public datasets and a proprietary mobile video dataset collected with Klarity Health.

---

## Results

### Individual augmentations on PURE (cross-dataset: trained on UBFC-rPPG)

| Augmentation | Model | MAE (bpm) ↓ | Pearson r ↑ |
|---|---|---|---|
| Baseline (no augmentation) | TS-CAN | ~baseline | 0.67 |
| Face-box crop (1.5× box, 72×72) | TS-CAN | −4.5 bpm vs baseline | **0.96** |
| Face-box crop (1.5× box, 72×72) | DeepPhys | improved | improved |
| Face-box crop (1.5× box, 72×72) | EfficientPhys | improved | improved |

### Individual augmentations on My_dataset (smartphone video, Klarity Health)

| Augmentation | Models | Effect |
|---|---|---|
| Video settings adjustment | DeepPhys, EfficientPhys | −3 bpm MAE |
| Video settings adjustment | TS-CAN | MAE decrease observed |
| Brightness domain adaptation | All three models | −1.5 bpm MAE |
| Face-box crop | All three models | MAE improvement |

### Combined augmentations

| Model | Result |
|---|---|
| DeepPhys | Improved on My_dataset |
| TS-CAN | MAE and MAPE increased (degraded) |
| EfficientPhys | MAE and MAPE increased (degraded) |

Combined augmentation produced inconsistent results across models — individual augmentations are more reliable than their joint application. See dissertation for full discussion.

---

## What This Project Does

Standard rPPG benchmarks use controlled datasets (PURE, UBFC-rPPG): fixed lighting, tripod-mounted cameras, cooperative subjects. Real-world mobile video breaks all of these assumptions.

Three augmentation strategies were designed and evaluated to align training data distributions toward real-world mobile conditions:

### 1. Face-Box Crop Tuning

Systematic evaluation of face-box extraction parameters:
- `LARGE_BOX_COEF`: tested 1.5× vs 2.0× (default)
- Resolutions: 72×72 and 96×96
- Detection: static (first-frame) vs dynamic (every N frames)

Best result: `LARGE_BOX_COEF=1.5`, 72×72, static detection → Pearson 0.96 on PURE for TS-CAN.

### 2. Brightness Domain Adaptation

PURE dataset has a narrower brightness range (~28–108) than smartphone video (~95–181). For each training video, a brightness-jittered copy is generated with the transform factor sampled uniformly between the two distributions, then concatenated with the original — effectively doubling training size while calibrating the colour distribution toward the target domain.

```python
# PURELoader.py
brightness_const = random.uniform(
    smartphone_brightness_min / pure_brightness_max,   # 0.88
    smartphone_brightness_max / pure_brightness_min    # 6.46
)
transform = transforms.ColorJitter(brightness=brightness_const)
```

### 3. Video Settings Adjustment

Modifications to video resolution, frame rate, and related parameters in the preprocessing pipeline to better match the characteristics of mobile-captured video. Reduced MAE by ~3 bpm for DeepPhys and EfficientPhys on the smartphone dataset.

---

## Models

| Model | Paper |
|---|---|
| TS-CAN | [Liu et al., NeurIPS 2020](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf) |
| DeepPhys | [Chen et al., ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Weixuan_Chen_DeepPhys_Video-Based_Physiological_ECCV_2018_paper.pdf) |
| EfficientPhys | [Liu et al., WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf) |

---

## Datasets

| Dataset | Type | Used for |
|---|---|---|
| [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg) | Public, controlled | Training |
| [PURE](https://www.tu-ilmenau.de/neurob/data-sets-code/pulse-rate-detection-dataset-pure) | Public, controlled | Cross-dataset test |
| My_dataset | Proprietary, smartphone | Real-world test (Klarity Health) |

---

## Setup

```bash
git clone https://github.com/qpingWIN/Disser_final.git
cd Disser_final
pip install -r requirements.txt
```

Update `DATA_PATH` and `CACHED_PATH` in the config file to point to your local dataset copies.

**Run cross-dataset experiment (UBFC-rPPG → PURE, face-box crop, DeepPhys):**

```bash
python main.py \
  --config_file configs/train_configs/UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC_FFT_facebox.yaml
```

**Hardware:** configs include `mps` (Apple Silicon) and `cpu` variants. Change `DEVICE` to `cuda:0` for NVIDIA GPU.

---

## Config Naming Convention

```
{train_set}_{train_set}_{test_set}_{model}_BASIC_{eval_method}_{preprocess}.yaml
```

Example: `UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC_FFT_facebox96x96.yaml`  
= Train on UBFC-rPPG, test on PURE, TS-CAN, FFT evaluation, 96×96 face-box crop.

---

## Limitations and Future Work

- Combined augmentations degraded TS-CAN and EfficientPhys — interference between augmentations likely caused unexpected distribution shifts
- My_dataset is small, limiting statistical significance
- Dynamic face detection, forehead/cheek ROI, and video stabilisation are natural next steps
- Performance across skin tones and lighting conditions not fully characterised

---

## Codebase

Model architectures, training infrastructure, and unsupervised baselines from [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) (UbiComp Lab, University of Washington). Augmentation implementations, cross-dataset configs, and experimental analysis by Pavlo Petrashko, UCL MSc Dissertation, 2023.

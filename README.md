# Détection de Fractures Osseuses par Deep Learning

<div align="center">

![Banner](https://img.shields.io/badge/Deep%20Learning-YOLOv8-blue?style=for-the-badge&logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)
![Classes](https://img.shields.io/badge/Classes-7-green?style=for-the-badge)
![mAP@50](https://img.shields.io/badge/mAP%40050-63.2%25-orange?style=for-the-badge)
![Precision](https://img.shields.io/badge/Precision-79.8%25-red?style=for-the-badge)

**Système de détection automatique de fractures osseuses sur radiographies X**  
*Projet Data Science — École Polytechnique de Tunisie*

</div>

---

## Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Demo Video](#demo-video)
- [Dataset](#dataset)
- [Architecture du Projet](#architecture-du-projet)
- [Pipeline Complet](#pipeline-complet)
- [Résultats](#résultats)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure des Fichiers](#structure-des-fichiers)
- [Equipe](#equipe)

---

## Vue d'ensemble

Ce projet implémente un système de **détection d'objets multi-classe** capable d'identifier et localiser automatiquement des fractures osseuses sur des radiographies X. Il repose sur l'architecture **YOLOv8n** (Ultralytics) et couvre **7 classes anatomiques** distinctes.

### Problématique

En radiologie d'urgence, le diagnostic d'une fracture requiert l'attention d'un radiologue qualifié, ce qui peut entraîner des délais critiques. Ce système d'assistance au diagnostic vise à accélérer le triage et réduire le risque de fractures non détectées (faux négatifs).

### Classes détectées

| ID | Classe | Zone Anatomique |
|---|---|---|
| 0 | `fracture_elbow` | Coude |
| 1 | `fracture_humerus` | Humérus |
| 2 | `fracture_finger` | Doigt |
| 3 | `fracture_forearm` | Avant-bras |
| 4 | `fracture_femur` | Fémur |
| 5 | `normal` | Os sain (pas de fracture) |
| 6 | `fracture_leg` | Jambe |

---

## Demo Video

**Démonstration du modèle en action — détection de fractures en temps réel**

<div align="center">

[![Watch the Demo Video](https://img.shields.io/badge/▶️%20Regarder%20la%20Démo-Google%20Drive-red?style=for-the-badge)](https://drive.google.com/file/d/1dH8LXTJncZEdX_nktRS7PVUqtfjjsVzY/view?usp=sharing)

**Ou :**

<video width="720" height="480" controls style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin: 20px auto; display: block;">
  <source src="https://drive.google.com/uc?export=download&id=1dH8LXTJncZEdX_nktRS7PVUqtfjjsVzY" type="video/mp4">
  <p>Votre navigateur ne supporte pas la vidéo HTML5. <a href="https://drive.google.com/file/d/1dH8LXTJncZEdX_nktRS7PVUqtfjjsVzY/view?usp=sharing">Cliquez ici pour accéder à la vidéo.</a></p>
</video>

</div>

### Contenu de la vidéo :
- ✅ **Inférence en temps réel** sur des radiographies de test
- ✅ **Affichage des bounding boxes** avec score de confiance par classe
- ✅ **Comparaison Modèle 1** (baseline) vs **Modèle 2** (fine-tuné)
- ✅ **Visualisation des détections** par zone anatomique

> **💡 Conseil :** Pour une meilleure expérience, assurez-vous d'une connexion Internet stable. La vidéo se chargera automatiquement lors du scroll.

---

## Dataset

### Dataset personnalisé sur Kaggle

**[fracture-research-dataset — Kaggle](https://www.kaggle.com/datasets/aymenouerghi07/fracture-dataset/data)**

Ce dataset a été **entièrement construit par l'équipe** à partir de deux sources hétérogènes, puis unifié, nettoyé et formaté au format YOLO.

### Sources utilisées

| Source | Contenu | Classes extraites |
|---|---|---|
| **Roboflow Universe** | 5 datasets spécialisés | elbow, humerus, finger, forearm, femur |
| **FracAtlas (Figshare)** | Dataset médical annoté | normal, leg |

### Statistiques du dataset final

| Split | Images | Annotations |
|---|---|---|
| Train | 5 737 | ~6 750 |
| Validation | 883 | ~441 |
| Test | 753 | ~450 |
| **Total** | **7 373** | **~7 641** |

> ⚠️ Après downsampling de `fracture_finger` (classe sur-représentée : 27 714 → 1 448 annotations)

### Format

Le dataset est au format **YOLO v8** :
```
fracture_research_dataset/
├── train/
│   ├── images/      # Radiographies .jpg
│   └── labels/      # Annotations .txt (class cx cy w h normalisés)
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml        # Configuration des 7 classes
```

### Pipeline de construction du dataset

```
Roboflow API          FracAtlas (Kaggle)
     │                      │
     ▼                      ▼
5 datasets bruts     Images normales + leg
     │                      │
     ▼                      ▼
Remapping des labels ──────► Unification des IDs de classes
     │
     ▼
Split 70/15/15 (seed=42, reproductible)
     │
     ▼
Génération data.yaml  ──►  Dataset YOLO unifié (7 classes)
```

---

## Architecture du Projet

```
FRACTUREAI/
│
├── Notebooks/
│   ├── Collection des données.ipynb           # Construction & nettoyage du dataset
│   ├── notebook_Fracture_Detection.ipynb      # Pipeline ML complet (EDA -> Modele 2)
│   └── notebook_Supplementaire_DETR.ipynb     # Essai architecture DETR
│
├── Rapport - Très important pour la Lecture/
│   └── rapport_Détection_de_Fractures_Osseuses.pdf  # Rapport académique (120 pages)
│
├── Demo.mp4                                   # Démonstration du modèle
├── main.py                                    # Script principal
├── pyproject.toml                             # Dépendances Python
└── README.md                                  # Ce fichier
```

### Notebooks

| Notebook | Rôle | Étapes |
|---|---|---|
| `Collection_des_données.ipynb` | Construction du dataset | Téléchargement Roboflow, remapping labels, intégration FracAtlas, génération YAML |
| `notebook_Fracture_Detection.ipynb` | Pipeline ML complet | EDA, preprocessing, Modèle 1, augmentation, Modèle 2, comparaison |
| `notebook_Supplementaire_DETR.ipynb` | Exploration DETR | Implémentation et évaluation de l'architecture Transformer |

---

## Pipeline Complet

### Étape 1 — Exploration des Données (EDA)

- Distribution des 7 classes par split (détection du déséquilibre)
- Analyse des propriétés des images : résolution, aspect ratio, canaux
- Qualité des images : netteté via **variance du Laplacien**, exposition (sur/sous)
- Inspection visuelle par classe et image moyenne par classe
- Analyse des bounding boxes YOLO : taille, aspect ratio, position des centres

### Étape 2 — Preprocessing : Downsampling de `fracture_finger`

La classe `fracture_finger` représentait ~82% des annotations. Stratégie de downsampling intelligente :

```
Images "pure finger"   → sous-échantillonnées jusqu'à la cible
Images "mixtes"        → toutes conservées (évite la perte d'autres classes)
Images sans finger     → toutes conservées

Résultat train : 24 264 → 1 234 annotations finger (-95%)
                 18 753 images supprimées sur 24 490
```

### Étape 3 — Entraînement Modèle 1 (Baseline)

```python
model = YOLO('yolov8n.pt')
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    # GPU : NVIDIA Tesla T4
)
```

- **50 epochs**, image 640×640, GPU Tesla T4
- Losses suivies : `box_loss`, `cls_loss`, `dfl_loss`
- Seuil de confiance inférence : 0.25 | Seuil IoU matching : 0.50

### Étape 4 — Augmentation Offline

Justifiée par les références scientifiques (Garcea et al., 2023 ; Shorten & Khoshgoftaar, 2019) pour l'imagerie médicale :

| Transformation | Paramètres | Justification |
|---|---|---|
| Flip horizontal | p = 0.5 | Symétrie anatomique |
| Rotation | ±10° | Variabilité de positionnement |
| Brightness + Contrast | ±20% | Hétérogénéité des appareils radio |
| Bruit gaussien | std ∈ [5, 20] | Simulation du grain capteur |
| Cutout | 5–10% surface | Robustesse aux occlusions |

Résultat : **5 737 → 11 474 images** en train (+100%)

### Étape 5 — Modèle 2 (Fine-tuning intra-domaine)

```python
# Initialisation avec les poids du Modèle 1
model2 = YOLO(best_model1_path)  # Transfer learning intra-domaine
model2.train(data='data_augmented.yaml', epochs=50, imgsz=640)
```

Le modèle 2 repart des poids appris sur les fractures → convergence accélérée.

### Étape 6 — Error Analysis

Pour chaque image test : matching greedy prédictions/GT par IoU ≥ 0.50
- Comptage TP / FP / FN par classe
- IoU moyen sur les vrais positifs
- Analyse des erreurs les plus fréquentes par zone anatomique

### Bonus — Essai DETR (DEtection TRansformer)

L'équipe a également exploré l'architecture **DETR** (Carion et al., ECCV 2020) basée sur les Transformers, entraîné sur 7 epochs. Cette expérimentation, menée en parallèle, démontre la maîtrise des architectures de vision modernes au-delà de YOLO.

---

## Résultats

### Métriques Finales

| Métrique | Modèle 1 (Baseline) | Modèle 2 (Fine-tuné) | Δ |
|---|---|---|---|
| **Val mAP@50** | 0.6180 | **0.6273** | +0.0093 |
| Val mAP@50-95 | 0.2596 | 0.2445 | -0.0152 |
| **Val Precision** | 0.6680 | **0.7976** | **+0.1296** |
| Val Recall | 0.6137 | 0.5929 | -0.0208 |
| **Test mAP@50** | **0.6323** | 0.6310 | -0.0012 |
| Test mAP@50-95 | 0.2628 | 0.2344 | -0.0284 |
| IoU moyen (TP) | **0.7310** | 0.6970 | -0.0341 |
| Vitesse inférence | — | **3.4 ms/image** | — |

### Interprétation

Le **Modèle 2** est recommandé pour usage clinique car :
- Precision de **79,8%** → 80% des alertes sont de vraies fractures (moins de faux positifs)
- Spécificité sur la classe `normal` : **0.93** → très peu de fausses alarmes sur os sains
- Vitesse d'inférence de **3.4 ms** → compatible avec l'usage en temps réel
- Meilleur Recall sur humérus, fémur et jambe (os longs)

> **Note clinique :** En contexte médical, le Recall (sensibilité) est la métrique prioritaire — rater une fracture est plus grave qu'une fausse alarme. Le Modèle 1 présente un meilleur Recall global (0.614 vs 0.593) et pourrait être préféré dans certains contextes de dépistage.

### Courbes d'entraînement

La progression sur 50 epochs montre une convergence régulière :
- `box_loss` : ~2.40 → ~1.85
- `cls_loss` : ~2.60 → ~1.70
- `dfl_loss` : ~1.90 → ~1.60

Aucun signe d'overfitting — les courbes train et validation restent proches.

---

## Installation

### Prérequis

- Python 3.9+
- GPU NVIDIA (recommandé, CUDA 12+) ou CPU
- 8 Go RAM minimum

### Cloner le dépôt

```bash
git clone https://github.com/letdatatalk-ao/FRACTUREAI.git
cd FRACTUREAI
```

### Installer les dépendances

```bash
pip install ultralytics albumentations opencv-python-headless pyyaml tqdm imagehash
pip install roboflow  # uniquement pour la collecte de données
```

### Télécharger le dataset

```bash
# Télécharger depuis Kaggle
kaggle datasets download -d aymenouerghi07/fracture-dataset
unzip fracture-dataset.zip -d ./data/
```

> Ou accéder directement : [https://www.kaggle.com/datasets/aymenouerghi07/fracture-dataset/data](https://www.kaggle.com/datasets/aymenouerghi07/fracture-dataset/data)

---

## Utilisation

### Entraînement

```python
from ultralytics import YOLO

# Modèle 1 — baseline
model = YOLO('yolov8n.pt')
model.train(
    data='data/fracture_research_dataset/data.yaml',
    epochs=50,
    imgsz=640,
    project='runs',
    name='fracture_model1_yolov8n'
)

# Modèle 2 — fine-tuning depuis M1
model2 = YOLO('runs/fracture_model1_yolov8n/weights/best.pt')
model2.train(
    data='data/fracture_augmented/data.yaml',
    epochs=50,
    imgsz=640,
    project='runs',
    name='fracture_model2_finetuned'
)
```

### Inférence

```python
from ultralytics import YOLO

model = YOLO('runs/fracture_model2_finetuned/weights/best.pt')

# Sur une image
results = model('radiographie.jpg', conf=0.25)
results[0].show()

# Sur un dossier
results = model('data/test/images/', conf=0.25, save=True)
```

### Évaluation

```python
# Validation
metrics = model.val(data='data.yaml', split='val')
print(f"mAP@50: {metrics.map50:.4f}")
print(f"Precision: {metrics.mp:.4f}")
print(f"Recall: {metrics.mr:.4f}")

# Test
metrics_test = model.val(data='data.yaml', split='test')
```

---

## Structure des Fichiers

```
FRACTUREAI/
│
├── Notebooks/
│   ├── Collection des données.ipynb           # Construction & nettoyage du dataset
│   ├── notebook_Fracture_Detection.ipynb      # Pipeline ML complet (EDA -> Modele 2)
│   └── notebook_Supplementaire_DETR.ipynb     # Essai architecture DETR
│
├── Rapport - Très important pour la Lecture/
│   └── rapport_Détection_de_Fractures_Osseuses.pdf  # Rapport académique (120 pages)
│
├── Demo.mp4                                   # Démonstration du modèle
├── main.py                                    # Script principal
├── pyproject.toml                             # Dépendances Python
└── README.md                                  # Ce fichier
```

---

## Equipe

Projet réalisé dans le cadre du module **Data Science**  
**École Polytechnique de Tunisie — Année universitaire 2026/2027**

| Nom | Rôle |
|---|---|
| **Yassine MEGDICHE** | Data Engineering & Pipeline |
| **Aymen OUERGHI** | Dataset Kaggle & Modélisation |
| **Achref GUESMI** | Augmentation & Évaluation |
| **Salmen BOUAZIZ** | EDA & Visualisation |
| **Mohammed MISSAOUI** | DETR & Analyse d'erreurs |

**Encadrante :** Mme. Selima BESBES

---

## Références Scientifiques

```bibtex
@article{garcea2023augmentation,
  title={Data augmentation for medical imaging: A systematic literature review},
  author={Garcea, F. and Serra, A. and Lamberti, F. and Morra, L.},
  journal={Computers in Biology and Medicine},
  volume={146}, pages={106391}, year={2023}
}

@article{jocher2023yolov8,
  title={Ultralytics YOLOv8},
  author={Jocher, G. and Chaurasia, A. and Qiu, J.},
  year={2023},
  url={https://github.com/ultralytics/ultralytics}
}

@inproceedings{carion2020detr,
  title={End-to-End Object Detection with Transformers},
  author={Carion, N. and Massa, F. and Synnaeve, G. and Usunier, N. and Kirillov, A. and Zagoruyko, S.},
  booktitle={ECCV}, year={2020}
}

@misc{abedeen2023fracatlas,
  title={FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation},
  author={Abedeen, I. and Rahman, M.A. et al.},
  year={2023}, doi={10.6084/m9.figshare.22363012.v6}
}

@inproceedings{rezatofighi2019giou,
  title={Generalized Intersection over Union},
  author={Rezatofighi, H. et al.},
  booktitle={CVPR}, year={2019}
}
```

---

## Licence

Ce projet est réalisé à des fins académiques dans le cadre de l'École Polytechnique de Tunisie.  
Le dataset est disponible publiquement sur Kaggle sous licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

<div align="center">

**École Polytechnique de Tunisie — Data Science 2026/2027**

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&style=flat-square)](https://www.kaggle.com/datasets/aymenouerghi07/fracture-dataset/data)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-blue?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![GitHub](https://img.shields.io/badge/GitHub-FRACTUREAI-black?logo=github&style=flat-square)](https://github.com/letdatatalk-ao/FRACTUREAI)
[![EPT](https://img.shields.io/badge/Institution-EPT%20Tunis-red?style=flat-square)](https://www.ept.tn)
[![Demo Video](https://img.shields.io/badge/Demo-Google%20Drive-red?logo=google&style=flat-square)](https://drive.google.com/file/d/1dH8LXTJncZEdX_nktRS7PVUqtfjjsVzY/view?usp=sharing)

</div>

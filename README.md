Projet GitHub : https://github.com/Arthur94R/m1-bert

# 📚 TP3 - Résumé de texte avec BERT (BARThez)

**Projet GitHub :** https://github.com/Arthur94R/m1-bert

Projet universitaire — Master 1 Informatique Big Data, Université Paris 8

## 🎯 Objectif

Résumer automatiquement un chapitre du roman **"Léon l'Africain"** d'Amin Maalouf en utilisant un modèle **BERT français** (BARThez) pré-entraîné pour le résumé de texte.

## 📋 Description

Le projet utilise le modèle **BARThez** (BART français) pour :
- Tokeniser le texte du chapitre
- Générer un résumé automatique
- Analyser les mécanismes d'attention du modèle
- Visualiser la matrice d'attention

**Chapitre traité :** Pages 267-273 (Chapitre 33)

## 🛠️ Stack technique

- **Python 3.13**
- **PyTorch** — Deep learning
- **Transformers (Hugging Face)** — Modèles pré-entraînés
- **BARThez** — Modèle BERT français pour résumé
- **Matplotlib** — Visualisation
- **NumPy** — Calculs matriciels

## 📁 Structure

```
TP3-Bert/
├── src/
│   └── main.py
├── data/
│   └── chapitre33.txt                      → Texte du chapitre
├── results/
│   ├── attention_matrix_complete.png       → Matrice d'attention (générée)
│   └── attention_matrix_zoom.png           → Matrice zoomée (générée)
└── README.md
```

## 🚀 Installation et lancement

### 1. Créer un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# OU
venv\Scripts\activate      # Windows
```

### 2. Installer les dépendances

```bash
pip install torch transformers tokenizers numpy matplotlib sentencepiece protobuf
```

**Versions recommandées :**
```bash
pip install transformers==4.46.0 tokenizers==0.20.3
```

### 3. Lancer le script

```bash
cd src
python main.py
```

**Important :** Le fichier `chapitre33.txt` doit être dans le même dossier que `main.py`.

## 📊 Résultats

Le script affiche :
1. ✅ **Quelques tokens** (15 premiers)
2. ✅ **Résumé généré** (~100-300 mots)
3. ✅ **Attentions > 0.5** (liens forts entre texte source et résumé)
4. ✅ **Matrice d'attention** (statistiques + visualisation)
5. ✅ **Paramètres du modèle** (explications)

**Fichiers générés :**
- `attention_matrix_complete.png` — Heatmap complète
- `attention_matrix_zoom.png` — Zoom 30×30 avec tokens

## ⚙️ Paramètres du modèle

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `max_length` | 300 | Longueur maximale du résumé (en tokens) |
| `min_length` | 100 | Longueur minimale du résumé |
| `num_beams` | 4 | Beam search : explore 4 possibilités en parallèle |
| `length_penalty` | 2.0 | Favorise les résumés un peu plus longs |
| `early_stopping` | True | Arrête dès qu'un bon résumé est trouvé |
| `no_repeat_ngram_size` | 3 | Évite la répétition de 3-grammes |

## 🧠 Concepts clés

### BARThez
- **BART** = Bidirectional Auto-Regressive Transformer
- Version **française** de BART
- Pré-entraîné sur **OrangeSum** (dataset de résumés français)
- Résumé **abstractif** (génère de nouvelles phrases)

### Matrice d'attention
- Montre **quels mots du texte source** le modèle regarde pour générer **chaque mot du résumé**
- Valeurs entre 0 et 1 (poids d'attention)
- Permet de comprendre la **stratégie du modèle**

### Tokenisation
- Découpe le texte en **tokens** (sous-mots)
- Exemple : "appelé" → ["app", "elé"]
- Le modèle traite des tokens, pas des mots

## 🔍 Résumé obtenu

> *"Hans, un élève allemand du narrateur, était convaincu que le pape Léon X ne devait pas être considéré comme un saint par les chrétiens, car il avait été baptisé dans la basilique Saint-Pierre de Rome, où il avait reçu les prénoms de Jean et de Léon, ainsi que le nom de la famille Médicis. Le narrateur, qui s'appelait Jean-Léon, avait été baptisé le 6 janvier 1520, jour où Raphaël d'Urbino, le divin Raphaël, mourut trois mois plus tard. Après avoir quitté sa prison, Jean-Léon se rendit à Rome pour visiter la ville."*

**Analyse :**
- Résumé **cohérent** couvrant ~60% du chapitre
- Capture les **événements clés** : débat religieux, baptême, nouveau nom, mort de Raphaël, libération, visite de Rome
- Qualité correcte pour un texte source de **~2800 mots** résumé en **~300 tokens**
- Manque encore : le livre en arabe, Guicciardini, réflexions philosophiques sur Rome

**Résumé obtenu avec max_length=300 (au lieu de 150 initialement).**
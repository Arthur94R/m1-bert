import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')

# ===========================
# CONFIGURATION
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "moussaKam/barthez-orangesum-abstract"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, attn_implementation="eager").to(device)
model.eval()
model.config.output_attentions = True
model.config.return_dict = True

print(f"Modèle : {model_name}")
print(f"Device : {device}\n")

# ===========================
# TEXTE (depuis fichier)
# ===========================
print("Chargement du texte depuis chapitre33.txt...")
with open('../data/chapitre33.txt', 'r', encoding='utf-8') as f:
    texte_chapitre = f.read()

print(f"Longueur texte : {len(texte_chapitre.split())} mots\n")

# ===========================
# TOKENISATION
# ===========================
inputs = tokenizer(texte_chapitre, return_tensors="pt", max_length=1024, truncation=True).to(device)
input_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print(f"=== QUELQUES TOKENS ({len(tokens)} total) ===")
print("Premiers 15 :")
for i, token in enumerate(tokens[:15]):
    print(f"  {i:3d} : {token}")
print()

# ===========================
# RÉSUMÉ
# ===========================
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=300,
    min_length=100,
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True,
    no_repeat_ngram_size=3
)

resume = tokenizer.decode(summary_ids.sequences[0].tolist(), skip_special_tokens=True)

print("=== RÉSUMÉ ===")
print(resume)
print()

# ===========================
# ATTENTIONS > 0.5
# ===========================
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        decoder_input_ids=summary_ids.sequences,
        output_attentions=True
    )

cross_attentions = outputs.cross_attentions[-1]
attention_matrix = cross_attentions[0].cpu().numpy()
avg_attention = attention_matrix.mean(axis=0)

print("=== ATTENTIONS > 0.5 ===")
high_attention = np.where(avg_attention > 0.5)

if len(high_attention[0]) > 0:
    print(f"Nombre : {len(high_attention[0])}\n")
    for i in range(min(10, len(high_attention[0]))):
        dec_pos = high_attention[0][i]
        enc_pos = high_attention[1][i]
        val = avg_attention[dec_pos, enc_pos]
        
        dec_token = tokenizer.decode([summary_ids.sequences[0][dec_pos]])
        enc_token = tokenizer.decode([input_ids[enc_pos]])
        
        print(f"[{dec_pos:3d}, {enc_pos:3d}] = {val:.3f} | '{dec_token}' ← '{enc_token}'")
else:
    print("Aucune (attentions diffuses)")

print()

# ===========================
# MATRICE D'ATTENTION
# ===========================
print("=== MATRICE D'ATTENTION ===")
print(f"Shape : {avg_attention.shape}")
print(f"  {avg_attention.shape[0]} tokens résumé × {avg_attention.shape[1]} tokens source")
print()

# Statistiques
print("Statistiques de la matrice :")
print(f"  Min : {avg_attention.min():.4f}")
print(f"  Max : {avg_attention.max():.4f}")
print(f"  Moyenne : {avg_attention.mean():.4f}")
print(f"  Médiane : {np.median(avg_attention):.4f}")
print()

# Afficher un extrait de la matrice (10x10)
print("Extrait de la matrice (10 premiers tokens) :")
print(avg_attention[:10, :10])
print()

# Visualisation
print("Génération de la visualisation...")

# Graphique 1 : Matrice complète
plt.figure(figsize=(14, 8))
plt.imshow(avg_attention, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar(label='Poids d\'attention')
plt.xlabel('Position dans le texte source')
plt.ylabel('Position dans le résumé')
plt.title('Matrice d\'attention (moyenne sur toutes les têtes)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'attention_matrix_complete.png'), dpi=300)
plt.close()
print("Sauvegardé : attention_matrix_complete.png")

# Graphique 2 : Zoom sur les 30 premiers tokens
max_display = min(30, avg_attention.shape[0], avg_attention.shape[1])
attention_subset = avg_attention[:max_display, :max_display]

plt.figure(figsize=(12, 10))
plt.imshow(attention_subset, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Poids d\'attention')
plt.xlabel('Tokens source')
plt.ylabel('Tokens résumé')
plt.title(f'Matrice d\'attention (zoom {max_display}×{max_display})')

# Ajouter les tokens sur les axes
encoder_tokens = [tokenizer.decode([input_ids[i]]) for i in range(min(max_display, len(input_ids)))]
decoder_tokens = [tokenizer.decode([summary_ids.sequences[0][i]]) for i in range(min(max_display, len(summary_ids.sequences[0])))]

plt.xticks(range(len(encoder_tokens)), encoder_tokens, rotation=90, fontsize=8)
plt.yticks(range(len(decoder_tokens)), decoder_tokens, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'attention_matrix_zoom.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Sauvegardé : attention_matrix_zoom.png")
print()

# ===========================
# PARAMÈTRES
# ===========================
print("=== PARAMÈTRES ===")
print("max_length=150 : Longueur max du résumé")
print("min_length=50 : Longueur min du résumé")
print("num_beams=4 : Beam search (explore 4 possibilités)")
print("length_penalty=2.0 : Favorise résumés plus longs")
print("early_stopping=True : Arrêt dès bon résumé trouvé")
print("no_repeat_ngram_size=3 : Évite répétitions de 3-grammes")
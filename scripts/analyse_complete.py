"""
=================================================================================
PROJET DE CLASSIFICATION SUPERVISÉE - VERSION FINALE
Prédiction de réclamations d'assurance automobile
=================================================================================
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
import sys

# Imports scikit-learn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                            recall_score, f1_score, classification_report)

warnings.filterwarnings('ignore')

# Essayer matplotlib en arrière-plan
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except:
    PLOT_AVAILABLE = False

# =============================================================================
# ÉTAPE 1: IMPORTATION DES DONNÉES
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 1: IMPORTATION DES DONNÉES")
print("="*80)

# Recherche du fichier CSV dans les emplacements possibles
possible_paths = [
    Path(__file__).parent / "car_insurance.csv",
    Path(__file__).parent.parent / "data" / "car_insurance.csv",
    Path("data/car_insurance.csv")
]

csv_path = None
for p in possible_paths:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    print(f"[ERREUR] Fichier car_insurance.csv introuvable dans {possible_paths}")
    sys.exit(1)

df = pd.read_csv(csv_path)

print(f"[OK] Fichier importé: {df.shape[0]} lignes x {df.shape[1]} colonnes")

# Vérification explicite des données manquantes (Étape 4)
print("\nNombre de valeurs manquantes par variable (Étape 4) :")
print(df.isna().sum())

# Traitement des données manquantes (Étape 5 - Solution à privilégier : fillna)
df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
df['annual_mileage'] = df['annual_mileage'].fillna(df['annual_mileage'].median())
print(f"[OK] Données manquantes imputées par la médiane.")

# Traitement des données aberrantes (Étape 5)
# On remplace les valeurs de speeding_violations > 50 par la médiane
median_speeding = df['speeding_violations'].median()
outliers_count = len(df[df['speeding_violations'] > 50])
df.loc[df['speeding_violations'] > 50, 'speeding_violations'] = median_speeding
print(f"[OK] {outliers_count} valeurs aberrantes dans speeding_violations remplacées par la médiane.")

print(f"\nAperçu:")
print(df.head(3))
print(f"\nInfo:")
print(df.info())
print(f"\nStatistiques:")
print(df.describe())

print(f"\nObjectif: Prédire si un client fait une réclamation (outcome: 0/1)")
value_counts = df['outcome'].value_counts()
for label, count in value_counts.items():
    pct = count / len(df) * 100
    print(f"   Classe {int(label)}: {count:,} ({pct:.1f}%)")

# =============================================================================
# ÉTAPE 2: EXAMEN DES DONNÉES
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 2: EXAMEN EXPLORATOIRE")
print("="*80)

missing = df.isnull().sum()
if missing.sum() == 0:
    print("\n[OK] Aucune donnée manquante!")
else:
    print(f"\n[ATTENTION] Données manquantes:\n{missing[missing > 0]}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nNumériques ({len(numeric_cols)}): {numeric_cols[:5]}...")
print(f"Catégorielles ({len(categorical_cols)}): {categorical_cols}")

# Histogrammes
if PLOT_AVAILABLE:
    print(f"\nCréation histogrammes...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            axes[idx].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[idx].set_title(col, fontsize=10)
    
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'histogrammes_variables.png', dpi=200)
    print("[OK] Histogrammes sauvegardés")
    plt.close()

# =============================================================================
# ÉTAPE 3: PRÉPARATION DES DONNÉES
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 3: PRÉPARATION")
print("="*80)

df_prep = df.copy()

# Supprimer colonnes inutiles
cols_drop = ['id', 'postal_code']
print(f"\nSuppression: {cols_drop}")
df_prep = df_prep.drop(columns=cols_drop)

# Encoder catégorielles
cat_cols = df_prep.select_dtypes(include=['object']).columns.tolist()
print(f"\nEncodage: {cat_cols}")

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_prep[col] = le.fit_transform(df_prep[col].astype(str))
    label_encoders[col] = le
    print(f"   {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# X et y
X = df_prep.drop(columns=['outcome'])
y = df_prep['outcome']

print(f"\n[OK] X shape: {X.shape}, y shape: {y.shape}")

# Normaliser
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(f"[OK] Données normalisées (moyenne~0, std~1)")

# =============================================================================
# ÉTAPE 4: CORRÉLATIONS
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 4: CORRÉLATIONS")
print("="*80)

df_corr = X_scaled.copy()
df_corr['outcome'] = y.values
corr_mat = df_corr.corr()
outcome_corr = corr_mat['outcome'].sort_values(ascending=False)

print(f"\nCorrélations avec outcome:")
for var, corr in outcome_corr.head(6).items():
    print(f"   {var:<25} {corr:>8.4f}")

print(f"\nCoefficient: proche +1 = corrélation positive, proche 0 = faible")

# Heatmap
if PLOT_AVAILABLE:
    print(f"\nCréation matrice de corrélation...")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax)
    plt.title('Matrice de Correlation', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'correlation_matrix.png', dpi=200)
    print("[OK] Matrice sauvegardée")
    plt.close()

# =============================================================================
# ÉTAPE 5: TRAIN/TEST SPLIT
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 5: DIVISION TRAIN/TEST")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n[OK] Train: {X_train.shape[0]} (70%), Test: {X_test.shape[0]} (30%)")

# =============================================================================
# ÉTAPE 6: LOGISTIC REGRESSION
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 6: RÉGRESSION LOGISTIQUE")
print("="*80)

print(f"\nEntrainement...")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)
print(f"[OK] Modèle entraîné!")

print(f"\nRegression Logistique:")
print(f"   Hypothese: log(p(y=1|x)/p(y=0|x)) = w0 + Sum(wi*xi)")
print(f"   Minimisation: Maximum de vraisemblance (descente de gradient)")
print(f"   Parametres: Coefficients et intercept estimes")

# =============================================================================
# ÉTAPE 7: ÉVALUATION
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 7: ÉVALUATION")
print("="*80)

y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)

print(f"\nPremiers 15 resultats:")
print(f"{'Index':<6} {'Prediction':<12} {'Probabilite':<15} {'Reel':<6}")
print("-" * 45)
for i in range(min(15, len(y_test))):
    prob = max(y_pred_proba[i])
    print(f"{i:<6} {y_pred[i]:<12} {prob:<15.4f} {int(y_test.iloc[i]):<6}")

# Metriques
acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall_val = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nMETRIQUES:")
print(f"   Accuracy:  {acc:.4f} - Taux de prediction correcte")
print(f"   Precision: {prec:.4f} - Fiabilite des predictions positives")
print(f"   Recall:    {recall_val:.4f} - Detection des vrais positifs")
print(f"   F1-Score:  {f1:.4f} - Equilibre precision/recall")

print(f"\nMatrice de Confusion:")
print(f"              Prediction")
print(f"            Negatif  Positif")
print(f"Reel Negatif {conf[0,0]:<8} {conf[0,1]:<8}")
print(f"     Positif {conf[1,0]:<8} {conf[1,1]:<8}")

print(f"\nRapport complet:")
print(classification_report(y_test, y_pred, target_names=['No Claim', 'Made Claim']))

# Confusion matrix plot
if PLOT_AVAILABLE:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Claim', 'Made Claim'],
                yticklabels=['No Claim', 'Made Claim'])
    plt.title('Matrice de Confusion', fontweight='bold')
    plt.ylabel('Reel')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'confusion_matrix.png', dpi=200)
    print("[OK] Matrice sauvegardee")
    plt.close()

# =============================================================================
# ÉTAPE 8: VALIDATION CROISÉE
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 8: VALIDATION CROISÉE (K-FOLD=5)")
print("="*80)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(log_reg, X_scaled, y, cv=kfold, scoring='accuracy')

print(f"\n[OK] Scores par fold:")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i}: {score:.4f}")

print(f"\nResume CV:")
print(f"   Moyenne: {cv_scores.mean():.4f}")
print(f"   Std:     {cv_scores.std():.4f}")
print(f"   Min:     {cv_scores.min():.4f}")
print(f"   Max:     {cv_scores.max():.4f}")

print(f"\nComparaison Test vs CV:")
print(f"   Accuracy Test: {acc:.4f}")
print(f"   Accuracy CV:   {cv_scores.mean():.4f}")
diff = abs(acc - cv_scores.mean())
print(f"   Ecart: {diff:.4f}")
if diff < 0.05:
    print(f"   [OK] Resultats stables (pas de surapprentissage)")

# =============================================================================
# ÉTAPE 9: COMPARAISON MODÈLES
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 9: COMPARAISON MODÈLES")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Perceptron': Perceptron(random_state=42, max_iter=1000),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
}

results = {}
print(f"\nEntrainement et evaluation:\n")

for name, clf in models.items():
    clf.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    cv = cross_val_score(clf, X_scaled, y, cv=kfold, scoring='accuracy')
    
    results[name] = {
        'test': test_acc,
        'cv_mean': cv.mean(),
        'cv_std': cv.std(),
        'model': clf
    }
    
    print(f"{name}")
    print(f"   Test: {test_acc:.4f}, CV: {cv.mean():.4f} +/- {cv.std():.4f}\n")

# Classement
ranked = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
print(f"CLASSEMENT (par CV):")
for rank, (name, res) in enumerate(ranked, 1):
    print(f"   {rank}. {name:<25} {res['cv_mean']:.4f}")

best_name = ranked[0][0]
best_model = ranked[0][1]['model']
best_score = ranked[0][1]['cv_mean']

print(f"\nMEILLEUR: {best_name} ({best_score:.4f})")

# Comparaison plot
if PLOT_AVAILABLE:
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(results.keys())
    means = [results[n]['cv_mean'] for n in names]
    stds = [results[n]['cv_std'] for n in names]
    
    x_pos = np.arange(len(names))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparaison Modeles', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'model_comparison.png', dpi=200)
    print("\n[OK] Graphique sauvegarde")
    plt.close()

# =============================================================================
# ÉTAPE 10: SAUVEGARDE
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 10: SAUVEGARDE")
print("="*80)

# Sauvegarder
model_path = Path(__file__).parent / 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n[OK] Modele: {model_path}")

scaler_path = Path(__file__).parent / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[OK] Scaler: {scaler_path}")

encoders_path = Path(__file__).parent / 'label_encoders.pkl'
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"[OK] Encoders: {encoders_path}")

# Tester le chargement
print(f"\nTest chargement...")
with open(model_path, 'rb') as f:
    loaded = pickle.load(f)

pred_loaded = loaded.predict(X_test)
acc_loaded = accuracy_score(y_test, pred_loaded)
print(f"[OK] Modele charge! Accuracy: {acc_loaded:.4f}")

# =============================================================================
# RAPPORT FINAL
# =============================================================================
print("\n" + "="*80)
print("RAPPORT FINAL")
print("="*80)

rapport = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║          RAPPORT FINAL - CLASSIFICATION D'ASSURANCE AUTOMOBILE            ║
╚══════════════════════════════════════════════════════════════════════════╝

1. OBJECTIF
   Prédire si un client fera une demande d'indemnisation (binaire: 0/1)

2. DONNÉES
   ├─ Total: {len(df):,} clients
   ├─ Features: {X.shape[1]}
   ├─ Classes: 0 ({sum(y==0):,} | {sum(y==0)/len(df)*100:.1f}%)
   │           1 ({sum(y==1):,} | {sum(y==1)/len(df)*100:.1f}%)
   └─ Train/Test: {X_train.shape[0]} / {X_test.shape[0]}

3. PRÉPARATION
   ├─ Suppression: id, postal_code
   ├─ Encodage: {len(label_encoders)} colonnes catégorielles
   └─ Normalisation: StandardScaler

4. VARIABLES PERTINENTES (Corrélation avec outcome)
   ├─ 1. {outcome_corr.index[1]}: {outcome_corr.iloc[1]:.4f}
   ├─ 2. {outcome_corr.index[2]}: {outcome_corr.iloc[2]:.4f}
   ├─ 3. {outcome_corr.index[3]}: {outcome_corr.iloc[3]:.4f}
   ├─ 4. {outcome_corr.index[4]}: {outcome_corr.iloc[4]:.4f}
   └─ 5. {outcome_corr.index[5]}: {outcome_corr.iloc[5]:.4f}

5. MEILLEUR MODÈLE: {best_name}
   ├─ Accuracy (Test): {acc:.4f}
   ├─ Accuracy (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
   ├─ Precision: {prec:.4f}
   ├─ Recall: {recall_val:.4f}
   └─ F1-Score: {f1:.4f}

6. INTERPRÉTATION
   ├─ {acc*100:.1f}% des prédictions sont correctes
   ├─ {prec*100:.1f}% des cas prédits positifs sont vrais positifs
   └─ {recall_val*100:.1f}% des cas positifs réels sont détectés

7. RÉGRESSION LOGISTIQUE - EXPLICATION
   ├─ Hypothèse:
   │  log(p(y=1|x)/p(y=0|x)) = w₀ + Σ(wᵢxᵢ)
   │
   ├─ Minimisation:
   │  • Algorithme: Descente de gradient ou Newton-Raphson
   │  • Fonction de coût: Maximum de vraisemblance
   │
   └─ Apprentissage:
      • Paramètres estimés: Coefficients (w) et intercept (w₀)
      • Méthode: Optimisation itérative

8. TOUS LES MODÈLES (Classement par Accuracy CV)
"""

for rank, (name, res) in enumerate(ranked, 1):
    rapport += f"   {rank}. {name:<28} {res['cv_mean']:.4f}\n"

rapport += f"""
9. FICHIERS GÉNÉRÉS
   ├─ car_insurance_analysis.py - Script principal
   ├─ best_model.pkl - Modèle entraîné
   ├─ scaler.pkl - Normaliseur
   ├─ label_encoders.pkl - Encodeurs
   ├─ histogrammes_variables.png
   ├─ correlation_matrix.png
   ├─ confusion_matrix.png
   ├─ model_comparison.png
   └─ rapport_final.txt - Ce rapport

╚══════════════════════════════════════════════════════════════════════════╝
"""

print(rapport)

# Sauvegarder rapport
rapport_path = Path(__file__).parent / 'rapport_final.txt'
with open(rapport_path, 'w', encoding='utf-8') as f:
    f.write(rapport)
print(f"\n[OK] Rapport: {rapport_path}")

print("\n" + "="*80)
print("PROJET TERMINE!")
print("="*80 + "\n")

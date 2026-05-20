"""
=================================================================================
PROJET DE CLASSIFICATION SUPERVISÉE - PRÉDICTION DE RÉCLAMATIONS D'ASSURANCE
=================================================================================

Objectif: Construire un modèle de machine learning pour prédire si un client
d'une compagnie d'assurance automobile fera une demande d'indemnisation.

Structure du projet:
1. Importation des données
2. Examen exploratoire des données (EDA)
3. Préparation des données
4. Analyse des corrélations
5. Division train/test
6. Entraînement des modèles
7. Évaluation et comparaison
8. Sauvegarde du meilleur modèle
=================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from pathlib import Path

# Imports scikit-learn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                            recall_score, f1_score, classification_report)

warnings.filterwarnings('ignore')

# Configuration matplotlib
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
sns.set_palette("husl")

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
    print(f"❌ Erreur: Le fichier car_insurance.csv n'existe pas dans {possible_paths}.")
    exit(1)

# Charger le fichier CSV
df = pd.read_csv(csv_path)

print(f"\n✓ Fichier importé avec succès!")
print(f"  - Dimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(f"\n📋 Aperçu des premières lignes:")
print(df.head())

# Vérification explicite des données manquantes (Étape 4)
print("\n❓ Données manquantes (Étape 4):")
print(df.isna().sum())

# Traitement des données manquantes (Étape 5 - Solution à privilégier : fillna)
df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
df['annual_mileage'] = df['annual_mileage'].fillna(df['annual_mileage'].median())
print(f"\n✓ Données manquantes imputées par la médiane (Étape 5).")

# Traitement des données aberrantes (Étape 5)
# On remplace les valeurs de speeding_violations > 50 par la médiane
median_speeding = df['speeding_violations'].median()
outliers_count = len(df[df['speeding_violations'] > 50])
df.loc[df['speeding_violations'] > 50, 'speeding_violations'] = median_speeding
print(f"✓ {outliers_count} valeurs aberrantes dans speeding_violations remplacées par la médiane (Étape 5).")

print(f"\n📊 Informations sur le DataFrame:")
print(df.info())
print(f"\n📈 Statistiques descriptives:")
print(df.describe())

# Identifier la variable de sortie
print(f"\n🎯 OBJECTIF DU PROBLÈME:")
print(f"   Prédire si un client fera une réclamation d'assurance (outcome: 0/1)")
print(f"   - Classe 0: Pas de réclamation (No claim)")
print(f"   - Classe 1: Réclamation (Made a claim)")
print(f"\n   Distribution des classes:")
print(df['outcome'].value_counts())
print(f"   Proportion: \n{df['outcome'].value_counts(normalize=True)}")

# =============================================================================
# ÉTAPE 2: EXAMEN DES DONNÉES
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 2: EXAMEN EXPLORATOIRE DES DONNÉES")
print("="*80)

# Identifier les types de colonnes
print(f"\n📝 Types de données:")
print(f"   - Colonnes numériques: {df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"   - Colonnes catégorielles: {df.select_dtypes(include=['object']).columns.tolist()}")

# Statistiques pour les variables numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\n📊 Analyse des variables numériques:")
for col in numeric_cols:
    print(f"   {col}: min={df[col].min()}, max={df[col].max()}, "
          f"mean={df[col].mean():.2f}, std={df[col].std():.2f}")

# Créer les histogrammes des variables numériques
print(f"\n📈 Création des histogrammes...")
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('Distribution des variables numériques', fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    if idx < len(axes):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col}')
        axes[idx].set_xlabel('Valeur')
        axes[idx].set_ylabel('Fréquence')

# Masquer les subplots inutilisés
for idx in range(len(numeric_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'histogrammes_variables.png', dpi=300, bbox_inches='tight')
print("   ✓ Histogrammes sauvegardés")
plt.close()

# Observations sur les données
print(f"\n🔍 OBSERVATIONS:")
print(f"   ✓ Les variables numériques couvrent différentes échelles")
print(f"   ✓ Certaines variables sont déjà encodées (0/1)")
print(f"   ✓ La variable 'postal_code' n'est probablement pas utile pour la prédiction")
print(f"   ✓ Les variables 'married' et 'children' peuvent contenir du texte")

# =============================================================================
# ÉTAPE 3: PRÉPARATION DES DONNÉES
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 3: PRÉPARATION DES DONNÉES")
print("="*80)

# Créer une copie du DataFrame
df_prepared = df.copy()

# Supprimer les colonnes non utiles
print(f"\n🗑️  Suppression des colonnes non pertinentes:")
cols_to_drop = ['id', 'postal_code']
df_prepared = df_prepared.drop(columns=cols_to_drop)
print(f"   ✓ Colonnes supprimées: {cols_to_drop}")

# Identifier les colonnes catégorielles
categorical_cols = df_prepared.select_dtypes(include=['object']).columns.tolist()
print(f"\n🏷️  Colonnes catégorielles à encoder: {categorical_cols}")

# Encoder les variables catégorielles avec LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_prepared[col] = le.fit_transform(df_prepared[col].astype(str))
    label_encoders[col] = le
    print(f"   ✓ {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Vérifier la structure après encodage
print(f"\n✓ Toutes les variables sont maintenant numériques")
print(f"   Shape: {df_prepared.shape}")

# Séparer X (features) et y (target)
X = df_prepared.drop(columns=['outcome'])
y = df_prepared['outcome']

print(f"\n✓ Séparation X/y:")
print(f"   - X (features): shape {X.shape}")
print(f"   - y (target): shape {y.shape}")

# Normalisation avec StandardScaler
print(f"\n📊 Normalisation des données avec StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"   ✓ Données normalisées")
print(f"   ✓ Moyenne: {X_scaled.mean().mean():.6f} (≈ 0)")
print(f"   ✓ Écart-type: {X_scaled.std().mean():.6f} (≈ 1)")

# Vérifier les données préparées
print(f"\n✅ Aperçu des données préparées et normalisées:")
print(X_scaled.head())

# =============================================================================
# ÉTAPE 4: RECHERCHE DE CORRÉLATIONS
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 4: RECHERCHE DE CORRÉLATIONS")
print("="*80)

# Ajouter la variable target pour l'analyse de corrélation
df_corr = X_scaled.copy()
df_corr['outcome'] = y.values

# Calculer la matrice de corrélation
correlation_matrix = df_corr.corr()

# Afficher les corrélations avec la variable target
print(f"\n🔗 Corrélations avec la variable de sortie (outcome):")
outcome_corr = correlation_matrix['outcome'].sort_values(ascending=False)
print(outcome_corr)

print(f"\n📊 INTERPRÉTATION DU COEFFICIENT DE CORRÉLATION:")
print(f"   - Un coefficient proche de 1 indique une corrélation positive forte")
print(f"   - Un coefficient proche de -1 indique une corrélation négative forte")
print(f"   - Un coefficient proche de 0 indique une corrélation faible")

print(f"\n🎯 Variables les plus pertinentes pour la classification:")
top_corr = outcome_corr[1:6]  # Exclure outcome lui-même
for var, corr in top_corr.items():
    print(f"   - {var}: {corr:.4f}")

# Créer une heatmap de la matrice de corrélation
print(f"\n📈 Création de la matrice de corrélation...")
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax, cbar_kws={'label': 'Coefficient de corrélation'})
plt.title('Matrice de Corrélation - Toutes les variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Matrice de corrélation sauvegardée")
plt.close()

# Scatter plot pour les variables les plus importantes
print(f"\n📊 Création des scatter plots pour les variables principales...")
top_features = ['past_accidents', 'speeding_violations', 'duis', 'annual_mileage']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Scatter plots - Variables principales vs outcome', fontsize=14, fontweight='bold')

for idx, feature in enumerate(top_features):
    ax = axes[idx // 2, idx % 2]
    colors = ['blue', 'red']
    for outcome_val in [0, 1]:
        mask = y == outcome_val
        ax.scatter(df_corr[mask][feature], df_corr[mask]['outcome'] + np.random.normal(0, 0.02, mask.sum()),
                  alpha=0.5, label=f'Outcome={outcome_val}', color=colors[outcome_val], s=50)
    ax.set_xlabel(feature)
    ax.set_ylabel('outcome')
    ax.set_title(f'{feature}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'scatter_plots.png', dpi=300, bbox_inches='tight')
print("   ✓ Scatter plots sauvegardés")
plt.close()

# =============================================================================
# ÉTAPE 5: EXTRACTION DES JEUX D'APPRENTISSAGE ET DE TEST
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 5: EXTRACTION DES JEUX D'APPRENTISSAGE ET DE TEST")
print("="*80)

# Diviser les données 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n✓ Division des données:")
print(f"   - Jeu d'apprentissage: {X_train.shape[0]} échantillons ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"   - Jeu de test: {X_test.shape[0]} échantillons ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"\n   Distribution des classes en train:")
print(f"   {y_train.value_counts()}")
print(f"\n   Distribution des classes en test:")
print(f"   {y_test.value_counts()}")

# =============================================================================
# ÉTAPE 6: ENTRAÎNEMENT DU MODÈLE PRINCIPAL (RÉGRESSION LOGISTIQUE)
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 6: ENTRAÎNEMENT - RÉGRESSION LOGISTIQUE")
print("="*80)

print(f"\n🤖 Entraînement du modèle LogisticRegression...")
log_reg = LogisticRegression(random_state=42, max_iter=1000, verbose=0)
log_reg.fit(X_train, y_train)
print(f"   ✓ Modèle entraîné avec succès!")

print(f"\n📝 Compréhension du modèle LogisticRegression:")
print(f"   - Hypothèse: La fonction logit log(p(y=1|x)/p(y=0|x)) est linéaire en x")
print(f"   - Minimisation: L'algorithme utilise la descente de gradient ou Newton-Raphson")
print(f"   - Apprentissage: Les coefficients (poids) et l'intercept sont estimés")
print(f"\n   Coefficients appris:")
for feature, coef in zip(X_train.columns, log_reg.coef_[0]):
    print(f"   - {feature}: {coef:.4f}")
print(f"   Intercept: {log_reg.intercept_[0]:.4f}")

# =============================================================================
# ÉTAPE 7: ÉVALUATION DU MODÈLE
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 7: ÉVALUATION DU MODÈLE")
print("="*80)

# Prédictions
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)

# Afficher quelques prédictions
print(f"\n🔮 Exemples de prédictions (premiers 10 échantillons):")
print(f"{'Index':<6} {'Prédiction':<12} {'Probabilité':<20} {'Réel':<6}")
print("-" * 50)
for i in range(min(10, len(y_test))):
    pred_prob = max(y_pred_proba[i])
    print(f"{i:<6} {y_pred[i]:<12} {pred_prob:<20.4f} {y_test.iloc[i]:<6}")

# Calcul des métriques
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n📊 MÉTRIQUES DE PERFORMANCE:")
print(f"   - Accuracy: {accuracy:.4f}")
print(f"     (Proportion des prédictions correctes parmi tous les cas)")
print(f"   - Precision: {precision:.4f}")
print(f"     (Parmi les cas prédits positifs, combien sont vrais positifs?)")
print(f"   - Recall: {recall:.4f}")
print(f"     (Parmi les cas réels positifs, combien sont détectés?)")
print(f"   - F1-score: {f1:.4f}")
print(f"     (Moyenne harmonique entre precision et recall)")

# Matrice de confusion
print(f"\n📋 Matrice de Confusion:")
print(f"                Prédiction")
print(f"              Négative  Positive")
print(f"Réel Négative  {conf_matrix[0,0]:<9} {conf_matrix[0,1]:<9}")
print(f"     Positive  {conf_matrix[1,0]:<9} {conf_matrix[1,1]:<9}")

print(f"\n📈 Rapport de classification:")
print(classification_report(y_test, y_pred, target_names=['No Claim', 'Made Claim']))

# Visualiser la matrice de confusion
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Claim', 'Made Claim'],
            yticklabels=['No Claim', 'Made Claim'])
plt.title('Matrice de Confusion - Régression Logistique', fontweight='bold')
plt.ylabel('Valeur Réelle')
plt.xlabel('Valeur Prédite')
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Matrice de confusion sauvegardée")
plt.close()

# =============================================================================
# ÉTAPE 8: AMÉLIORATION - VALIDATION CROISÉE
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 8: AMÉLIORATION - VALIDATION CROISÉE (K-FOLD)")
print("="*80)

# K-Fold validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(log_reg, X_scaled, y, cv=kfold, scoring='accuracy')

print(f"\n✓ Validation croisée avec K=5 effectuée")
print(f"\n📊 Résultats par fold:")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i}: {score:.4f}")

print(f"\n📈 RÉSULTATS FINAUX DE LA VALIDATION CROISÉE:")
print(f"   - Moyenne: {cv_scores.mean():.4f}")
print(f"   - Écart-type: {cv_scores.std():.4f}")
print(f"   - Min: {cv_scores.min():.4f}")
print(f"   - Max: {cv_scores.max():.4f}")

print(f"\n🔄 COMPARAISON:")
print(f"   - Accuracy sur test: {accuracy:.4f}")
print(f"   - Accuracy CV (moyenne): {cv_scores.mean():.4f}")
print(f"   - Différence: {abs(accuracy - cv_scores.mean()):.4f}")
if abs(accuracy - cv_scores.mean()) < 0.05:
    print(f"   ✓ Les résultats sont stables (bon signe - pas de surapprentissage)")
else:
    print(f"   ⚠️  Écart notable entre test et CV")

# =============================================================================
# ÉTAPE 9: COMPARAISON AVEC D'AUTRES ALGORITHMES
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 9: COMPARAISON AVEC D'AUTRES ALGORITHMES")
print("="*80)

# Définir les classifieurs
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Perceptron': Perceptron(random_state=42, max_iter=1000),
    'K-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5),
    'K-Nearest Neighbors (k=7)': KNeighborsClassifier(n_neighbors=7),
    'K-Nearest Neighbors (k=3)': KNeighborsClassifier(n_neighbors=3),
}

results = {}
print(f"\n🤖 Entraînement et évaluation des modèles...\n")

for name, clf in classifiers.items():
    # Entraînement
    clf.fit(X_train, y_train)
    
    # Évaluation sur test
    y_pred_test = clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    # Validation croisée
    cv_scores_model = cross_val_score(clf, X_scaled, y, cv=kfold, scoring='accuracy')
    
    results[name] = {
        'accuracy_test': accuracy_test,
        'accuracy_cv': cv_scores_model.mean(),
        'std_cv': cv_scores_model.std(),
        'model': clf
    }
    
    print(f"📊 {name}")
    print(f"   - Accuracy (Test): {accuracy_test:.4f}")
    print(f"   - Accuracy (CV): {cv_scores_model.mean():.4f} ± {cv_scores_model.std():.4f}")
    print()

# Classement des modèles
print(f"\n🏆 CLASSEMENT DES MODÈLES (par Accuracy CV):")
sorted_results = sorted(results.items(), 
                       key=lambda x: x[1]['accuracy_cv'], 
                       reverse=True)

for rank, (name, metrics) in enumerate(sorted_results, 1):
    print(f"   {rank}. {name}: {metrics['accuracy_cv']:.4f}")

best_model_name = sorted_results[0][0]
best_model = sorted_results[0][1]['model']

print(f"\n✨ MEILLEUR MODÈLE: {best_model_name}")
print(f"   Accuracy: {sorted_results[0][1]['accuracy_cv']:.4f}")

# Visualiser la comparaison
fig, ax = plt.subplots(figsize=(12, 6))
names = list(results.keys())
cv_means = [results[name]['accuracy_cv'] for name in names]
cv_stds = [results[name]['std_cv'] for name in names]

x_pos = np.arange(len(names))
ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='steelblue')
ax.set_ylabel('Accuracy')
ax.set_title('Comparaison des Modèles - Validation Croisée', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Ajouter les valeurs sur les barres
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("\n   ✓ Graphique de comparaison sauvegardé")
plt.close()

# =============================================================================
# ÉTAPE 10: SAUVEGARDE DU MODÈLE
# =============================================================================
print("\n" + "="*80)
print("ÉTAPE 10: SAUVEGARDE DU MODÈLE")
print("="*80)

# Sauvegarder le meilleur modèle
model_path = Path(__file__).parent / 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✓ Meilleur modèle sauvegardé: {model_path}")

# Sauvegarder le scaler
scaler_path = Path(__file__).parent / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler sauvegardé: {scaler_path}")

# Sauvegarder les encodeurs
encoders_path = Path(__file__).parent / 'label_encoders.pkl'
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"✓ Label encoders sauvegardés: {encoders_path}")

# Charger et tester le modèle sauvegardé
print(f"\n🔄 Test du chargement du modèle...")
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

y_pred_loaded = loaded_model.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"✓ Modèle chargé avec succès!")
print(f"  Accuracy: {accuracy_loaded:.4f} (identique au modèle original)")

# =============================================================================
# RÉSUMÉ ET RAPPORT FINAL
# =============================================================================
print("\n" + "="*80)
print("RAPPORT FINAL - RÉSUMÉ DU PROJET")
print("="*80)

rapport = f"""
OBJECTIF DU PROJET:
  Prédire si un client d'assurance automobile fera une demande d'indemnisation
  (classification binaire: 0 = pas de réclamation, 1 = réclamation)

DONNÉES:
  - Nombre total d'échantillons: {len(df)}
  - Nombre de features: {X.shape[1]}
  - Variables numériques: {len(numeric_cols)}
  - Variables catégorielles encodées: {len(categorical_cols)}
  - Distribution des classes: {sum(y==0)} (classe 0) vs {sum(y==1)} (classe 1)

PRÉPARATION DES DONNÉES:
  ✓ Colonnes supprimées: {', '.join(cols_to_drop)}
  ✓ Variables catégorielles encodées avec LabelEncoder
  ✓ Données normalisées avec StandardScaler
  ✓ Division: 70% train ({X_train.shape[0]}), 30% test ({X_test.shape[0]})

VARIABLES LES PLUS PERTINENTES:
  1. {outcome_corr.index[1]}: {outcome_corr.iloc[1]:.4f}
  2. {outcome_corr.index[2]}: {outcome_corr.iloc[2]:.4f}
  3. {outcome_corr.index[3]}: {outcome_corr.iloc[3]:.4f}

RÉSULTATS DU MEILLEUR MODÈLE: {best_model_name}
  - Accuracy (Test): {results[best_model_name]['accuracy_test']:.4f}
  - Accuracy (Validation Croisée): {results[best_model_name]['accuracy_cv']:.4f} ± {results[best_model_name]['std_cv']:.4f}
  - Precision: {precision:.4f}
  - Recall: {recall:.4f}
  - F1-Score: {f1:.4f}

INTERPRÉTATION:
  - Le modèle prédit correctement {accuracy*100:.1f}% des cas
  - Sur les cas positifs réels, il en détecte {recall*100:.1f}%
  - Parmi les cas prédits positifs, {precision*100:.1f}% sont vrais positifs
  
MODÈLES TESTÉS:
  1. Logistic Regression: {results['Logistic Regression']['accuracy_cv']:.4f}
  2. Perceptron: {results['Perceptron']['accuracy_cv']:.4f}
  3. KNeighborsClassifier (k=3): {results['K-Nearest Neighbors (k=3)']['accuracy_cv']:.4f}
  4. KNeighborsClassifier (k=5): {results['K-Nearest Neighbors (k=5)']['accuracy_cv']:.4f}
  5. KNeighborsClassifier (k=7): {results['K-Nearest Neighbors (k=7)']['accuracy_cv']:.4f}

FICHIERS GÉNÉRÉS:
  ✓ car_insurance_analysis.py - Script d'analyse complet
  ✓ best_model.pkl - Modèle sauvegardé
  ✓ scaler.pkl - StandardScaler
  ✓ label_encoders.pkl - Encodeurs pour variables catégorielles
  ✓ histogrammes_variables.png - Distributions des variables
  ✓ correlation_matrix.png - Matrice de corrélation
  ✓ scatter_plots.png - Graphiques de dispersion
  ✓ confusion_matrix.png - Matrice de confusion
  ✓ model_comparison.png - Comparaison des modèles
  ✓ rapport_final.txt - Ce rapport
"""

print(rapport)

# Sauvegarder le rapport dans un fichier
rapport_path = Path(__file__).parent / 'rapport_final.txt'
with open(rapport_path, 'w', encoding='utf-8') as f:
    f.write(rapport)
print(f"\n✓ Rapport sauvegardé: {rapport_path}")

print("\n" + "="*80)
print("✨ PROJET TERMINÉ AVEC SUCCÈS!")
print("="*80 + "\n")

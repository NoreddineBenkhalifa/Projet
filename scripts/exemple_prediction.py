"""
Exemple: Utiliser le modèle entraîné pour faire des prédictions
Ce script montre comment charger et utiliser le modèle sauvegardé
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_model_components():
    """Charger le modèle, scaler et encodeurs"""
    model_path = Path(__file__).parent / 'best_model.pkl'
    scaler_path = Path(__file__).parent / 'scaler.pkl'
    encoders_path = Path(__file__).parent / 'label_encoders.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    return model, scaler, encoders

def prepare_new_data(new_data_df, scaler, encoders):
    """Préparer les nouvelles données pour prédiction"""
    df = new_data_df.copy()
    
    # Supprimer colonnes inutiles si présentes
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'postal_code' in df.columns:
        df = df.drop(columns=['postal_code'])
    
    # Encoder variables catégorielles
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))
    
    # Normaliser
    X_scaled = scaler.transform(df)
    
    return X_scaled

def predict_claim(model, X_new):
    """Faire des prédictions"""
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)
    
    return predictions, probabilities

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("EXEMPLE: UTILISATION DU MODÈLE SAUVEGARDÉ")
    print("="*80)
    
    # Charger les composants
    print("\n📂 Chargement du modèle...")
    try:
        model, scaler, encoders = load_model_components()
        print("✓ Modèle chargé avec succès!")
    except FileNotFoundError as e:
        print(f"❌ Erreur: {e}")
        print("Assurez-vous d'avoir exécuté analyse_complete.py d'abord")
        exit(1)
    
    # Créer des exemples de nouveaux clients
    print("\n👤 Création d'exemples de clients à prédire...\n")
    
    new_clients = pd.DataFrame({
        'age': [1, 3],                              # 0=16-25, 1=26-39, 2=40-64, 3=65+
        'gender': [0, 1],                            # 0=Femme, 1=Homme
        'driving_experience': ['0-9y', '30y+'],
        'education': ['high school', 'university'],
        'income': ['working class', 'upper class'],
        'credit_score': [0.5, 0.8],
        'vehicle_ownership': [1.0, 1.0],
        'vehicle_year': ['after 2015', 'before 2015'],
        'married': [0.0, 1.0],
        'children': [0.0, 2.0],
        'annual_mileage': [10000.0, 15000.0],
        'vehicle_type': ['sedan', 'sedan'],
        'speeding_violations': [0, 5],              # ATTENTION: haute corrélation!
        'duis': [0, 1],
        'past_accidents': [0, 3]                    # ATTENTION: haute corrélation!
    })
    
    print("Clients à prédire:")
    print(new_clients)
    
    # Préparer les données
    print("\n🔄 Préparation des données...")
    X_new = prepare_new_data(new_clients, scaler, encoders)
    print("✓ Données préparées et normalisées")
    
    # Faire les prédictions
    print("\n🔮 Prédictions...")
    predictions, probabilities = predict_claim(model, X_new)
    
    # Afficher les résultats
    print("\n📊 RÉSULTATS DES PRÉDICTIONS:")
    print("="*80)
    
    for i in range(len(new_clients)):
        pred = predictions[i]
        prob_no_claim = probabilities[i][0]
        prob_claim = probabilities[i][1]
        
        prediction_text = "RÉCLAMATION" if pred == 1 else "PAS DE RÉCLAMATION"
        confidence = max(prob_no_claim, prob_claim)
        
        print(f"\nClient {i+1}:")
        print(f"  └─ Prédiction: {prediction_text}")
        print(f"  └─ Confiance: {confidence*100:.1f}%")
        print(f"  └─ Probabilité pas de réclamation: {prob_no_claim:.4f}")
        print(f"  └─ Probabilité réclamation: {prob_claim:.4f}")
    
    print("\n" + "="*80)
    
    # Notes importantes
    print("\n📝 NOTES IMPORTANTES:")
    print("  • Les prédictions dépendent fortement de:")
    print("    - past_accidents (accidents antérieurs)")
    print("    - speeding_violations (infractions vitesse)")
    print("    - duis (conduites sous influence)")
    print("  • Ces variables ont une corrélation positive avec les réclamations")
    print("  • Un client avec beaucoup de violations aura une probabilité plus haute")
    
    print("\n✨ Exemple complété!\n")

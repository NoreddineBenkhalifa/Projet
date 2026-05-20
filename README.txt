PROJET DE CLASSIFICATION SUPERVISÉE - ASSURANCE AUTOMOBILE

FICHIERS PRINCIPAUX :
1. Analyse_Prediction_Assurance_Auto.ipynb - Notebook principal (Sujet complet)
2. scripts/analyse_complete.py - Code principal (commenté)
3. scripts/car_insurance_analysis.py - Code détaillé
4. scripts/rapport_final.txt - Résultats (généré après exécution)

COMMENT EXÉCUTER :
1. Installer : python -m pip install -r requirements.txt
2. Exécuter : python scripts/analyse_complete.py
3. Attendre quelques secondes
4. Consulter scripts/rapport_final.txt et les graphiques PNG

DONNÉES GÉNÉRÉES (dans scripts/) :
- rapport_final.txt - Résultats complets
- histogrammes_variables.png - Distributions
- correlation_matrix.png - Corrélations
- confusion_matrix.png - Performance
- model_comparison.png - Comparaison modèles
- best_model.pkl - Modèle entraîné
- scaler.pkl - Normalisation
- label_encoders.pkl - Encodages

STRUCTURE DE LA DIAPO :
1. Titre
2. Objectif (prédiction réclamations)
3. Données (10,000 clients, 18 variables)
4. Exploration (graphiques)
5. Préparation (nettoyage, normalisation, gestion des valeurs aberrantes)
6. Modèles testés (Régression Logistique, Perceptron, KNN)
7. Meilleur modèle (Logistic Regression ~84%)
8. Conclusion

RÉSULTATS OBTENUS :
- Meilleur modèle : Logistic Regression
- Accuracy : ~84%
- Précision : ~74%
- Rappel : ~74%
- F1-Score : ~74%
- Cross-Validation 5-Fold : ~84%

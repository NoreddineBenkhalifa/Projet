PROJET DE CLASSIFICATION SUPERVISEE - ASSURANCE AUTOMOBILE

FICHIERS A RENDRE:
1. analyse_complete.py - Code principal (commente)
2. car_insurance_analysis.py - Code detaille (optionnel)
3. rapport_final.txt - Resultats (genere apres execution)
4. diapo.pptx - Presentation (a creer)

COMMENT EXECUTER:
1. Installer: python -m pip install -r requirements.txt
2. Executer: python analyse_complete.py
3. Attendre 2-3 minutes
4. Consulter rapport_final.txt et les graphiques PNG

DONNEES GENEREES:
- rapport_final.txt - Resultats complets
- histogrammes_variables.png - Distributions
- correlation_matrix.png - Correlations
- confusion_matrix.png - Performance
- model_comparison.png - Comparaison modeles
- best_model.pkl - Modele entraîne
- scaler.pkl - Normalisation
- label_encoders.pkl - Encodages

STRUCTURE DE LA DIAPO:
1. Titre
2. Objectif (prediction reclamations)
3. Donnees (12,960 clients, 17 variables)
4. Exploration (graphiques)
5. Preparation (nettoyage, normalisation)
6. Modeles testes (5 modeles)
7. Meilleur modele (Logistic Regression 78%)
8. Conclusion

RESULTATS ATTENDUS:
- Meilleur modele: Logistic Regression
- Accuracy: 78%
- Precision: 79%
- Recall: 73%
- F1-Score: 75%
- Cross-Validation 5-Fold: 77%

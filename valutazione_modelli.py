import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Configurazione per la riproducibilità dei risultati
RANDOM_STATE = 42

def main():
    # 1. Caricamento del Dataset
    try:
        df = pd.read_csv('Train Dataset.csv')
    except FileNotFoundError:
        print("Errore: Il file 'Train Dataset.csv' non è stato trovato nella directory corrente.")
        return

    # 2. Pre-processing e Pulizia Dati
    # Rimozione della colonna indice 'Unnamed: 0' in quanto non porta contenuto informativo
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Separazione delle feature (X) e della variabile target (y)
    X = df.drop(columns=['Crop'])
    y = df['Crop']

    # Encoding della variabile target (conversione da etichette testuali a valori numerici)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. Definizione dei Modelli di Apprendimento
    # Si confrontano un modello interpretabile (Decision Tree) e un modello di ensemble (Random Forest)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }

    # 4. Configurazione della Valutazione Sperimentale
    # Adozione della 10-Fold Cross-Validation per ottenere una stima robusta delle prestazioni
    # e minimizzare la varianza dovuta alla selezione del set di training/test.
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    print("Avvio della valutazione comparativa dei modelli (10-Fold Cross-Validation)...")
    print("-" * 70)

    # 5. Esecuzione dei Test
    for name, model in models.items():
        # Calcolo dell'accuratezza per ogni fold
        scores = cross_val_score(model, X, y_encoded, cv=kf, scoring='accuracy')
        
        # Stampa dei risultati statistici: Media e Deviazione Standard
        print(f"Modello: {name}")
        print(f"   Accuratezza Media:      {scores.mean():.4f}")
        print(f"   Deviazione Standard:    {scores.std():.4f}")
        print("-" * 70)

    # 6. Salvataggio del dataset pre-processato
    # Il dataset pulito verrà utilizzato successivamente per l'estrazione della Knowledge Base
    df.to_csv('Train_Dataset_Clean.csv', index=False)
    print("Pre-processing completato. Il dataset pulito è stato salvato come 'Train_Dataset_Clean.csv'.")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Creazione cartella output se non esiste
OUTPUT_DIR = 'grafici_per_relazione'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    print("--- VALUTAZIONE COMPARATIVA MODELLI ---")
    
    # 1. Caricamento Dati
    try:
        df = pd.read_csv('Train_Dataset_Clean.csv')
    except FileNotFoundError:
        print("[ERRORE] File 'Train_Dataset_Clean.csv' non trovato. Assicurati di averlo nella cartella.")
        return

    X = df.drop(columns=['Crop'])
    y = df['Crop']
    
    # 2. Encoding (Corretto: usiamo un nome univoco)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Qui era l'errore: ora la chiamo y_encoded coerentemente
    
    # 3. Scaling per SVM (necessario per modelli basati su distanze)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Definizione Modelli
    models = [
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('Naive Bayes', GaussianNB()),
        ('SVM (RBF)', SVC(kernel='rbf', probability=True))
    ]
    
    results = []
    names = []
    
    print(f"{'Modello':<20} | {'Accuratezza':<10} | {'Std Dev':<10}")
    print("-" * 45)
    
    # 5. Cross Validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for name, model in models:
        # SVM richiede dati scalati, gli altri usano X originale (o scalato, va bene uguale per alberi)
        data_to_use = X_scaled if name == 'SVM (RBF)' else X
        
        # Qui usiamo 'y_encoded' che ora è definita correttamente sopra
        cv_results = cross_val_score(model, data_to_use, y_encoded, cv=kfold, scoring='accuracy')
        
        results.append(cv_results)
        names.append(name)
        
        print(f"{name:<20} | {cv_results.mean():.4f}     | {cv_results.std():.4f}")

    # 6. Generazione Grafico
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results, palette="Set3")
    plt.xticks(range(4), names)
    plt.title('Confronto Prestazioni Modelli (10-Fold CV)', fontsize=15)
    plt.ylabel('Accuratezza')
    plt.xlabel('Algoritmo')
    
    outfile = os.path.join(OUTPUT_DIR, '7_confronto_modelli.png')
    plt.savefig(outfile, dpi=300)
    plt.close() # Chiude la figura per liberare memoria
    print(f"\n[GRAPH] Grafico comparativo salvato in: {outfile}")

if __name__ == "__main__":
    main()
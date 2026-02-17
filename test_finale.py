import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import os

# --- CONFIGURAZIONE ---
TRAIN_FILE = 'Train_Dataset_Clean.csv'
TEST_FILE = 'test dataset.csv'
OUTPUT_DIR = 'grafici_per_relazione'  # Cartella di destinazione

# Assicuriamoci che la cartella esista
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    print("--- AVVIO VALUTAZIONE FINALE (HOLD-OUT TEST) ---")

    # 1. Caricamento dei Dataset
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
        
        print(f"[INFO] Dataset caricati.")
        print(f"       Training Set: {df_train.shape[0]} righe")
        print(f"       Test Set:     {df_test.shape[0]} righe")

    except FileNotFoundError as e:
        print(f"[ERROR] File non trovato: {e}")
        return

    # 2. Pre-processing
    for df in [df_train, df_test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    X_train = df_train.drop(columns=['Crop'])
    y_train = df_train['Crop']

    X_test = df_test.drop(columns=['Crop'])
    y_test = df_test['Crop']

    # 3. Encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Gestione classi test
    try:
        y_test_encoded = le.transform(y_test)
    except ValueError as e:
        print("[WARNING] Il Test Set contiene classi non viste nel Training Set.")
        return

    # 4. Addestramento
    print("[INFO] Addestramento del Decision Tree sul 100% del Training Set...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train_encoded)

    # 5. Predizione
    print("[INFO] Calcolo delle predizioni sul Test Set...")
    y_pred = model.predict(X_test)

    # 6. Metriche
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print("-" * 50)
    print(f"ACCURATEZZA FINALE (Test Set): {accuracy:.4f}")
    print("-" * 50)

    # --- SALVATAGGIO REPORT (TXT) ---
    report = classification_report(y_test_encoded, y_pred, target_names=le.classes_)
    
    report_path = os.path.join(OUTPUT_DIR, "report_classificazione_finale.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[INFO] Report salvato in '{report_path}'.")

    # --- SALVATAGGIO MATRICE DI CONFUSIONE (PNG) ---
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    
    plt.title('Matrice di Confusione - Valutazione Finale')
    plt.ylabel('Classe Reale (Verità)')
    plt.xlabel('Classe Predetta dal Modello')
    
    img_path = os.path.join(OUTPUT_DIR, 'matrice_confusione_finale.png')
    plt.savefig(img_path)
    plt.close() # Chiude la figura
    
    print(f"[INFO] Grafico salvato in '{img_path}'.")
    print("--- VALUTAZIONE COMPLETATA ---")

if __name__ == "__main__":
    main()
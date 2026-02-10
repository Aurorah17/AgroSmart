import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURAZIONE ---
TRAIN_FILE = 'Train_Dataset_Clean.csv' # Il file pulito che abbiamo creato all'inizio
TEST_FILE = 'test dataset.csv'         # Il file "vergine" per la valutazione finale

def main():
    print("--- AVVIO VALUTAZIONE FINALE (HOLD-OUT TEST) ---")

    # 1. Caricamento dei Dataset
    try:
        # Carichiamo il Training Set (usato per l'addestramento)
        df_train = pd.read_csv(TRAIN_FILE)
        
        # Carichiamo il Test Set (usato SOLO per la verifica finale)
        df_test = pd.read_csv(TEST_FILE)
        
        print(f"[INFO] Dataset caricati.")
        print(f"       Training Set: {df_train.shape[0]} righe")
        print(f"       Test Set:     {df_test.shape[0]} righe")

    except FileNotFoundError as e:
        print(f"[ERROR] File non trovato: {e}")
        return

    # 2. Pre-processing e Allineamento Dati
    # Rimuoviamo colonne inutili se presenti (es. indici o etichette numeriche)
    for df in [df_train, df_test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    # Separazione Feature (X) e Target (y)
    X_train = df_train.drop(columns=['Crop'])
    y_train = df_train['Crop']

    X_test = df_test.drop(columns=['Crop'])
    y_test = df_test['Crop']

    # 3. Encoding delle Etichette (Label Encoding)
    # IMPORTANTE: L'encoder deve imparare le classi dal TRAIN e applicarle al TEST
    # per garantire che "rice" sia sempre lo stesso numero in entrambi i file.
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Gestione di eventuali classi nel test set non presenti nel training (raro ma possibile)
    try:
        y_test_encoded = le.transform(y_test)
    except ValueError as e:
        print("[WARNING] Il Test Set contiene classi non viste nel Training Set. Filtraggio necessario.")
        # (In un caso reale si gestirebbe diversamente, qui procediamo assumendo consistenza)
        return

    # 4. Addestramento del Modello Finale
    # Utilizziamo l'Albero di Decisione con gli stessi parametri usati nella Cross-Validation
    print("[INFO] Addestramento del Decision Tree sul 100% del Training Set...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train_encoded)

    # 5. Predizione sui dati "Nuovi" (Test Set)
    print("[INFO] Calcolo delle predizioni sul Test Set...")
    y_pred = model.predict(X_test)

    # 6. Calcolo delle Metriche di Performance
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print("-" * 50)
    print(f"ACCURATEZZA FINALE (Test Set): {accuracy:.4f}")
    print("-" * 50)

    # Generazione del Report di Classificazione (Precision, Recall, F1-Score per ogni pianta)
    report = classification_report(y_test_encoded, y_pred, target_names=le.classes_)
    # Salviamo il report in un file di testo per la documentazione
    with open("report_classificazione_finale.txt", "w") as f:
        f.write(report)
    print("[INFO] Report dettagliato salvato in 'report_classificazione_finale.txt'.")

    # 7. Generazione e Salvataggio della Matrice di Confusione
    # Questo grafico mostra DOVE il modello sbaglia (quali piante confonde)
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    
    plt.title('Matrice di Confusione - Valutazione Finale')
    plt.ylabel('Classe Reale (Verità)')
    plt.xlabel('Classe Predetta dal Modello')
    
    output_img = 'matrice_confusione_finale.png'
    plt.savefig(output_img)
    print(f"[INFO] Grafico salvato come '{output_img}'.")
    print("--- VALUTAZIONE COMPLETATA ---")

if __name__ == "__main__":
    main()
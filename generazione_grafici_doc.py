import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx # Per il grafo bayesiano
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURAZIONE ---
FILE_DATI = 'Train_Dataset_Clean.csv'
CARTELLA_OUTPUT = 'grafici_per_relazione'

# Crea una sottocartella per tenere in ordine i file generati
if not os.path.exists(CARTELLA_OUTPUT):
    os.makedirs(CARTELLA_OUTPUT)
    print(f"[INFO] Creata cartella '{CARTELLA_OUTPUT}' per salvare le immagini.")

# Imposta lo stile grafico per renderli più professionali
sns.set_theme(style="whitegrid")

def carica_dati():
    """Helper per caricare e pulire i dati."""
    try:
        df = pd.read_csv(FILE_DATI)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    except FileNotFoundError:
        print(f"[ERRORE] Impossibile trovare '{FILE_DATI}'. Assicurati che sia nella cartella.")
        exit()

# ==============================================================================
# 1. GRAFICO DISTRIBUZIONE DELLE CLASSI (Dataset Balance)
# ==============================================================================
def plot_distribuzione_classi(df):
    print("--- Generazione Grafico 1: Distribuzione Classi ---")
    plt.figure(figsize=(14, 8))
    
    # Ordiniamo le barre dalla più frequente alla meno frequente per leggibilità
    ordine = df['Crop'].value_counts().index
    
    sns.countplot(data=df, x='Crop', palette='viridis', order=ordine)
    
    plt.title('Analisi del Dataset: Distribuzione delle Colture', fontsize=16)
    plt.xlabel('Coltura (Target)', fontsize=12)
    plt.ylabel('Numero di Campioni', fontsize=12)
    plt.xticks(rotation=90) # Ruota le etichette per leggerle meglio
    plt.tight_layout()
    
    nome_file = os.path.join(CARTELLA_OUTPUT, '1_distribuzione_classi.png')
    plt.savefig(nome_file, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {nome_file}")

# ==============================================================================
# 2. GRAFICO FEATURE IMPORTANCE (Importanza dei Parametri)
# ==============================================================================
def plot_feature_importance(df):
    print("\n--- Generazione Grafico 2: Feature Importance ---")
    X = df.drop(columns=['Crop'])
    y = df['Crop']
    
    # Serve l'encoding per addestrare il modello al volo
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Addestriamo un Decision Tree veloce per estrarre l'importanza
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y_enc)
    
    # Creiamo un DataFrame per facilitare il plotting
    importance_df = pd.DataFrame({
        'Parametro': X.columns,
        'Importanza (Gini)': model.feature_importances_
    }).sort_values(by='Importanza (Gini)', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importanza (Gini)', y='Parametro', palette='magma')
    
    plt.title('Machine Learning: Importanza dei Parametri (Feature Importance)', fontsize=16)
    plt.xlabel('Importanza Relativa (Indice di Gini)', fontsize=12)
    plt.ylabel('Parametro Fisico-Chimico', fontsize=12)
    plt.tight_layout()
    
    nome_file = os.path.join(CARTELLA_OUTPUT, '2_feature_importance.png')
    plt.savefig(nome_file, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {nome_file}")

# ==============================================================================
# 3. LEARNING CURVE (Curva di Apprendimento)
# ==============================================================================
def plot_learning_curve_graph(df):
    print("\n--- Generazione Grafico 3: Learning Curve (Richiede alcuni secondi...) ---")
    X = df.drop(columns=['Crop'])
    y = df['Crop']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Usiamo il modello Decision Tree.
    # cv=5 indica una cross-validation a 5 strati per ogni punto della curva.
    train_sizes, train_scores, test_scores = learning_curve(
        DecisionTreeClassifier(random_state=42), 
        X, y_enc, 
        cv=5, 
        n_jobs=-1, # Usa tutti i core della CPU
        train_sizes=np.linspace(0.1, 1.0, 10), # 10 punti dal 10% al 100% dei dati
        scoring='accuracy'
    )

    # Calcolo media e deviazione standard per le bande ombreggiate
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    
    # Plot delle curve medie
    plt.plot(train_sizes, train_mean, 'o-', color="firebrick", label="Training Score (Addestramento)")
    plt.plot(train_sizes, test_mean, 'o-', color="seagreen", label="Cross-Validation Score (Validazione)")

    # Plot delle aree di varianza (ombreggiatura)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="firebrick")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="seagreen")

    plt.title("Analisi delle Prestazioni: Curva di Apprendimento (Learning Curve)", fontsize=16)
    plt.xlabel("Numero di campioni utilizzati per l'addestramento", fontsize=12)
    plt.ylabel("Accuratezza", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    
    nome_file = os.path.join(CARTELLA_OUTPUT, '3_learning_curve.png')
    plt.savefig(nome_file, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {nome_file}")

# ==============================================================================
# 4. VISUALIZZAZIONE GRAFO RETE BAYESIANA
# ==============================================================================
def plot_rete_bayesiana():
    print("\n--- Generazione Grafico 4: Struttura Rete Bayesiana ---")
    # Definiamo manualmente la struttura causale (la stessa usata in diagnosi_bayesiana.py)
    # Archi diretti: Causa -> Effetto
    archi = [
        ('Presenza_Malattia', 'Macchie_Fogliari'),
        ('Presenza_Malattia', 'Ingiallimento_Bordi'),
        ('Presenza_Malattia', 'Appassimento_Precoce')
    ]
    
    # Creazione del grafo diretto
    G = nx.DiGraph()
    G.add_edges_from(archi)

    plt.figure(figsize=(8, 6))
    
    # Algoritmo per disporre i nodi in modo che la "causa" stia in alto
    pos = nx.spring_layout(G, seed=42) 

    # Disegno dei nodi (cerchi)
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color='lightblue', edgecolors='navy')
    
    # Disegno delle etichette (testo dentro i nodi)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='black')
    
    # Disegno degli archi (frecce)
    nx.draw_networkx_edges(G, pos, edgelist=archi, edge_color='gray', arrows=True, arrowsize=25, width=2)

    plt.title("Struttura della Conoscenza Probabilistica (Rete Bayesiana)", fontsize=14)
    plt.axis('off') # Nasconde gli assi cartesiani che qui non servono
    plt.tight_layout()

    nome_file = os.path.join(CARTELLA_OUTPUT, '4_rete_bayesiana_grafo.png')
    plt.savefig(nome_file, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {nome_file}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    df_clean = carica_dati()
    
    # Esecuzione sequenziale delle funzioni di plotting
    plot_distribuzione_classi(df_clean)
    plot_feature_importance(df_clean)
    plot_learning_curve_graph(df_clean)
    plot_rete_bayesiana()
    
    print(f"\n--- TUTTI I GRAFICI GENERATI CON SUCCESSO NELLA CARTELLA '{CARTELLA_OUTPUT}' ---")
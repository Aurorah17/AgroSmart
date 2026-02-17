import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pyswip import Prolog
from pianificazione_drone import a_star_search, mappa_agricola, euristica
from diagnosi_bayesiana import DiagnosticaFitopatologica

# --- CONFIGURAZIONE SISTEMA ---
DATASET_PATH = 'Train_Dataset_Clean.csv'
KB_PATH = 'kb_agricola.pl'

class AgroSmartAI:
    def __init__(self):
        self.model_dt = None
        self.model_rf = None
        self.le = LabelEncoder()
        self.prolog = Prolog()
        self.diagnostica = DiagnosticaFitopatologica()
        
        print(f"[INIT] Caricamento Knowledge Base da '{KB_PATH}'...")
        self.prolog.consult(KB_PATH)

    def train_models(self, csv_path):
        """
        Addestra DUE modelli per confronto: Decision Tree (interpretabile) e Random Forest (robusto).
        """
        print("[ML] Avvio addestramento modelli...")
        try:
            df = pd.read_csv(csv_path)
            X = df.drop(columns=['Crop'])
            y = df['Crop']
            
            y_encoded = self.le.fit_transform(y)
            
            # 1. Decision Tree (White Box)
            self.model_dt = DecisionTreeClassifier(random_state=42, max_depth=10)
            self.model_dt.fit(X, y_encoded)
            
            # 2. Random Forest (Black Box - Ensemble)
            self.model_rf = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model_rf.fit(X, y_encoded)
            
            print("[ML] Modelli addestrati. Useremo Random Forest per la predizione principale.")
            
        except FileNotFoundError:
            print(f"[ERROR] File dataset non trovato: {csv_path}")

    def reasoning_pipeline(self, n, p, k, ph, rain, temp):
        """
        Pipeline Neuro-Simbolica Avanzata:
        ML (Random Forest) -> Prolog (Validazione) -> Prolog (Recovery) -> A* -> Bayes
        """
        # --- FASE 1: PREDIZIONE ML ---
        input_data = [[n, p, k, ph, rain, temp]]
        
        # Usiamo Random Forest per maggiore accuratezza
        pred_idx = self.model_rf.predict(input_data)[0]
        prediction_name = self.le.inverse_transform([pred_idx])[0]
        probabilita = np.max(self.model_rf.predict_proba(input_data))
        
        print(f"\n[AI] Random Forest suggerisce: '{prediction_name}' (Confidenza: {probabilita:.2f})")

        # --- FASE 2: RAGIONAMENTO ONTOLOGICO ---
        lotto_id = "lotto_corrente"
        list(self.prolog.query(f"retractall(dati_lotto({lotto_id},_,_,_,_,_,_))"))
        list(self.prolog.query(f"assertz(dati_lotto({lotto_id}, {n}, {p}, {k}, {ph}, {rain}, {temp}))"))
        
        # Validazione
        query_val = f"valida_raccomandazione({lotto_id}, {prediction_name})"
        is_valid = list(self.prolog.query(query_val))

        if is_valid:
            print(f">>> [PROLOG] VALIDATO. La coltura '{prediction_name}' rispetta i vincoli.")
        else:
            print(f">>> [PROLOG] CONFLITTO! '{prediction_name}' viola i vincoli bio-climatici.")
            
            # --- FASE 3: SEMANTIC RECOVERY (Novità rispetto a prima) ---
            print("    [RECOVERY] Avvio ricerca ontologica di alternative nella stessa famiglia...")
            query_alt = f"suggerisci_alternativa({lotto_id}, {prediction_name}, Alternativa)"
            alternative = list(self.prolog.query(query_alt))
            
            if alternative:
                # Deduplichiamo e prendiamo la prima
                nuova_coltura = alternative[0]['Alternativa']
                print(f"    [ADVISOR] Suggerimento Sostitutivo: '{nuova_coltura}'.")
                print(f"              (Motivo: È della stessa famiglia di '{prediction_name}' ma adatta al terreno)")
                return # Problema risolto col ragionamento
            else:
                print("    [ADVISOR] Nessuna alternativa tassonomica trovata. Situazione critica.")
                self.activate_drone_protocol(rain)

    def activate_drone_protocol(self, rain_val):
        """
        Gestisce la missione del drone se il ragionamento fallisce o rileva anomalie.
        """
        print("\n[MISSION] Attivazione Drone per ispezione fisica...")
        path, cost = a_star_search(mappa_agricola, 'Stazione_Ricarica', 'Lotto_Critico', euristica)
        
        if path:
            print(f"          Pathfinding A*: {' -> '.join(path)} (Costo: {cost}m)")
            
            # Simulazione rilevamento visivo
            print("          [CAM] Rilevamento: Foglie Gialle diffuse.")
            
            # --- FASE 4: DIAGNOSI PROBABILISTICA ---
            # Bayes usa i dati meteo reali
            p_mal, p_stress = self.diagnostica.stima_rischio(
                macchie=0, giallo=1, pioggia_mm=rain_val, umidita_pct=rain_val*0.8
            )
            print(f"          [BAYES] Stress Idrico: {p_stress:.2%}, Malattia: {p_mal:.2%}")
            
            if p_mal > p_stress:
                print("          [DECISION] Applicare funghi-cida (Evidence 'Explaining Away' usata).")
            else:
                print("          [DECISION] Aumentare irrigazione.")
        else:
            print("          [ERROR] Percorso bloccato (No-Fly Zones attive).")

def main():
    app = AgroSmartAI()
    app.train_models(DATASET_PATH)
    
    print("\n--- TEST CASO 1: STANDARD (Successo) ---")
    # Dati ideali per il Riso (molta acqua, ph neutro)
    app.reasoning_pipeline(n=80, p=40, k=40, ph=6.5, rain=250, temp=25)
    
    print("\n--- TEST CASO 2: CONFLITTO E RECUPERO (Intelligenza Simbolica) ---")
    # Proviamo a forzare un errore: Terreno ACIDO (pH 4.5) e ARIDO (Rain 300).
    # ML probabilmente suggerirà 'Jute' o 'Rice' o 'Beans' in base ai nutrienti, ma Prolog dovrebbe bloccare.
    # Nutrienti bassi -> Kidneybeans? Ma ph acido blocca legumi.
    # Vediamo se trova un'alternativa.
    app.reasoning_pipeline(n=20, p=60, k=20, ph=4.5, rain=350, temp=20)

if __name__ == "__main__":
    main()
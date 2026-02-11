import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from pyswip import Prolog
from pianificazione_drone import a_star_search, mappa_agricola, euristica
from diagnosi_bayesiana import DiagnosticaFitopatologica

# --- CONFIGURAZIONE SISTEMA ---
DATASET_PATH = 'Train_Dataset_Clean.csv'
KB_PATH = 'kb_agricola.pl'

class AgroSmartAI:
    """
    Classe principale che gestisce il Sistema Ibrido (ML + Reasoning).
    Integra un modello di Machine Learning (Decision Tree) con una 
    Knowledge Base Prolog per la validazione delle decisioni.
    """

    def __init__(self):
        self.model = None
        self.le = LabelEncoder()
        self.prolog = Prolog()
        # Inizializzazione Modulo Bayesiano
        self.diagnostica = DiagnosticaFitopatologica()
        
        # Caricamento della Knowledge Base
        print(f"[INIT] Caricamento Knowledge Base da '{KB_PATH}'...")
        self.prolog.consult(KB_PATH)
        print("[INIT] Knowledge Base caricata con successo.")

    def train_model(self, csv_path):
        """
        Addestra il modello di Machine Learning sui dati storici.
        Utilizza un Decision Tree per garantire interpretabilità.
        """
        print("[ML] Avvio addestramento modello...")
        try:
            df = pd.read_csv(csv_path)
            X = df.drop(columns=['Crop'])
            y = df['Crop']
            
            # Encoding delle label (Crop) in interi
            y_encoded = self.le.fit_transform(y)
            
            # Addestramento su tutto il dataset
            self.model = DecisionTreeClassifier(random_state=42)
            self.model.fit(X, y_encoded)
            print("[ML] Modello addestrato con successo.")
            
        except FileNotFoundError:
            print(f"[ERROR] File dataset non trovato: {csv_path}")

    def predict_and_validate(self, n, p, k, ph, rain, temp):
        """
        Metodo Core: Esegue la pipeline di inferenza.
        1. Predizione ML (Data-Driven)
        2. Validazione Prolog (Knowledge-Driven)
        """
        # 1. Predizione Machine Learning
        input_data = [[n, p, k, ph, rain, temp]]
        prediction_idx = self.model.predict(input_data)[0]
        prediction_name = self.le.inverse_transform([prediction_idx])[0]
        
        print(f"\n[AI RESULT] Il modello ML suggerisce: '{prediction_name}'")

        # 2. Interazione con Prolog per la validazione
        # Asserzione dinamica dei fatti (iniezione dei dati nel motore logico)
        lotto_id = "lotto_corrente"
        
        # Pulizia vecchi fatti per evitare conflitti
        list(self.prolog.query(f"retractall(dati_lotto({lotto_id},_,_,_,_,_,_))"))
        
        # Inserimento nuovi fatti: dati_lotto(ID, N, P, K, pH, Rain, Temp)
        # Nota: Prolog vuole i numeri, assicuriamoci siano formattati bene
        assert_query = f"assertz(dati_lotto({lotto_id}, {n}, {p}, {k}, {ph}, {rain}, {temp}))"
        list(self.prolog.query(assert_query))
        # Query di validazione
        # Chiediamo: "valida_raccomandazione(lotto_corrente, pianta_predetta)?"
        print(f"[LOGIC] Validazione logica in corso per '{prediction_name}'...")
        query = f"valida_raccomandazione({lotto_id}, {prediction_name})"
        result = list(self.prolog.query(query))

        # Analisi del risultato logico
        if result:
            print(">>> ESITO: VALIDATO. La coltura è sicura e coerente con i vincoli ambientali.")
        else:
            print(">>> ESITO: ATTENZIONE! La raccomandazione è stata RESPINTA dalla Knowledge Base.")
            print("    Motivo: Violazione di vincoli biologici.")

            # --- INTEGRAZIONE DRONE ---
            print("\n[ACTION] Avvio procedura di ispezione automatica...")
            print("         Calcolo percorso drone dalla Stazione al 'Lotto_Critico'...")
            
            percorso, costo = a_star_search(mappa_agricola, 'Stazione_Ricarica', 'Lotto_Critico', euristica)
            
            if percorso:
                print(f"         DRONE IN VOLO: {' -> '.join(percorso)}")
                print(f"         Distanza stimata: {costo} metri.")
                print("         [DRONE] Arrivo a destinazione. Avvio scansione visiva...")

                # --- SIMULAZIONE DIAGNOSI BAYESIANA ---
                # Simuliamo che il drone osservi dei sintomi: Macchie SI, Giallo SI
                print("\n[BAYES] Analisi Probabilistica Salute Pianta")
                print("        Sintomi rilevati: Macchie Foglie [SI], Ingiallimento [SI]")
                
                rischio = self.diagnostica.stima_rischio(macchie=1, giallo=1, crescita_lenta=0)
                
                print(f"        >>> PROBABILITÀ MALATTIA CALCOLATA: {rischio*100:.2f}%")
                
                if rischio > 0.80:
                    print("        [ALERT] Rischio elevato! Si consiglia trattamento fitosanitario immediato.")
                else:
                    print("        [INFO] Rischio moderato. Monitoraggio consigliato.")
            else:
                print("         [ERROR] Impossibile calcolare un percorso per il drone.")

def main():
    # Istanziazione del sistema
    app = AgroSmartAI()
    
    # Addestramento
    app.train_model(DATASET_PATH)
    
    # --- SIMULAZIONE INTERATTIVA ---
    print("\n--- AGRO-SMART ADVISOR: INSERIMENTO DATI ---")
    try:
        # Input utente (o valori di test fissi)
        n = int(input("Inserisci Azoto (N) [es. 90]: "))
        p = int(input("Inserisci Fosforo (P) [es. 40]: "))
        k = int(input("Inserisci Potassio (K) [es. 40]: "))
        ph = float(input("Inserisci pH [es. 5.5]: "))
        rain = float(input("Inserisci Pioggia mm (rainfall) [es. 200]: "))
        temp = float(input("Inserisci Temperatura [es. 25]: "))
        
        # Esecuzione
        app.predict_and_validate(n, p, k, ph, rain, temp)
        
    except ValueError:
        print("Errore: Inserire valori numerici validi.")

if __name__ == "__main__":
    main()
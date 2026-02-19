# 🌱 AgroSmart Advisor
- Studenti: Aurora Pesare & Fabrizio Stimolo
- Ambito: AI Ibrida per l'Agricoltura 4.0

## 🚀 Visione del Progetto
Agro-Smart Advisor è un ecosistema IA progettato per il supporto decisionale in agricoltura di precisione. 
Combina l'apprendimento statistico con il ragionamento logico e probabilistico per ottimizzare la scelta delle colture, la diagnosi delle malattie e la navigazione autonoma di droni.

## 🧠 Moduli del Sistema
Il sistema si basa sull'integrazione di quattro pilastri dell'IA:
- Machine Learning (Decision Tree)
- Knowledge Base (Prolog)
- Pianificazione
- Diagnostica (Rete Bayesiana)

## 🛠️ Installazione 

### 1. Prerequisiti
Assicurati di avere installato *SWI-Prolog* sul tuo sistema (necessario per la logica del progetto):
* Scaricalo da [swi-prolog.org](https://www.swi-prolog.org/)
* *Importante:* Durante l'installazione su Windows, seleziona l'opzione Add swipl to the system PATH.
### 2. Clonazione del Progetto
Apri il terminale e scarica il repository:
```bash
git clone [https://github.com/Aurorah17/AgroSmart.git](https://github.com/Aurorah17/AgroSmart.git)
cd AgroSmart
###3. Configurazione Ambiente Python
Su Windows: 
python -m venv venv
.\venv\Scripts\activate
Su macOS/Linux:
python3 -m venv venv
source venv/bin/activate
###4. Installazione Dipendenze
pip install --upgrade pip
pip install -r requirements.txt
```

## 💻 Istruzioni per l'Esecuzione
Per interagire con il sistema, utilizza i seguenti comandi:
  1. Avvio del Sistema Completo: Esegue il workflow integrato (ML → Prolog → A* → Bayes).
    python sistema_ibrido.py
  2. Valutazione Comparativa: Visualizza le metriche di performance e il confronto tra i modelli testati.
    python valutazione_modelli.py
  3. Visualizzazione Architettura ML: Genera il grafico dell'albero di decisione per interpretare le scelte del modello.
    python visualizza_albero.py
  4. Generazione grafici di performance, matrice di confusione e confronto tra i modelli testati (salvati in grafici_per_relazione/):
  - python valutazione_modelli.py
  - python test_finale.py
  - python generazione_grafici_doc.py
  - python generazione_extra_doc.py

## 📂 Struttura Repository
- sistema_ibrido.py: [Core] Script principale con interfaccia interattiva CLI.
- kb_agricola.pl: Knowledge Base in Prolog con ontologia e regole di dominio.
- valutazione_modelli.py / test_finale.py: Script di training, Cross-Validation (10-Fold) e test hold-out.
- generazione_.py: Script per l'estrazione automatica di grafici e schemi (A, Feature Importance).
- Train_Dataset_Clean.csv / test dataset.csv: Dataset utilizzati per il progetto.
- grafici_per_relazione/: Cartella di output contenente tutti i plot generati dal sistema.

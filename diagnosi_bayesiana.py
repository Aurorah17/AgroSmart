from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import logging

# Disabilitiamo i warning logistici di pgmpy per pulizia output
logging.getLogger("pgmpy").setLevel(logging.ERROR)

class DiagnosticaFitopatologica:
    """
    Modulo di Ragionamento Probabilistico (Capitolo 9).
    Implementa una Rete Bayesiana per diagnosticare la probabilità di malattia
    delle piante basandosi su osservazioni sintomatiche incerte (evidence).
    """

    def __init__(self):
        # 1. Definizione della Topologia della Rete (Grafo Aiciclico Diretto)
        # La 'Presenza_Malattia' influenza la probabilità dei sintomi.
        self.model = BayesianModel([
            ('Presenza_Malattia', 'Macchie_Foglie'),
            ('Presenza_Malattia', 'Ingiallimento_Foglie'),
            ('Presenza_Malattia', 'Crescita_Lenta')
        ])

        # 2. Definizione delle Tabelle delle Probabilità Condizionate (CPT)
        
        # Prior probability della malattia (senza evidenze, assumiamo 10% di rischio base)
        cpd_malattia = TabularCPD(variable='Presenza_Malattia', variable_card=2, 
                                  values=[[0.90], [0.10]]) # 0=No, 1=Sì

        # Probabilità di avere Macchie se: [Sano, Malato]
        # Se Sano: 5% probabilità macchie (falso positivo). Se Malato: 80% probabilità.
        cpd_macchie = TabularCPD(variable='Macchie_Foglie', variable_card=2, 
                                 values=[[0.95, 0.20], 
                                         [0.05, 0.80]],
                                 evidence=['Presenza_Malattia'], evidence_card=[2])

        # Probabilità Ingiallimento se: [Sano, Malato]
        cpd_giallo = TabularCPD(variable='Ingiallimento_Foglie', variable_card=2, 
                                values=[[0.90, 0.30], 
                                        [0.10, 0.70]],
                                evidence=['Presenza_Malattia'], evidence_card=[2])

        # Probabilità Crescita Lenta se: [Sano, Malato]
        cpd_crescita = TabularCPD(variable='Crescita_Lenta', variable_card=2, 
                                  values=[[0.95, 0.40], 
                                          [0.05, 0.60]],
                                  evidence=['Presenza_Malattia'], evidence_card=[2])

        # 3. Aggiunta delle CPD al modello e validazione
        self.model.add_cpds(cpd_malattia, cpd_macchie, cpd_giallo, cpd_crescita)
        assert self.model.check_model()
        
        # Inizializzazione motore inferenziale
        self.inferenza = VariableElimination(self.model)

    def stima_rischio(self, macchie=0, giallo=0, crescita_lenta=0):
        """
        Esegue l'inferenza bayesiana date le evidenze osservate dal drone.
        Input: 0 (Assente), 1 (Presente)
        Output: Probabilità di malattia (float)
        """
        evidence = {}
        # Inseriamo nel dizionario solo le evidenze certe osservate
        evidence['Macchie_Foglie'] = macchie
        evidence['Ingiallimento_Foglie'] = giallo
        evidence['Crescita_Lenta'] = crescita_lenta

        # Calcolo della probabilità a posteriori P(Malattia | Evidence)
        risultato = self.inferenza.query(variables=['Presenza_Malattia'], evidence=evidence)
        
        # Estraiamo il valore di probabilità per Malattia=1 (True)
        prob_malattia = risultato.values[1]
        return prob_malattia

# Test rapido del modulo se eseguito da solo
if __name__ == "__main__":
    bn = DiagnosticaFitopatologica()
    
    print("--- TEST RETE BAYESIANA ---")
    # Caso 1: Pianta con Macchie e Ingiallimento
    p = bn.stima_rischio(macchie=1, giallo=1, crescita_lenta=0)
    print(f"Evidenza: Macchie + Giallo. Probabilità Malattia: {p*100:.2f}%")
    
    # Caso 2: Pianta sana (nessun sintomo)
    p_sana = bn.stima_rischio(macchie=0, giallo=0, crescita_lenta=0)
    print(f"Evidenza: Nessuna. Probabilità Malattia: {p_sana*100:.2f}%")
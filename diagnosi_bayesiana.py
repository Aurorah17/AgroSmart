from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import logging

# Disabilitiamo i warning logistici
logging.getLogger("pgmpy").setLevel(logging.ERROR)

class DiagnosticaFitopatologica:
    """
    Rete Bayesiana Causale "Folta".
    Struttura:
    [Pioggia] ----> [Stress_Idrico] ----> [Ingiallimento] <---- [Presenza_Malattia]
                                                                        |
    [Umidità] --------------------------------------------------------->|
                                                                        v
                                                                  [Macchie_Foglie]
    """

    def __init__(self):
        # 1. Definizione della Topologia (Grafo Orientato)
        self.model = BayesianModel([
            ('Pioggia', 'Stress_Idrico'),           # La pioggia riduce lo stress idrico
            ('Umidità', 'Presenza_Malattia'),       # L'umidità favorisce i funghi (Malattia)
            ('Stress_Idrico', 'Ingiallimento'),     # Lo stress causa giallo
            ('Presenza_Malattia', 'Ingiallimento'), # Anche la malattia causa giallo (CAUSA COMUNE)
            ('Presenza_Malattia', 'Macchie_Foglie') # La malattia causa macchie
        ])

        # 2. TABELLE DELLE PROBABILITÀ (CPT)

        # --- NODI RADICE (Cause Ambientali) ---
        # Pioggia: 0=Scarsa, 1=Abbondante (Assumiamo 70% casi sia scarsa in estate)
        cpd_pioggia = TabularCPD(variable='Pioggia', variable_card=2, values=[[0.7], [0.3]])
        
        # Umidità: 0=Bassa, 1=Alta
        cpd_umidita = TabularCPD(variable='Umidità', variable_card=2, values=[[0.6], [0.4]])

        # --- NODI INTERMEDI (Stati Nascosti) ---
        
        # Stress Idrico dipende da Pioggia
        # Se Pioggia=Scarsa(0) -> Stress prob 80%. Se Pioggia=Abbondante(1) -> Stress prob 10%
        cpd_stress = TabularCPD(variable='Stress_Idrico', variable_card=2, 
                                values=[[0.2, 0.9],   # No Stress
                                        [0.8, 0.1]],  # Si Stress
                                evidence=['Pioggia'], evidence_card=[2])

        # Malattia dipende da Umidità
        # Se Umidità=Bassa(0) -> Malattia 10%. Se Umidità=Alta(1) -> Malattia 40%
        cpd_malattia = TabularCPD(variable='Presenza_Malattia', variable_card=2, 
                                  values=[[0.9, 0.6], 
                                          [0.1, 0.4]],
                                  evidence=['Umidità'], evidence_card=[2])

        # --- NODI FOGLIA (Sintomi Osservabili) ---

        # Macchie dipendono solo dalla Malattia
        cpd_macchie = TabularCPD(variable='Macchie_Foglie', variable_card=2, 
                                 values=[[0.95, 0.2], 
                                         [0.05, 0.8]], # 80% prob se malato
                                 evidence=['Presenza_Malattia'], evidence_card=[2])

        # Ingiallimento: IL NODO CRUCIALE (V-Structure)
        # Dipende sia da Stress(S) che da Malattia(M).
        # Ordine colonne evidence: 
        # S=0,M=0 (Sano) | S=0,M=1 (Solo Malato) | S=1,M=0 (Solo Stress) | S=1,M=1 (Entrambi)
        cpd_giallo = TabularCPD(variable='Ingiallimento', variable_card=2, 
                                values=[[0.95, 0.3, 0.2, 0.05],  # No Giallo
                                        [0.05, 0.7, 0.8, 0.95]], # Si Giallo
                                evidence=['Stress_Idrico', 'Presenza_Malattia'], evidence_card=[2, 2])

        # 3. Aggiunta e Validazione
        self.model.add_cpds(cpd_pioggia, cpd_umidita, cpd_stress, cpd_malattia, cpd_macchie, cpd_giallo)
        assert self.model.check_model()
        self.inferenza = VariableElimination(self.model)

    def stima_rischio(self, macchie, giallo, pioggia_mm, umidita_pct):
        """
        Input:
        - macchie, giallo: 0 o 1 (osservati dal drone)
        - pioggia_mm: float (dato dai sensori)
        - umidita_pct: float (dato dai sensori)
        """
        evidence = {}
        
        # Discretizzazione dei dati continui (trasformiamo i numeri in stati 0/1)
        stato_pioggia = 1 if pioggia_mm > 50 else 0  # Soglia arbitraria 50mm
        stato_umidita = 1 if umidita_pct > 60 else 0 # Soglia 60%
        
        evidence['Macchie_Foglie'] = macchie
        evidence['Ingiallimento'] = giallo
        evidence['Pioggia'] = stato_pioggia
        evidence['Umidità'] = stato_umidita

        # Query: Qual è la probabilità di malattia date TUTTE queste info?
        risultato = self.inferenza.query(variables=['Presenza_Malattia'], evidence=evidence)
        prob_malattia = risultato.values[1]
        
        # Extra: Diagnosi differenziale (opzionale per debug)
        # Possiamo chiedere anche quanto è probabile che sia solo stress idrico
        res_stress = self.inferenza.query(variables=['Stress_Idrico'], evidence=evidence)
        prob_stress = res_stress.values[1]

        return prob_malattia, prob_stress

if __name__ == "__main__":
    bn = DiagnosticaFitopatologica()
    
    print("--- TEST INTELLIGENTE ---")
    # Caso A: Foglie Gialle, Niente Macchie, MA HA PIOVUTO MOLTO.
    # Il sistema dovrebbe dire: "Non è stress idrico (ha piovuto), quindi quel giallo è sospetto Malattia".
    p_mal, p_stress = bn.stima_rischio(macchie=0, giallo=1, pioggia_mm=100, umidita_pct=80)
    print(f"Scenario A (Pioggia, Giallo): Rischio Malattia: {p_mal:.2f} | Rischio Stress: {p_stress:.2f}")

    # Caso B: Foglie Gialle, Niente Macchie, SICCITÀ TOTALE.
    # Il sistema dovrebbe dire: "Probabilmente è solo sete (Stress), rischio malattia basso".
    p_mal, p_stress = bn.stima_rischio(macchie=0, giallo=1, pioggia_mm=0, umidita_pct=20)
    print(f"Scenario B (Secco, Giallo):   Rischio Malattia: {p_mal:.2f} | Rischio Stress: {p_stress:.2f}")
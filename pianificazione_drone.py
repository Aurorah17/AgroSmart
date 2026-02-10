import heapq

# --- 1. DEFINIZIONE DELLA MAPPA (Grafo) ---
# Rappresentiamo l'azienda agricola come un grafo.
# I nodi sono i luoghi, i numeri sono le distanze in metri.
mappa_agricola = {
    'Stazione_Ricarica': [('Lotto_A', 10), ('Lotto_B', 15)],
    'Lotto_A': [('Lotto_C', 12), ('Lotto_D', 25)],
    'Lotto_B': [('Lotto_D', 10), ('Magazzino', 20)],
    'Lotto_C': [('Lotto_Critico', 5)],   # Destinazione vicina a C
    'Lotto_D': [('Lotto_Critico', 15)],  # Destinazione vicina a D
    'Magazzino': [('Lotto_Critico', 30)],
    'Lotto_Critico': [] # Destinazione finale
}

# --- 2. EURISTICA (Stima h(n)) ---
# Stima della distanza in linea d'aria verso 'Lotto_Critico'.
# Serve all'algoritmo A* per capire quale strada è "promettente".
euristica = {
    'Stazione_Ricarica': 30,
    'Lotto_A': 20,
    'Lotto_B': 25,
    'Lotto_C': 5,
    'Lotto_D': 12,
    'Magazzino': 28,
    'Lotto_Critico': 0
}

def a_star_search(grafo, start, goal, h):
    """
    Algoritmo A* (A-Star) per la ricerca del percorso ottimo.
    F(n) = G(n) + H(n)
    """
    # La frontiera è una coda di priorità ordinata per F
    frontiera = []
    heapq.heappush(frontiera, (0 + h[start], 0, start, [start]))
    
    visitati = set()
    
    print(f"Avvio ricerca percorso da '{start}' a '{goal}'...")

    while frontiera:
        # Prendo il nodo con F minore
        f, g, corrente, percorso = heapq.heappop(frontiera)

        if corrente == goal:
            return percorso, g # Trovato!

        if corrente not in visitati:
            visitati.add(corrente)
            
            # Espando i vicini
            for vicino, costo_arco in grafo.get(corrente, []):
                nuovo_g = g + costo_arco
                nuovo_f = nuovo_g + h.get(vicino, 0)
                heapq.heappush(frontiera, (nuovo_f, nuovo_g, vicino, percorso + [vicino]))

    return None, float('inf')

if __name__ == "__main__":
    # Testiamo il drone
    percorso, costo = a_star_search(mappa_agricola, 'Stazione_Ricarica', 'Lotto_Critico', euristica)
    
    print("-" * 40)
    print(f"PERCORSO OTTIMO: {' -> '.join(percorso)}")
    print(f"COSTO TOTALE: {costo} metri")
    print("-" * 40)
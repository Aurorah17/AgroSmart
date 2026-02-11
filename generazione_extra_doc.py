import matplotlib.pyplot as plt
import networkx as nx
import os

# Cartella output (la stessa di prima)
OUTPUT_DIR = 'grafici_per_relazione'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_mappa_drone():
    print("--- Generazione Mappa Drone con Percorso A* ---")
    
    # 1. Definiamo la Mappa (Grafo)
    # Nodi: Posizioni nell'azienda agricola
    G = nx.Graph()
    positions = {
        'Base': (0, 0),
        'Lotto_A': (2, 5),
        'Lotto_B': (5, 2),
        'Lotto_C': (6, 6),
        'Lotto_D': (8, 0),
        'Ricarica': (4, 4)
    }
    
    # Archi e costi (Distanze ipotetiche)
    edges = [
        ('Base', 'Lotto_A', 5), ('Base', 'Lotto_B', 6),
        ('Lotto_A', 'Ricarica', 3), ('Lotto_B', 'Ricarica', 3),
        ('Lotto_A', 'Lotto_C', 4), ('Lotto_B', 'Lotto_D', 4),
        ('Ricarica', 'Lotto_C', 3), ('Ricarica', 'Lotto_D', 3),
        ('Lotto_C', 'Lotto_D', 5)
    ]
    
    G.add_weighted_edges_from(edges)
    
    plt.figure(figsize=(10, 8))
    
    # Disegniamo tutti i nodi e gli archi (grigio chiaro)
    nx.draw_networkx_nodes(G, positions, node_size=2000, node_color='lightgray', edgecolors='black')
    nx.draw_networkx_edges(G, positions, width=2, edge_color='gray', style='dashed')
    nx.draw_networkx_labels(G, positions, font_size=10, font_weight='bold')
    
    # Etichette dei costi sugli archi
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels)
    
    # 2. Evidenziamo il Percorso Ottimo (Simulato A*: Base -> A -> C)
    path_nodes = ['Base', 'Lotto_A', 'Lotto_C']
    path_edges = [('Base', 'Lotto_A'), ('Lotto_A', 'Lotto_C')]
    
    nx.draw_networkx_nodes(G, positions, nodelist=path_nodes, node_color='orange', node_size=2100, edgecolors='red', linewidths=2)
    nx.draw_networkx_edges(G, positions, edgelist=path_edges, width=4, edge_color='red')
    
    plt.title("Pianificazione Percorso Drone (Algoritmo A*)", fontsize=15)
    plt.axis('off')
    
    outfile = os.path.join(OUTPUT_DIR, '5_mappa_drone_Astar.png')
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {outfile}")

def plot_tassonomia_ontologia():
    print("\n--- Generazione Tassonomia (Ontologia) ---")
    
    # Gerarchia delle classi (Simile a quella Prolog)
    hierarchy = [
        ('Pianta', 'Cereale'),
        ('Pianta', 'Legume'),
        ('Pianta', 'Frutto'),
        ('Pianta', 'Tuberosa'), # Es. Juta o Cotone se categorizzati variamente
        ('Cereale', 'Riso'),
        ('Cereale', 'Mais'),
        ('Legume', 'Cece'),
        ('Legume', 'Lenticchia'),
        ('Legume', 'Fagiolo'),
        ('Frutto', 'Cocomero'),
        ('Frutto', 'Melone'),
        ('Frutto', 'Uva'),
        ('Frutto', 'Mela'),
        ('Frutto', 'Banana'),
        ('Tuberosa', 'Juta'),
        ('Tuberosa', 'Caffe')
    ]
    
    G = nx.DiGraph()
    G.add_edges_from(hierarchy)
    
    plt.figure(figsize=(12, 8))
    
    # Layout ad albero (Graphviz sarebbe meglio, ma usiamo shell/spring per semplicità senza installare altro)
    # Cerchiamo di dare una struttura gerarchica manuale o semi-automatica
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    nx.draw(G, pos, with_labels=True, 
            node_color='lightgreen', 
            node_size=2500, 
            font_size=9, 
            font_weight='bold', 
            arrows=True, 
            arrowsize=20,
            edge_color='darkgreen')
    
    plt.title("Ontologia delle Colture (Gerarchia delle Classi)", fontsize=15)
    
    outfile = os.path.join(OUTPUT_DIR, '6_tassonomia_ontologia.png')
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {outfile}")

if __name__ == "__main__":
    plot_mappa_drone()
    plot_tassonomia_ontologia()
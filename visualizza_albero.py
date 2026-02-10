import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
import matplotlib.pyplot as plt
from sklearn import tree

def main():
    # 1. Carichiamo il dataset pulito che abbiamo salvato prima
    df = pd.read_csv('Train_Dataset_Clean.csv')
    X = df.drop(columns=['Crop'])
    y = df['Crop']

    # 2. Addestriamo un albero "semplice" per poterlo leggere (max_depth=3)
    # Se l'albero è troppo profondo, le regole diventano illeggibili per un umano
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    # 3. Visualizzazione Grafica
    plt.figure(figsize=(20,10))
    tree.plot_tree(clf, 
                   feature_names=X.columns,  
                   class_names=clf.classes_,
                   filled=True, 
                   rounded=True, 
                   fontsize=12)
    
    plt.title("Rappresentazione Semplificata della Conoscenza (Decision Tree)")
    plt.savefig('albero_decisionale.png')
    print("Immagine 'albero_decisionale.png' generata con successo!")

    # 4. Estrazione delle regole in formato testo
    tree_rules = export_text(clf, feature_names=list(X.columns))
    print("\nRegole estratte dall'albero:")
    print("-" * 40)
    print(tree_rules)

if __name__ == "__main__":
    main()
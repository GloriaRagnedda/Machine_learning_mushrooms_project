
import Preprocessing as pr
import decision_tree as dt
import knn_custom as knn 
import SVM as svm
import Naive_Bayes as nb
import AnalisiDati as an
import Random_Forest_Custom as rfc
import pandas as pd
from utils import get_user_input, plot_graphs, plot_compare_models, no_op
import combinazione_ottimale as co  


def prompt():
    text = """ 
    === MENU PRINCIPALE ===
    0. Esci
    1. Addestrare un Modello
    2. Analisi Dataset
    3. Confronto Modelli
    4. Trova la Combinazione Ottimale per un Modello
    """
    print(text)

    while True:
        choice = get_user_input("Cosa vuoi fare? (0/1/2/3/4): ")

        if choice in range(5):
            return choice
        else:
            print("Scelta non valida! Riprova.")



def choose_model():
    text = """
    === SCELTA MODELLO ===
    0. Decision Tree
    1. K-Nearest Neighbors
    2. Support Vector Machine
    3. Naive Bayes
    4. Random Forest (Custom)
    """
    print(text)

    models = [dt, knn, svm, nb, rfc]

    while True:
        choice = get_user_input("Quale modello vuoi addestrare? (0/1/2/3/4/5): ")

        if choice in range(5):
            return models[choice]
        else:
            print("Scelta modello non valida!")



def choose_dataset():
    text = """
    === SCELTA DATASET ===
    0. Dataset originale
    1. Dataset standardizzato
    2. Dataset con feature selezionate
    3. Dataset senza outlier
    4. Dataset bilanciato
    """
    print(text)

    getters = [
        pr.load_data_original,
        pr.standardized,
        pr.select_features,
        pr.remove_outliers,
        pr.bilanciamento
    ]

    while True:
        choice = get_user_input("Quale dataset vuoi usare? (0/1/2/3/4): ")

        if choice in range(5):
            return getters[choice]()
        else:
            print("Scelta dataset non valida!")


def train_single_model():
    model = choose_model()
    X, y = choose_dataset()

    print(f"\n=== INIZIO ADDESTRAMENTO ===")
    model.train_model(X, y)


def dataset_analysis():
    text = """
    === ANALISI DATASET ===
    0. Stampare le informazioni del DataFrame
    1. Mostrare le prime e ultime righe 
    2. Analisi descrittiva del dataset
    3. Mostrare la matrice di correlazione
    4. Mostrare i box plot delle variabili
    5. Distribuzione della variabile target 'class'
    """
    print(text)

    funcs = [
        an.print_info,
        an.mostra_prime_ultime_righe,
        an.descriptive_analysis,
        an.show_correlation_matrix,
        an.show_boxPlot_variables,
        an.plot_class_distribution
    ]

    dataset = pd.read_csv('./mushroom_cleaned.csv')  # Carichiamo il dataset intero

    while True:
        choice = get_user_input("Quale analisi vuoi eseguire? (0/1/2/3/4/5): ")

        if choice in range(6):
            return funcs[choice](dataset)
        else:
            print("Scelta non valida! Riprova.")



def compare_models():
    print("\n=== CONFRONTO MODELLI ===")
    print("\n--- Primo modello ---")
    model1 = choose_model()
    X1, y1 = choose_dataset()

    # Disabilita i plot intermedi per l'addestramento dei modelli
    model1.plot_graphs = no_op
    _, stats1 = model1.train_model(X1, y1)

    print("\n--- Secondo modello ---")
    model2 = choose_model()
    X2, y2 = choose_dataset()

    # Disabilita i plot intermedi per l'addestramento dei modelli
    model2.plot_graphs = no_op
    _, stats2 = model2.train_model(X2, y2)

    # Ripristina le funzioni di plotting originali
    model1.plot_graphs = model2.plot_graphs = plot_graphs

    plot_compare_models(stats1, stats2)



def optimal_preprocessing():
    print("\n=== PREPROCESSING OTTIMALE ===")

    model = choose_model()
    combo, data, stats = zip(*co.best_preprocessing_combination(model))

    #stampa tutte le statistiche per tutte le combinazioni di preprocessing
    for c, s in zip(combo, stats):
        print(c)
        for k,v in s.items():
            print(f"    {k} = {v}")
        print("------------------------------------")

    print(f"la combinazione migliore è: {combo[0]}")

    X,y = data[0]
    model.train_model(X,y)




def main():
    while True:
        match prompt():
            case 0:
                break

            case 1:
                train_single_model()

            case 2:
                dataset_analysis()

            case 3:
                compare_models()
            
            case 4:
                optimal_preprocessing()

    print("Uscita dal programma. Arrivederci!")
               


if __name__ == "__main__":
    main()



import numpy as np
import Preprocessing as pr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import plot_roc_curve, plot_confusion_matrix, plot_graphs, split_data

# -----------------------------
# Funzione per generare un report sui migliori risultati ottenuti dopo il tuning
# -----------------------------
def report_tuning(best_result, val_1, val_2):
    # Creazione di una tabella con le metriche di valutazione per Train, Validation e Test
    report_data = {
        "Metrica": ["Accuratezza", "F1-score", "Precisione", "Recall"],
        "Train": [best_result["Train Accuracy"], best_result["Train F1"], best_result["Train Precision"], best_result["Train Recall"]],
        "Validazione": [best_result["Validation Accuracy"], best_result["Validation F1"], best_result["Validation Precision"], best_result["Validation Recall"]],
        "Test": [best_result["Test Accuracy"], best_result["Test F1"], best_result["Test Precision"], best_result["Test Recall"]]
    }

    report_df = pd.DataFrame(report_data)

    # Stampa i valori migliori e il report in formato tabellare
    print(f"Valori migliori: {val_1}, {val_2}")
    print(report_df)

#-----------------------------
# Funzione per visualizzare i risultati del tuning con un grafico
# -----------------------------
def sub_plot(results_df, valori_k):
    # Creazione di un grafico per confrontare le performance del modello KNN con diverse configurazioni di k e metriche di distanza
    fig, ax = plt.subplots(figsize=(10, 6))

    # Definizione dei colori e degli stili per ogni tipo di distanza e dataset
    colors = {'Euclidea': 'magenta', 'Manhattan': 'green'}
    styles = {'Train': '-', 'Validation': '--', 'Test': ':'}
    
    # Iterazione sulle metriche per ciascun tipo di distanza e dataset
    for distance in ['Manhattan', 'Euclidea']:
        subset = results_df[results_df['Distance'] == distance]
        for dataset in ['Train', 'Validation', 'Test']:
            metric_col = f'{dataset} F1'
            ax.plot(valori_k, subset[metric_col], 
                    label=f'{distance} {dataset.lower()}', 
                    color=colors[distance], 
                    linestyle=styles[dataset])
    
    ax.set_title('KNN Model Performance Comparison')
    ax.set_xlabel('k Value')
    ax.set_ylabel('F1')
    ax.legend()
    ax.grid(True)

    # Mostra il grafico
    plt.show()

# -----------------------------
# Classe KNN personalizzato
# -----------------------------
class SimpleKNN(BaseEstimator):
    # Implementazione di un algoritmo K-Nearest Neighbors (KNN) personalizzato.

    def __init__(self, k=9, distance_type="Manhattan"):
        # Inizializza il modello con il numero di vicini (k) e il tipo di distanza.
        self.k = k
        self.distance_type = distance_type

    def fit(self, X, y):
        # Memorizza i dati di training.
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)  # Salva le classi presenti

    def predict(self, X):
        # Effettua la previsione per ogni punto nel set di test.
        X = np.array(X)
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Predice la classe di un singolo campione in base ai vicini più prossimi.

        # Calcola la distanza in base alla metrica scelta
        if self.distance_type == "Euclidea":
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.distance_type == "Manhattan":
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError("Tipo di distanza non supportato")

        # Seleziona i k vicini più vicini
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]

        # Calcola i pesi in base alla distanza (maggior peso ai più vicini)
        weights = 1 / (k_distances + 1e-5)
        weighted_votes = {}
        for label, weight in zip(k_nearest_labels, weights):
            weighted_votes[label] = weighted_votes.get(label, 0) + weight

        # Restituisce la classe con il peso più alto
        return max(weighted_votes, key=weighted_votes.get)
    
#-------------------------------
# Funzione per addestrare il modello KNN
#-------------------------------
def train_KNN(X_train, X_test, y_train, y_test, k, distance_type):
    
    # Inizializza il modello KNN con parametri predefiniti
    knn = SimpleKNN(k=k, distance_type=distance_type)
    
    # Addestra il modello
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Calcola le metriche di valutazione
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Visualizza il grafico delle metriche
    plot_graphs(y_test, y_pred)

    # Salva i risultati in un dizionario
    results = {
        "Model": "KNN",
        "K": k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

    return knn, results

# -----------------------------
# Crea il plot e stampa i migliori risultati del tuning
# -----------------------------
def plot_and_display_results_knn(results_df, valori_k, X_test, y_test, best_model):
    # Seleziona solo le distanze Manhattan ed Euclidea
    best_subset = results_df[results_df["Distance"].isin(["Manhattan", "Euclidea"])]

    # Trova il miglior risultato in base al Validation F1-score
    best_result = results_df.sort_values(by="Validation F1", ascending=False).iloc[0]
    best_distance = best_result["Distance"]
    best_k = int(best_result["k"])

    # Matrice di Confusione per il modello ottimizzato
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)

    # Usa direttamente la funzione esistente per la ROC Curve
    plot_roc_curve(y_test, y_pred)

    sub_plot(results_df, valori_k)

    report_tuning(best_result,best_distance,best_k)

# -------------------------------------
# Funzione per addestramento con tuning 
# -------------------------------------
def train_with_tuning_knn(X_train, X_test, y_train, y_test):
    # Tipi di distanza e range di k
    distance_types = ['Manhattan','Euclidea']
    valori_k = list(range(1, 19, 2))  # Solo valori dispari per k

    # Lista per memorizzare i risultati
    results = []
    best_f1 = 0  
    best_model = None  

    for distance in distance_types:
        print(f"Valutazione con distanza: {distance}")
        for k in valori_k:
            print(f" - Testando k = {k}...")
            knn = SimpleKNN(k=k, distance_type=distance)
            
            scores = cross_validate(
                estimator=knn,
                X=X_train,
                y=y_train,
                cv=10,
                scoring=['accuracy', 'f1', 'precision', 'recall'],
                n_jobs=-1,
                return_train_score=True
            )
            
            # Calcola le metriche
            train_accuracy = scores['train_accuracy'].mean()
            validation_accuracy = scores['test_accuracy'].mean()
            train_f1 = scores['train_f1'].mean()
            validation_f1 = scores['test_f1'].mean()
            train_precision = scores['train_precision'].mean()
            validation_precision = scores['test_precision'].mean()
            train_recall = scores['train_recall'].mean()
            validation_recall = scores['test_recall'].mean()
            
            # Score sul set di test 
            knn.fit(X_train, y_train)
            y_test_pred = knn.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            
            # Memorizza i risultati
            results.append({
                "k": k,
                "Distance": distance,
                "Train Accuracy": train_accuracy,
                "Validation Accuracy": validation_accuracy,
                "Test Accuracy": test_accuracy,
                "Train F1": train_f1,
                "Validation F1": validation_f1,
                "Test F1": test_f1,
                "Train Precision": train_precision,
                "Validation Precision": validation_precision,
                "Test Precision": test_precision,
                "Train Recall": train_recall,
                "Validation Recall": validation_recall,
                "Test Recall": test_recall
            })

            # Controlla se è il miglior modello basato sul F1-score di validazione
            if validation_f1 > best_f1:
                best_f1 = validation_f1
                best_model = knn

    # Converte i risultati in un DataFrame
    results_df = pd.DataFrame(results)

    # Visualizza i risultati
    plot_and_display_results_knn(results_df, valori_k, X_test, y_test, best_model)

# -----------------------------
# Funzione per la gestione completa del training
# -----------------------------
def train_model(X, y):
    # Inizializzo k e la distanza per il training
    k=9
    distance_type='Manhattan'
    
    # split dei dati
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    
    return train_KNN(X_train, X_test, y_train, y_test, k, distance_type)

    #print("Inizio il tuning dei parametri...")
    #train_with_tuning_knn(X_train, X_test, y_train, y_test)

# -----------------------------
# Esecuzione principale dello script
# -----------------------------
if __name__ == '__main__':
    X, y = pr.load_data_original()
    X_train, X_test, y_train, y_test = split_data(X, y)
    train_with_tuning_knn(X_train, X_test, y_train, y_test)
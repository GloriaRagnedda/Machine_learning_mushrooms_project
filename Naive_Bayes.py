import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import stats, plot_graphs, split_data

# -----------------------------
# Funzione per addestramento del modello
# -----------------------------
def train_naive_bayes(X_train, X_test, y_train, y_test):

    # Creazione e addestramento del modello Naive Bayes
    GNB = GaussianNB()
    GNB.fit(X_train, y_train)

    # Predizione sui dati di test
    y_pred = GNB.predict(X_test)
    
    # Matrice di confusione + curva roc
    plot_graphs(y_test, y_pred)

    results = stats("Naive Bayes",y_test,y_pred)
    # Ritorna il modello e un record con tutte le metriche
    return GNB, results

# -----------------------------
# Funzione per addestramento del modello con tuning
# -----------------------------
def train_NB_with_tuning(X_train, X_test, y_train, y_test):
    # Definizione della griglia di parametri
    param_grid = {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    }

    # Creazione del modello base
    GNB = GaussianNB()

    # Ricerca del miglior modello con validazione incrociata
    grid_search = GridSearchCV(GNB, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Migliori parametri trovati
    best_params = grid_search.best_params_
    print(f"\nMigliori parametri trovati: {best_params}")

    plot_tuning(grid_search)

    # Addestriamo il modello con i migliori parametri trovati
    best_GNB = grid_search.best_estimator_

    # Predizione sui dati di test
    y_pred = best_GNB.predict(X_test)

    # Calcola metriche principali
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Matrice di confusione + curva roc
    plot_graphs(y_test, y_pred)

    # Restituisce il modello ottimizzato e le metriche
    results = {
        "Model": "Naive Bayes Tuning",
        "Best Params": best_params,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
    return best_GNB, results

# -----------------------------
# Funzione per gestione completa del training
# -----------------------------
def train_model(X, y):
    # Split dei dati
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Addestramento con tuning
    return train_naive_bayes(X_train, X_test, y_train, y_test)
    # Addestramento con tuning
    # train_NB_with_tuning(X_train, X_test, y_train, y_test)


def plot_tuning(grid_search):
        # 1) Estraiamo i risultati dal grid search
    cv_results = grid_search.cv_results_
    # 2) Ricaviamo i valori testati di var_smoothing, la media e la std del F1
    var_smoothing_values = [p["var_smoothing"] for p in cv_results["params"]]
    mean_test_scores = cv_results["mean_test_score"]
    std_test_scores = cv_results["std_test_score"]

    # 3) Creiamo un subplot per mostrare l’andamento dei punteggi
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        var_smoothing_values,
        mean_test_scores,
        yerr=std_test_scores,
        fmt='o-',
        capsize=4,
        label="F1"
    )
    ax.set_xscale("log")  # spesso utile per var_smoothing (scala log)
    ax.set_xlabel("var_smoothing")
    ax.set_ylabel("Mean CV F1")
    ax.set_title("Naive Bayes - Hyperparameter Tuning")
    ax.legend()
    plt.show()
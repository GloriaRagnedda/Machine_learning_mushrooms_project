# Importazione delle librerie necessarie
import pandas as pd  # Per la gestione e l'analisi dei dati in formato tabellare
import numpy as np  # Per operazioni numeriche e matriciali
import Preprocessing as pr  # Modulo personalizzato per la pre-elaborazione dei dati
import matplotlib.pyplot as plt  # Per la visualizzazione dei dati
from sklearn.svm import SVC  # Support Vector Machine per la classificazione
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score  # Metriche per la valutazione del modello
import joblib  # Per salvare e caricare il modello addestrato
from sklearn.model_selection import GridSearchCV, cross_val_predict  # Per la ricerca degli iperparametri e la validazione incrociata
from utils import plot_graphs, stats, split_data  # Funzioni personalizzate di supporto

# -----------------------------
# Funzione per report formattato come tabella
# -----------------------------
def report_tuning(best_result):
    """
    Genera un report tabellare con le metriche di accuratezza, F1-score, precisione e recall 
    per i set di addestramento, validazione e test.
    """
    report_data = {
        "Metrica": ["Accuratezza", "F1-score", "Precisione", "Recall"],
        "Train": [
            best_result["Train Accuracy"],
            best_result["Train F1"],
            best_result["Train Precision"],
            best_result["Train Recall"]
        ],
        "Validazione": [
            best_result["Validation Accuracy"],
            best_result["Validation F1"],
            best_result["Validation Precision"],
            best_result["Validation Recall"]
        ],
        "Test": [
            best_result["Test Accuracy"],
            best_result["Test F1"],
            best_result["Test Precision"],
            best_result["Test Recall"]
        ]
    }
    
    # Creazione del DataFrame per una visualizzazione più chiara
    report_df = pd.DataFrame(report_data)
    print(report_df)


# -----------------------------
# Funzione per addestramento del modello SVM
# -----------------------------
def train_svm(X_train, y_train, X_test, y_test):
    """
    Addestra un modello di Support Vector Machine (SVM) con kernel RBF 
    e lo testa sui dati di test.
    """
    # Creazione del modello SVM con kernel RBF (Radial Basis Function)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)  
    # probability=True consente di ottenere le probabilità di appartenenza alle classi

    # Addestramento del modello sui dati di training
    svm_model.fit(X_train, y_train)

    # Predizione delle classi sul set di test
    y_pred = svm_model.predict(X_test)
    
    # Visualizzazione dei risultati con funzioni di supporto
    plot_graphs(y_test, y_pred)

    # Ritorno del modello addestrato e delle metriche di valutazione
    return svm_model, stats("SVM", y_test, y_pred)


# -----------------------------
# Funzione di tuning con GridSearchCV
# -----------------------------
def tuning(X_train, y_train, X_test, y_test):
    """
    Ottimizza il modello SVM usando GridSearchCV per il tuning degli iperparametri.
    """

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler()  # Normalizzazione dei dati tra 0 e 1
    X_train_scaled = scaler.fit_transform(X_train)  # Applicazione dello scaling ai dati di training
    X_test_scaled = scaler.transform(X_test)  # Applicazione dello stesso scaling ai dati di test

    # Griglia di iperparametri da testare
    param_grid = [
        {
            'kernel': ['rbf'],  
            'C': [1.0, 10, 100, 1000],  # Valori della costante di penalizzazione
            'gamma': ['scale'],
            'max_iter': [5000],  # Numero massimo di iterazioni
        }
    ]

    # Creazione del modello SVM con opzioni aggiuntive
    svm = SVC(
        probability=True, 
        random_state=42,
        cache_size=2000,  # Aumento della cache per migliorare la velocità
        class_weight='balanced'  # Gestione degli squilibri tra le classi
    )
    
    # Implementazione della ricerca degli iperparametri con GridSearchCV
    grid_search = GridSearchCV(
        estimator=svm, 
        param_grid=param_grid,
        scoring='f1_weighted',  # Metriche usate per la valutazione del modello
        cv=5,  # Validazione incrociata a 5 fold
        verbose=2,  # Livello di verbosità
        n_jobs=-1,  # Uso di tutti i processori disponibili
        return_train_score=True
    )

    # Debug: stampa delle dimensioni dei dati
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Addestramento con ricerca degli iperparametri
    grid_search.fit(X_train_scaled, y_train)
    
    # Miglior modello trovato
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\nMigliori parametri trovati: {best_params}")

    # Creazione di un DataFrame con tutti i risultati del tuning
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Predizione sui dati di test con il miglior modello trovato
    y_pred = best_model.predict(X_test_scaled)

    # Calcolo delle metriche principali
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Visualizzazione dei risultati con grafici
    plot_graphs(y_test, y_pred)

    # Ritorno del modello ottimizzato e delle metriche
    results = {
        "Model": "SVM Tuning",
        "Best Params": best_params,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
    return best_model, results
    
# ------------------------------------------------------
#  Funzione che stampa i risultati migliori e plotta
# ------------------------------------------------------

def plot_tuning(grid_search):
    """
    Visualizza l'andamento dei punteggi medi (con relative deviazioni standard)
    in funzione del parametro C, suddividendo i plot per kernel e tracciando 
    una linea per ogni valore di gamma.
    """
    # Estraiamo i risultati dal grid search
    cv_results = grid_search.cv_results_
    
    # Estraiamo i valori unici dei parametri
    kernels = sorted(set(cv_results['param_kernel']))
    gammas = sorted(set(cv_results['param_gamma']), key=lambda x: str(x))
    # Anche se i valori di C verranno ordinati in seguito, li definiamo per completezza
    Cs = sorted(set(cv_results['param_C']))
    
    # Creiamo un subplot per ciascun kernel
    n_kernels = len(kernels)
    fig, axs = plt.subplots(1, n_kernels, figsize=(6 * n_kernels, 6), sharey=True)
    # Se c'è un solo kernel, trasformiamo axs in una lista per uniformare la gestione
    if n_kernels == 1:
        axs = [axs]
    
    # Per ogni kernel, tracciamo una linea (con barre d'errore) per ciascun valore di gamma
    for ax, kernel in zip(axs, kernels):
        for gamma in gammas:
            # Filtriamo gli indici in cui i parametri 'kernel' e 'gamma' corrispondono
            indices = [
                i for i, params in enumerate(cv_results['params'])
                if params['kernel'] == kernel and params['gamma'] == gamma
            ]
            # Estraiamo per questi indici i valori di C, il punteggio medio e lo std
            C_vals = [cv_results['param_C'][i] for i in indices]
            mean_scores = [cv_results['mean_test_score'][i] for i in indices]
            std_scores = [cv_results['std_test_score'][i] for i in indices]
            
            # Ordiniamo i dati in base al valore di C
            sorted_idx = sorted(range(len(C_vals)), key=lambda i: C_vals[i])
            C_sorted = [C_vals[i] for i in sorted_idx]
            mean_sorted = [mean_scores[i] for i in sorted_idx]
            std_sorted = [std_scores[i] for i in sorted_idx]
            
            # Tracciamo la curva con barre d'errore per questo valore di gamma
            ax.errorbar(
                C_sorted,
                mean_sorted,
                yerr=std_sorted,
                marker='o',
                capsize=4,
                label=f"gamma: {gamma}"
            )
        
        ax.set_xscale('log')  # Poiché C assume valori distribuiti su scala logaritmica
        ax.set_xlabel("C")
        ax.set_title(f"Kernel: {kernel}")
        ax.grid(True)
    
    # Impostiamo l'etichetta dell'asse y sul primo subplot
    axs[0].set_ylabel("Mean CV F1 (weighted)")
    
    # Creiamo una legenda comune
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Gamma", loc='upper right')
    
    plt.suptitle("Hyperparameter Tuning")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# -# -----------------------------
# Funzione per l'addestramento del modello
# -----------------------------
def train_model(X, y):
    """
    Questa funzione suddivide i dati in set di training e test, 
    quindi addestra un modello SVM utilizzando la funzione `train_svm`.

    Parametri:
    - X: Matrice delle caratteristiche (features)
    - Y: Vettore dei target (etichette di classe)

    Ritorna:
    - Il modello addestrato e le metriche di valutazione.
    """

    # Split dei dati in training e test set
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Addestramento del modello SVM
    return train_svm(X_train, y_train, X_test, y_test)  

    # Codice commentato: possibilità di eseguire il tuning degli iperparametri invece dell'addestramento standard
    # print("Inizio il tuning dei parametri...")
    # tuning(X_train, y_train, X_test, y_test)  


# -----------------------------
# Blocco principale di esecuzione
# -----------------------------
if __name__ == '__main__':
    """
    Questo blocco viene eseguito solo se lo script è avviato direttamente,
    e non se viene importato come modulo in un altro script.
    """

    # Caricamento dei dati utilizzando la funzione `load_data_original` dal modulo di preprocessing
    X, y = pr.load_data_original()

    # Suddivisione dei dati in training e test set
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Avvio del tuning degli iperparametri sul modello SVM
    tuning(X_train, y_train, X_test, y_test)

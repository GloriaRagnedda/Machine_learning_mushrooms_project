from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import Preprocessing as pr
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from utils import plot_roc_curve, plot_confusion_matrix, plot_graphs, split_data, stats

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


# -----------------------------
# Funzione per visualizzare i risultati del tuning con subplot
# -----------------------------
def sub_plot_tuning(best_subset):
    # Creazione di un layout 2x2 per la visualizzazione delle metriche in sottografici
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f'Model Performance', fontsize=16)

    # Definizione delle metriche e dei titoli corrispondenti
    metrics = [
        ("Train Accuracy", "Validation Accuracy", "Test Accuracy", "Accuracy"),
        ("Train F1", "Validation F1", "Test F1", "F1 Score"),
        ("Train Precision", "Validation Precision", "Test Precision", "Precision"),
        ("Train Recall", "Validation Recall", "Test Recall", "Recall"),
    ]

    # Creazione di ciascun subplot iterando attraverso le metriche
    for ax, (train_metric, val_metric, test_metric, title) in zip(axes.flatten(), metrics):
        ax.plot(best_subset["Max Depth"], best_subset[train_metric], color='red', lw=2, label='Train')
        ax.plot(best_subset["Max Depth"], best_subset[val_metric], color='blue', lw=2, label='Validation')
        ax.plot(best_subset["Max Depth"], best_subset[test_metric], color='green', lw=2, label='Test')

        ax.set_xlabel("Max Depth", fontsize=10, labelpad=10)  # Etichetta asse x con maggiore spaziatura
        ax.set_ylabel(title, fontsize=10, labelpad=10)  # Etichetta asse y con maggiore spaziatura
        ax.set_title(title, fontsize=14, pad=15)  # Titolo del subplot con padding aumentato
        ax.grid(True)
        ax.legend()

    # Mostra il grafico
    plt.show()


# -----------------------------
# Funzione per addestrare un albero decisionale sui dati di training
# -----------------------------
def train_decision_tree(X_train, X_test, y_train, y_test):
    # Inizializza il modello Decision Tree con un seed fisso per la riproducibilità
    dt = DecisionTreeClassifier(random_state=42)
    
    # Addestra il modello sui dati di training
    dt.fit(X_train, y_train)
    
    # Effettua le previsioni sul set di test
    y_pred = dt.predict(X_test)
    
    # Chiamata alla funzione per visualizzare le metriche di valutazione del modello
    plot_graphs(y_test, y_pred)

    # Salvataggio dei risultati in un dizionario per analisi successive
    results = stats("Decision Tree", y_test, y_pred)

    return dt, results


# -----------------------------
# Funzione per generare e visualizzare i migliori risultati del tuning
# -----------------------------
def plot_and_display_results_tuning(results_df, X_test, y_test, best_model):    
    # Seleziona il criterio con il miglior Validation F1-score
    best_result = results_df.sort_values(by="Validation F1", ascending=False).iloc[0]
    best_criterion = best_result["Criterion"]
    best_depth = int(best_result["Max Depth"])
    
    # Filtra il dataframe per il criterio migliore selezionato
    best_subset = results_df[results_df["Criterion"] == best_criterion]

    # Genera la matrice di confusione per il modello ottimizzato
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)

    # Genera la curva ROC per il modello ottimizzato
    plot_roc_curve(y_test, y_pred)
    
    # Genera i grafici di tuning per visualizzare le metriche rispetto alla profondità dell'albero
    sub_plot_tuning(best_subset) 
    
    # Stampa il report dei migliori risultati
    report_tuning(best_result, best_depth, best_criterion)


# -----------------------------
# Funzione per addestrare un albero decisionale con tuning dei parametri
# -----------------------------
def train_with_tuning(X_train, X_test, y_train, y_test):
    # Definizione dei criteri di impurezza da testare
    criterions = ['gini', 'entropy']
    
    # Definizione dell'intervallo di valori per la profondità massima dell'albero
    max_depth_range = list(range(1, 25))
    
    # Variabili per memorizzare i migliori risultati
    best_model = None
    best_f1 = 0
    results = []
    
    # Loop su ogni combinazione di criterio di impurezza e profondità massima
    for criterion in criterions:
        for depth in max_depth_range:
            # Inizializza un albero decisionale con i parametri attuali
            clf = DecisionTreeClassifier(criterion=criterion,
             max_depth=depth,
             random_state=42)

            # Esegue una validazione incrociata (cross-validation) con 10 fold
            scores = cross_validate(
                estimator=clf,
                X=X_train,
                y=y_train,
                cv=10,
                n_jobs=-1,
                scoring=['accuracy', 'f1', 'precision', 'recall'],
                return_train_score=True,
                return_estimator=True
            )
            
            # Calcola le metriche medie sulla validazione incrociata
            train_accuracy = scores['train_accuracy'].mean()
            validation_accuracy = scores['test_accuracy'].mean()
            train_f1 = scores['train_f1'].mean()
            validation_f1 = scores['test_f1'].mean()
            train_precision = scores['train_precision'].mean()
            validation_precision = scores['test_precision'].mean()
            train_recall = scores['train_recall'].mean()
            validation_recall = scores['test_recall'].mean()
            
            # Addestra il modello e calcola le metriche sul set di test
            clf.fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            
            # Salva i risultati in un dizionario
            results.append({
                "Criterion": criterion,
                "Max Depth": depth,
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

            # Aggiorna il miglior modello basato sul Validation F1-score
            if validation_f1 > best_f1:
                best_f1 = validation_f1
                best_model = clf

    # Converte i risultati in un DataFrame
    results_df = pd.DataFrame(results)

    # Visualizza i risultati migliori
    plot_and_display_results_tuning(results_df, X_test, y_test, best_model)

    return results_df


# -----------------------------
# Funzione per gestione completa del training
# -----------------------------
def train_model(X, y):
    # Split dei dati
    X_train, X_test, y_train, y_test = split_data(X, y)
    return train_decision_tree(X_train, X_test, y_train, y_test)
    
    # Addestramento con tuning
    #print("Inizio il tuning dei parametri...")
    #train_with_tuning(X_train, X_test, y_train, y_test) 

# -----------------------------
# Esecuzione principale dello script
# -----------------------------
if __name__ == '__main__':
    # Carica i dati dal preprocessing
    X, y = pr.load_data_original()
    
    # Divide i dati in training e test set
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Avvia il tuning del modello KNN
    train_with_tuning(X_train, X_test, y_train, y_test)
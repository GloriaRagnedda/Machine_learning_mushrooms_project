import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

from sklearn.metrics import (ConfusionMatrixDisplay,
                            classification_report,
                            roc_curve,
                            auc,
                            confusion_matrix)


def no_op(*args, **kwargs):
    pass

def plot_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Linea diagonale
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    disp.ax_.set_title(f'Matrice di confusione')
    plt.show()

def plot_graphs(y_test, y_pred):
    report = classification_report(y_test, y_pred) #precision recall f1-score support

    print("\n=== Risultati ===")
    print(report)

    # Plotta la curva ROC
    plot_roc_curve(y_test, y_pred)

    # matrice di confusione
    plot_confusion_matrix(y_test, y_pred)

def split_data(X, y, test_size=0.20, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def get_user_input(text):
    try:
        return int(input(text))
    
    except ValueError:
        return -1

    except Exception as e:
        raise


def stats(model, y_test, y_pred):
    # Calcola metriche principali
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Salva i risultati in un dizionario
    return {
        "Model": model,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

def plot_compare_models(stats1, stats2):
    labels = [stats1["Model"], stats2["Model"]]
    scores = [stats1["F1-score"], stats2["F1-score"]]

    # Creazione del plot
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(labels))  
    width = 0.35  

    bars = ax.bar(x, scores, color=["#1f77b4", "#228B22"], width=width)

    # Mostra il valore sopra ciascuna barra
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title(f"Confronto tra modelli: {labels[0]} vs {labels[1]}", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)  
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Mostrare il plot
    plt.show()
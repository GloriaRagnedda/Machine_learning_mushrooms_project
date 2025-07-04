import pandas as pd  # Libreria per la manipolazione dei dati
import matplotlib.pyplot as plt  # Libreria per creare grafici
import seaborn as sns  # Libreria per visualizzazioni avanzate

# Percorso del file CSV
file_path = 'mushroom_cleaned.csv'
# Carica il dataset dal file CSV
dataset = pd.read_csv(file_path)

# Funzione per stampare informazioni di base sul DataFrame
def print_info(dataframe):
    if isinstance(dataframe, pd.DataFrame):  # Controlla se l'oggetto passato è un DataFrame
        print("\nIformazioni Dataframe")
        
        # Cattura l'output del metodo info() per manipolarlo
        from io import StringIO
        buffer = StringIO()
        dataframe.info(buf=buffer)  # Scrive l'output in un buffer
        info_output = buffer.getvalue()  # Recupera il contenuto del buffer
        
        # Rimuove le ultime due righe dell'output (ad esempio, memoria utilizzata)
        lines = info_output.strip().split('\n')[1:-2]
        truncated_output = '\n'.join(lines)
        
        print(truncated_output)  # Stampa l'output troncato
    else:
        print("Errore")  # Messaggio d'errore se l'input non è un DataFrame

# Funzione per visualizzare le prime e ultime righe del DataFrame
def mostra_prime_ultime_righe(dataframe, n=5):
    if isinstance(dataframe, pd.DataFrame):  # Controlla se l'oggetto passato è un DataFrame
        print("\nPrime righe del DataFrame:")
        print(dataframe.head(n))  # Mostra le prime n righe

        print("\nUltime righe del DataFrame:")
        print(dataframe.tail(n))  # Mostra le ultime n righe
    else:
        print("Errore: l'input non è un DataFrame")  # Messaggio d'errore se l'input non è un DataFrame
      
# Funzione per mostrare la matrice di correlazione degli attributi
def show_correlation_matrix(dataframe):
    plt.figure(figsize=(12, 8))  # Imposta la dimensione del grafico

    correlation_matrix = dataframe.corr()  # Calcola la matrice di correlazione

    # Crea una heatmap per visualizzare la matrice di correlazione
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

    plt.title("Correlation Matrix")  # Titolo del grafico
    plt.show()  # Mostra il grafico

# Funzione per l'analisi descrittiva completa
def descriptive_analysis(data):

    analysis = pd.DataFrame({
        "Mean": data.mean(numeric_only=True),
        "Median": data.median(numeric_only=True),
        "Standard Deviation": data.std(numeric_only=True),
        "Variance": data.var(numeric_only=True),
        "Min": data.min(numeric_only=True),
        "Max": data.max(numeric_only=True),
        "Missing Values": data.isnull().sum(),
        "Duplicate Values": data.duplicated().sum()
    })

    # Calcolare quartili separatamente
    quartiles = data.quantile([0.25, 0.5, 0.75], numeric_only=True).T
    quartiles.columns = ['25th Percentile', '50th Percentile', '75th Percentile']

    # Unire le tabelle
    result = pd.concat([analysis, quartiles], axis=1)

    # Ordinare per valore medio
    result = result.sort_values(by="Mean", ascending=False)
    
    # Mostrare il risultato completo senza abbreviazioni
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    
    # Mostrare il risultato
    print(result)
    
#Grafico della distribuzione per `class`
def show_distribution_quality(dataset):
    # Configurazione dello stile
    sns.set(style="whitegrid")

    # 1. Grafico della distribuzione per `class`
    plt.figure(figsize=(6, 4))
    sns.countplot(x='class', hue='class', data=dataset, palette='pastel', legend=False)
    plt.title('Distribuzione di class', fontsize=14)
    plt.xlabel('Quality Binary', fontsize=12)
    plt.ylabel('Conteggio', fontsize=12)
    plt.show()

# Box plot per evidenziare outlier nelle variabili numeriche
def show_boxPlot_variables(dataset):

    # Creazione dei boxplot per tutte le variabili eccetto quella target ('class')
    #columns = dataset.drop(columns=["class"]).columns

    # Generare boxplot per ogni variabile per identificare outlier
    numerical_columns = dataset.drop(columns=["class"]).columns

    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        plt.boxplot(dataset[col], vert=False, patch_artist=True)
        plt.title(f"Boxplot di {col}")
        plt.xlabel(col)
        plt.show()

# distribuzione della variabile Class
def plot_class_distribution(dataframe):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='class', data=dataframe, palette='pastel', hue='class')
    plt.title('Distribuzione della variabile Class', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Conteggio', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=["Class 0", "Class 1"])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

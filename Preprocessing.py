import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import InstanceHardnessThreshold, NearMiss
from sklearn.model_selection import train_test_split

# -----------------------------
# Funzione per il caricamento dei dati originali
# -----------------------------
def load_data_original():
        file_path = './mushroom_cleaned.csv'
        df = pd.read_csv(file_path)
        X = df.drop(columns=["class"])
        y = df["class"]
        return X, y

#------------------------------
# Carica un sottinsieme minuscolo del dataset (utile per i test)
#------------------------------
def dummy(data=load_data_original()):
    X, y = data
    X, y = train_test_split(X, y, train_size=0.05, test_size=0.01)
    return X, y

# -----------------------------
# Funzione per il caricamento dei dati standardizzati
# -----------------------------
def standardized(data=load_data_original()):
    X, y = data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y
    

#-----------------------------------------------------------------------
#                              FEATURE SELECTION
#    Seleziona le feature con una varianza superiore alla soglia specificata.
#--------------------------------------------------------------------------
def select_features(data=load_data_original()):

    X, y = data

    # Calcolo della matrice di correlazione
    X = pd.DataFrame(X)
    corr_matrix = X.corr()

    # Identifica colonne con alta correlazione (> 0.82)
    high_corr = np.where(np.abs(corr_matrix.to_numpy()) > 0.82)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y])
                        for x, y in zip(*high_corr) if x != y and x < y]
    to_remove_high = list(set([pair[1] for pair in high_corr_pairs]))

    # Rimuovi colonne ad alta correlazione
    X_reduced = X.drop(columns=to_remove_high)

    # Applica VarianceThreshold per rimuovere colonne con bassa varianza
    threshold = 0.1
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = X_reduced.loc[:, selector.fit(X_reduced).get_support()]

    X = X_reduced

    return X, y

#---------------------------------
#    Rimozione degli outliers
#---------------------------------
def remove_outliers(data=load_data_original(), colonna_target='class'):
    # Carica il dataset originale come tuple (X, y)
    X, y = data
    # Combina X e y in un unico DataFrame per elaborare gli outlier
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    
    df = X.copy()
    df[colonna_target] = y.copy()

    # Identifica le colonne da processare (tutte tranne la colonna target)
    colonne_da_processare = [col for col in df.columns if col != colonna_target]

    # Rimuovi gli outlier per ogni colonna
    for col in colonne_da_processare:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Dividi il DataFrame risultante in X e y
    X = df.drop(columns=[colonna_target])
    y = df[colonna_target]

    return X, y

#---------------------------------
#    Bilanciamento del dataset
#---------------------------------
def bilanciamento(data=load_data_original(), metodo_undersampling="NearMiss_v1"):
    X, y = data
    # Calcolo dell'aggiustamento necessario per il bilanciamento
    class_counts = Counter(y)
    adjustment = (class_counts[1] - class_counts[0]) / 2
    smote_class_0_balanced = round(class_counts[0] + adjustment)
    undersampled_class_1_balanced = round(class_counts[1] - adjustment)

    # 1. Applicare SMOTE per bilanciare la classe 0
    smote = SMOTE(sampling_strategy={0: smote_class_0_balanced}, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # 2. Applicare Instance Hardness Threshold (IHT) per ridurre la classe 1
    iht = InstanceHardnessThreshold()
    X_balanced, y_balanced = iht.fit_resample(X_smote, y_smote)

    # 3. Se la classe 1 è ancora maggiore della classe 0, applicare Nearmiss
    final_counts = Counter(y_balanced)
    if final_counts[1] > final_counts[0]:  
        # Applico nearmiss per bilanciare definitivamente il dataset
        nm = NearMiss(version=1)  
        X_final, y_final = nm.fit_resample(X_balanced, y_balanced)        
    else:
        X_final, y_final = X_balanced, y_balanced
   
        X = X_final 
        y = y_final

    return X, y
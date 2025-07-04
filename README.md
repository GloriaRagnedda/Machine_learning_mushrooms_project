# Mushroom Classification

**Mushroom Classification** è un progetto di machine learning che mira a classificare funghi come **commestibili** (`edible`) o **velenosi** (`poisonous`) attraverso una **classificazione binaria**.

Il dataset utilizzato è il [Mushroom Dataset](https://www.kaggle.com/datasets) tratto da Kaggle (già sottoposto a data cleaning), e include le seguenti feature:

- **Cap Diameter**: diametro del cappello  
- **Cap Shape**: forma del cappello  
- **Gill Attachment**: tipo di attacco delle lamelle al gambo  
- **Gill Color**: colore delle lamelle  
- **Stem Height**: altezza del gambo  
- **Stem Width**: larghezza del gambo  
- **Stem Color**: colore del gambo  
- **Season**: stagione di raccolta  
- **Target Class**: etichetta binaria (0 = commestibile, 1 = velenoso)

---

## Istruzioni step-by-step

1. **Scaricare i file** dal repository GitHub, utilizzando il comando `git clone` oppure scaricando direttamente l’archivio `.zip`.

2. **Aprire il terminale di Conda** e posizionarsi nella cartella contenente i file del progetto  `MR_requirements.yml`.

3. **Creare l’ambiente Conda** utilizzando il comando:

   ```bash
   conda env create -f MR_requirements.yml
4. **Attivare l’ambiente Conda** conda activate mushrooms
5. **Eseguire**  il file principale con python main.py



## Librerie utilizzate

Il progetto è stato sviluppato in **Python** utilizzando le seguenti librerie principali:

| Libreria            | Utilizzo                                                                 |
|---------------------|--------------------------------------------------------------------------|
| `numpy`             | Calcoli numerici, distanze nella KNN custom, feature selection           |
| `pandas`            | Lettura e manipolazione del dataset                                      |
| `matplotlib`        | Creazione di grafici e visualizzazioni (ROC curve, confusion matrix)     |
| `seaborn`           | Visualizzazioni statistiche avanzate                                     |
| `collections`       | Uso di `Counter` per analisi distribuzione etichette                     |
| `pathlib`           | Gestione dei percorsi dei file                                           |
| `pickle`            | Salvataggio e caricamento di modelli addestrati                          |
| `joblib`            | Alternativa efficiente a pickle per modelli di grandi dimensioni         |
| `imbalanced-learn`  | Gestione di dataset sbilanciati tramite:                                 |
|                     | • `SMOTE` – oversampling sintetico della classe minoritaria              |
|                     | • `InstanceHardnessThreshold` – selezione campioni robusti della classe dominante |
|                     | • `NearMiss` – undersampling basato sulla distanza dai vicini più prossimi |
| `scikit-learn`      | Preprocessing, addestramento, valutazione e visualizzazione dei modelli  |

---

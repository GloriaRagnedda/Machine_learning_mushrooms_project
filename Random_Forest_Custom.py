from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from utils import plot_graphs, split_data, stats
import matplotlib.pyplot as plt
import pickle as pkl
from pathlib import Path



class RandomForestCustom(BaseEstimator):
    def __init__(self, 
                n_estimators=100,
                max_features='sqrt',
                max_samples=None,
                criterion='gini',
                bootstrap=True,
                max_depth=None,
                min_samples_split = 2,
                min_samples_leaf = 1,   
                random_state=None):
        
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf 
        self.random_state = random_state


        #ogni albero decisionale del random forest è allenato con una selezione random di features
        #questa lista conterrà gli indici delle colonne selezionate per ciascun albero
        self.columns = [None for _ in range(n_estimators)]

        #generatore di numeri random
        self.rng = np.random.RandomState(random_state)

        self.estimators = [
            DecisionTreeClassifier( 
                criterion=criterion,
                max_depth=max_depth,

                #settando il random state per ogni albero (altrimenti non sarebbe un RANDOM forest)
                random_state= self.rng.randint(0, 2**32-1, dtype=np.uint32),
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,

                #questo parametro fa si che i decision tree utilizzino tutte le features passate.
                #la selezione delle features è gestita nella funzione fit()
                #teoricamente avrei potuto passare tutte le colonne ai decision tree e lasciare 
                #che se ne occupi sklearn, ma a mio parere andrebbe contro lo spirito del progetto
                max_features=None 
                )
            for _ in range(n_estimators)
        ]


        
    def fit(self, X, y):
        #cast pandas DataFrame -> numpy array
        train_X = np.array(X)
        train_y = np.array(y)

        #rinominando numero righe e colonne (per maggiore leggibilità)
        tot_rows = train_X.shape[0]
        tot_cols = train_X.shape[1]

        """
        from sklearn:
            A random forest is a meta estimator that fits a number of decision tree
            classifiers on various sub-samples of the dataset and uses averaging to
            improve the predictive accuracy and control over-fitting.
            The sub-sample size is controlled with the `max_samples` parameter if
            `bootstrap=True` (default), otherwise the whole dataset is used to build
            each tree.

            max_samples : int or float, default=None
                If bootstrap is True, the number of samples to draw from X
                to train each base estimator.

                - If None (default), then draw `X.shape[0]` samples.
                - If int, then draw `max_samples` samples.
                - If float, then draw `max_samples * X.shape[0]` samples. Thus,
                  `max_samples` should be in the interval `(0.0, 1.0]`.
        """

        if self.bootstrap == True:

            match self.max_samples:
                case None:
                    n_rows = tot_rows

                case int(x):
                    n_rows = np.min([x, tot_rows])

                case float(x):
                    n_rows = np.min([tot_rows, int(tot_rows * x)])

                case _:
                    raise ValueError(f'unrecognized value "{self.max_samples}" for parameter "max_samples"')
        else:
            n_rows = tot_rows

        """ 
        from sklearn:   
            max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
                The number of features to consider when looking for the best split:

                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                  `round(max_features * n_features)` features are considered at each
                  split.
                - If "auto", then `max_features=sqrt(n_features)`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.
        """

        match self.max_features:
            case 'sqrt':
                n_cols = max(1, np.sqrt(tot_cols).astype(int))

            case 'log2':
                n_cols = max(1, np.log2(tot_cols).astype(int))

            case int(x):
                n_cols = np.min([x, tot_cols])
                
            case float(x):
                n_cols = np.min([tot_cols, int(tot_cols * x)])

            case None:
                n_cols = tot_cols

            case _:
                raise ValueError(f'unrecognized value "{self.max_features}" for parameter "max_features"')


        #iterando per ogni albero del random forest
        for i, dt in enumerate(self.estimators):

            #selezione random di sample
            if n_rows < tot_rows:
                rows = self.rng.choice(tot_rows, size=n_rows, replace=True)
            else:
                rows = np.arange(tot_rows)

            #selezione random di attributi
            if n_cols < tot_cols:
                columns = self.rng.choice(tot_cols, size=n_cols, replace=True)        
            else:
                columns = np.arange(tot_cols)

            #salvando gli indici delle colonne perché serviranno in predict()
            self.columns[i] = columns

            #subset di x e y per allenare l'aòberp
            partial_X = train_X[np.ix_(rows, columns)]
            partial_y = train_y[rows]
            dt.fit(partial_X, partial_y)


    def predict(self, test_X):
        proba = self.predict_proba(test_X) 

        #restituendo la classe con la probabilità più alta per ogni record
        return np.argmax(proba, axis=1)


    def predict_proba(self, test_X):
        #cast pandas DataFrame -> numpy array
        test_X = np.array(test_X)

        #chiamando predict_proba() su ogni albero (usando solo gli attributi di con cui è stato allenato)
        predictions = np.array([dt.predict_proba(test_X[:, columns]) for dt, columns in zip(self.estimators, self.columns)])

        #valore medio di tutte le probabilità calcolate da ogni albero per ciascuna classe
        return np.mean(predictions, axis=0)



def random_forest(X_train, X_test, y_train, y_test):

    rf_model = RandomForestCustom()
    
    rf_model.fit(X_train,y_train)
    y_pred = rf_model.predict(X_test)
    
    plot_graphs(y_test, y_pred)

    return rf_model, stats("Random Forest Custom", y_test, y_pred)



def random_forest_tuning(X_train, X_test, y_train, y_test):
    # griglia dei parametri
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'max_features': [2,4,6,8],
        'n_estimators': [10, 50, 100],     
    }

    #cercando se è già presente in memoria un dump di GridSearchCV per evitare di runnarlo ogni volta
    if Path("grid_search.pkl").exists():
        dump = pkl.load(Path("grid_search.pkl").open("rb"))
    else:

        # Inizializzazione Modello
        rf_model = RandomForestCustom(random_state=42)

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=10,  
            scoring='f1', 
            verbose=2,  
            n_jobs=4
        )

        # avviamento tunning
        grid_search.fit(X_train, y_train)

        dump = {
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "dataframe": pd.DataFrame(grid_search.cv_results_)
        }
        pkl.dump(dump, Path("grid_search.pkl").open("wb"))
    
    
    print(f"parametri migliori: {dump['best_params']}")

    dump["dataframe"].to_csv("random_forest_data.csv")

    # valutazione del modello
    y_pred = dump["best_model"].predict(X_test)

    plot_graphs(y_test, y_pred)
    tuning_plots(dump["dataframe"])

    return dump



def tuning_plots(df:pd.DataFrame):

    #selezionando solo le colonne che serviranno per i grafici
    df = df[["param_criterion", "param_n_estimators","param_max_features", "mean_test_score"]]

    #dividendo il dataframe in base al criterio usato dagli alberi decisionali (gini, entropy, log_loss)
    gini = df[df["param_criterion"] == "gini"]
    entropy = df[df["param_criterion"] == "entropy"]
    log_loss = df[df["param_criterion"] == "log_loss"]
    legend = ["gini", "entropy", "log_loss"]

    #-------------------------------------------------------------------------------------------#
    # Plot 1: come varia l'F1-score al variare di n_estimator (per vari valori di max_features) #
    #-------------------------------------------------------------------------------------------#

    max_features_for_each_subplot = df["param_max_features"].unique()
    n_subplots = len(max_features_for_each_subplot)
    
    #creando il frame principale (la larghezza dipende dal numero di subplots che inseriremo)
    fig = plt.figure(figsize=(n_subplots*6, 4))
    
    fig.suptitle("F1-score al variare di n_estimator (per vari valori di max_features)", fontsize=16)

    #creando i subplots
    axs = fig.subplots(1, n_subplots)

    #iterando sui subplots
    for i, max_features in enumerate(max_features_for_each_subplot):
        g = gini[gini["param_max_features"] == max_features].sort_values("param_n_estimators")
        e = entropy[entropy["param_max_features"] == max_features].sort_values("param_n_estimators")
        l = log_loss[log_loss["param_max_features"] == max_features].sort_values("param_n_estimators")

        #plot delle linee
        axs[i].plot(g["param_n_estimators"], g["mean_test_score"], color="r")
        axs[i].plot(e["param_n_estimators"], e["mean_test_score"], color="g")
        axs[i].plot(l["param_n_estimators"], l["mean_test_score"], color="b")

        axs[i].set_title(f"max_features: {max_features}")
        axs[i].set_xlabel("n_estimators")
        axs[i].set_ylabel("mean_test_score")
        axs[i].set_yticks(np.arange(0.7, 1, 0.1))
        axs[i].legend(legend)
    
    plt.show()

    #--------------------------------------------------------------------------------------------#
    # Plot 2: come varia l'F1-score al variare di max_features (per vari valori di n_estimators) #
    #--------------------------------------------------------------------------------------------#

    n_estimators_for_each_subplot =  df["param_n_estimators"].unique()
    n_subplots = len(n_estimators_for_each_subplot)
    
    #creando il frame principale (la larghezza dipende dal numero di subplots che inseriremo)
    fig = plt.figure(figsize=(n_subplots*6, 4))
    
    fig.suptitle("F1-score al variare di max_features (per vari valori di n_estimators)", fontsize=16)

    #creando i subplots
    axs = fig.subplots(1, n_subplots)

    #iterando sui subplots
    for i, n_estimators in enumerate(n_estimators_for_each_subplot):
        g = gini[gini["param_n_estimators"] == n_estimators].sort_values("param_max_features")
        e = entropy[entropy["param_n_estimators"] == n_estimators].sort_values("param_max_features")
        l = log_loss[log_loss["param_n_estimators"] == n_estimators].sort_values("param_max_features")

        #plot delle linee
        axs[i].plot(g["param_max_features"], g["mean_test_score"], color="r")
        axs[i].plot(e["param_max_features"], e["mean_test_score"], color="g")
        axs[i].plot(l["param_max_features"], l["mean_test_score"], color="b")

        axs[i].set_title(f"n_estimators: {n_estimators}")
        axs[i].set_xlabel("max_features")
        axs[i].set_ylabel("mean_test_score")
        axs[i].legend(legend)
    
    plt.show()
    
    #---------------------------------------------------------------------------#
    # Plot 3: 3D graph: riunendo le tre metriche analizzate in un unico grafico #
    #---------------------------------------------------------------------------#

    n_subplots = 3 #uno per ogni criterio (gini, entropy, log_loss)
    fig = plt.figure(figsize=(n_subplots*5, 5))
    
    fig.suptitle("3D graphs", fontsize=16)

    for i, criterion_df in enumerate([gini, entropy, log_loss]):
        ax = fig.add_subplot(1, n_subplots, i+1, projection="3d")
        pivot = criterion_df.pivot(index="param_max_features", columns="param_n_estimators",values="mean_test_score")
        
        x = pivot.columns.to_numpy()
        y = pivot.index.to_numpy()  
        X, Y = np.meshgrid(x, y)
        Z = pivot.to_numpy()

        surface = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='black')
        
        ax.set_xlabel("Number of Estimators", fontsize=12, labelpad=10)
        ax.set_ylabel("Max Depth", fontsize=12, labelpad=10)
        ax.set_zlabel("Mean F1-score", fontsize=12, labelpad=10)
        ax.set_title(legend[i], fontsize=14, pad=15)

        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="F1-score")

    plt.show()



def train_model(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)
    return random_forest(X_train, X_test, y_train, y_test)
    # Addestramento con tuning
    #random_forest_tuning(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    from Preprocessing import load_data_original, dummy

    #X, y = dummy()
    X, y = load_data_original()
    train_model(X,y)

    #X_train, X_test, y_train, y_test = split_data(X,y)
    #dump = random_forest_tuning(X_train, X_test, y_train, y_test)

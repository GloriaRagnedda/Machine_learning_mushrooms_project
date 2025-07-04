from itertools import chain, combinations
from functools import reduce
from sklearn.metrics import f1_score
import Preprocessing as pr
from utils import split_data, plot_graphs, no_op


def combos(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))


def best_preprocessing_combination(model):
    """
    Per ogni combinazione di tecniche di preprocessing,
    addestra un modello e calcola l'F1-score.
    Restituisce la combinazione ottimale e il relativo F1-score.
    """

    #turn off plotting
    model.plot_graphs = no_op

    fn_dict = {
        "Standardized": pr.standardized,
        "Feature Selection": pr.select_features,
        "Remove Outliers": pr.remove_outliers,
        "Bilanciamento": pr.bilanciamento
    }
    
    all_stats = []
    for combo in combos(fn_dict.keys()):
        name = " + ".join(combo)
        if name == "": name = "original dataset (no preprocessing)"
        data = reduce(lambda d, k: fn_dict[k](d), combo, pr.load_data_original())

        X, y = data
        _, stats = model.train_model(X, y)
        all_stats.append((name, data, stats))
    

    #turn plotting back on
    model.plot_graphs = plot_graphs

    #return the combination with highest f1 score
    all_stats.sort(key= lambda x: x[2]["F1-score"], reverse=True)

    return all_stats

    


if __name__ == "__main__":
    best_preprocessing_combination(None)
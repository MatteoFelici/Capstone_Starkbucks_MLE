import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score, f1_score, average_precision_score, \
    roc_auc_score, roc_curve



def analyse_cat_var(df, var, other_limit=0.01, tgt='tgt',
                    label=None, line_color='black'):
    """This function plots the average target value against a categorical
    variable, for all categories. It produces a plot with a black line
    representing the overall target mean.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    var : str
        Variable to analyse against the target. It has to be present into
        df.columns
    other_limit : float, default 0.001
        All categories with frequency below this threshold are binned into a
        "OTHER" category
    tgt : str, default "tgt"
        Target name. It has to be present into df.columns
    label : str, default None
        The title of the plot
    line_color : str, default 'black'
        The color of the line defining the overall target mean
    """

    # Select only important variables
    df_analysis = df[[var, tgt]].copy()
    df_analysis[var] = df_analysis[var].fillna('NULL')
    # Substitute categories with distribution less than "other_limit" with OTHER
    tmp = df_analysis[var].value_counts(1)
    others_list = tmp[tmp < other_limit].index.tolist()
    if len(others_list) > 1:
        df_analysis[var] = df_analysis[var].replace(others_list, 'OTHER')

    # Compute the overall target mean
    m = df_analysis[tgt].mean()

    plt.title(label)
        
    plt.xticks(rotation='vertical')
    df_barplot = df_analysis.groupby(var).agg({tgt: 'mean'}).reset_index()
    p = sns.barplot(x=var, y=tgt, data=df_barplot, ci=None)
    p.axhline(m, ls='--', color=line_color)




def analyse_num_var(df, var, q=(0, 0.25, 0.5, 0.75, 1), tgt='tgt',
                    label=None, line_color='black'):
    """This function plots the average target value against a numerical
    variable, binning the values against a list of given quantiles. It produces
    a plot with a black line representing the overall target mean.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    var : str
        Variable to analyse against the target. It has to be present into
        df.columns
    q : iterable of float, default (0, 0.25, 0.5, 0.75, 1)
        Quantiles to bin the nunmerical values
    tgt : str, default "tgt"
        Target name. It has to be present into df.columns
    label : str, default None
        The title of the plot
    line_color : str, default 'black'
        The color of the line defining the overall target mean
    """
    
    # Compute the overall target mean
    m = df[tgt].mean()

    plt.title(label)
        
    plt.xticks(rotation='vertical')
    tmp = df[[var, tgt]].copy()
    cuts = np.quantile(df[var].dropna(), q)
    tmp['aggr'] = pd.cut(tmp[var], bins=cuts, duplicates='drop',
                         include_lowest=True).astype(str)
    a = tmp.groupby('aggr').agg({tgt: 'mean'}).reset_index()
    a['ord'] = a['aggr'].apply(
        lambda x: float(x[1:].split(',')[0]) if x != 'nan' else np.nan
    )
    a.sort_values('ord', inplace=True)
    p = sns.barplot(x='aggr', y=tgt, data=a, ci=None)
    p.set_xlabel(var)
    p.axhline(m, ls='--', color=line_color)


def assess_model(y, y_prob, y_pred=None):
    """Given the truth labels and the model predictions (for a binary
    classification problem), this function provides a list of performance
    measures for a Machine Learning models. It also plots the ROC Curve and
    the Confusion/Classification Matrix.

    Parameters
    ----------
    y : list or numpy.array
        The ground truth labels
    y_prob : list or numpy.array
        Probabilities returned by the model
    y_pred : list or numpy.array, default None
        Predictions returned by the model. If None, these are calculated from
        y_prob with a 0.50 threshold

    Returns
    -------
    res : pandas.Datarame
        This dataset contains a list of the most used performance measures for
        the data provided

    """

    if y_pred is None:
        y_pred = pd.Series([1 if x >= 0.5 else 0 for x in y_prob])
        
    res = pd.DataFrame(
        {'value': [accuracy_score(y, y_pred),
                   balanced_accuracy_score(y, y_pred),
                   precision_score(y, y_pred), recall_score(y, y_pred),
                   f1_score(y, y_pred),
                   average_precision_score(y, y_prob),
                   roc_auc_score(y, y_prob)]
         },
        index=['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
               'average_precision', 'AUC']
    )

    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr)
    plt.show()

    display(pd.crosstab(y, y_pred, normalize='index'))

    return res

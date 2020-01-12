import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def analyse_cat_var(df, var, other_limit=0.01, tgt='tgt', label=None,
                    dist=True):
    """

    Parameters
    ----------
    df
    var
    other_limit
    tgt
    label
    dist

    Returns
    -------

    """

    # Select only important variables
    df_analysis = df[[var, tgt]].copy()
    df_analysis[var] = df_analysis[var].fillna('NULL')
    # Substitute categories with distribution less than "other_limit" with OTHER
    tmp = df_analysis[var].value_counts(1)
    others_list = tmp[tmp < other_limit].index.tolist()
    if len(others_list) > 1:
        df_analysis[var] = df_analysis[var].replace(others_list, 'OTHER')

    # If dist == True, plot the distribution
    if dist:
        ind = df_analysis[var].value_counts().index
        m = df_analysis[tgt].mean()

        ### Plot
        fig, axs = plt.subplots(figsize=(14, 4))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        fig.suptitle(label, fontsize=20)

        # Frequenza
        fig.add_subplot(121)
        plt.title('Distribution')
        plt.xticks(rotation='vertical')
        sns.countplot(x=var, data=df_analysis, order=ind)

        # Target
        plt.subplot(122)
        plt.title('Percentage of target')
        plt.xticks(rotation='vertical')
        df_barplot = df_analysis.groupby(var).agg({tgt: 'mean'}).reset_index()
        p = sns.barplot(x=var, y=tgt, data=df_barplot, ci=None, order=ind)
        p.axhline(m, ls='--', color='black')

    # Else plot only feature VS target
    else:
        m = df_analysis[tgt].mean()

        plt.title(label)
        plt.xticks(rotation='vertical')
        df_barplot = df_analysis.groupby(var).agg({tgt: 'mean'}).reset_index()
        p = sns.barplot(x=var, y=tgt, data=df_barplot, ci=None)
        p.axhline(m, ls='--', color='black')



def analyse_num_var(df, var, q=(0, 0.25, 0.5, 0.75, 1), tgt='tgt', label=None,
                    dist=True):
    """

    Parameters
    ----------
    df
    var
    q
    tgt
    label
    dist

    Returns
    -------

    """
    # If dist == True, plot the distribution
    if dist:
        m = df[tgt].mean()

        ### Plot
        fig, axs = plt.subplots(figsize=(14, 4))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        fig.suptitle(label, fontsize=20)

        # Frequenza
        fig.add_subplot(121)
        plt.title('Distribution')
        plt.xticks(rotation='vertical')
        inf, sup = np.quantile(df[var].dropna(), q=[0.01, 0.99])
        sns.distplot(
            df.loc[(df[var] >= inf) & (df[var] <= sup), var], kde=False, bins=10
        )

        # Target
        plt.subplot(122)
        plt.title('Percentage of target')
        plt.xticks(rotation='vertical')
        tmp = df[[var, tgt]].copy()
        cuts = np.quantile(df[var].dropna(), q)
        tmp['aggr'] = pd.cut(tmp[var], bins=cuts, duplicates='drop').astype(str)
        a = tmp.groupby('aggr').agg({tgt: 'mean'}).reset_index()
        a['ord'] = a['aggr'].apply(
            lambda x: float(x[1:].split(',')[0]) if x != 'nan' else np.nan
        )
        a.sort_values('ord', inplace=True)
        p = sns.barplot(x='aggr', y=tgt, data=a, ci=None)
        p.axhline(m, ls='--', color='black')


    # Else plot only feature VS target
    else:
        m = df[tgt].mean()

        plt.title(label)
        plt.xticks(rotation='vertical')
        tmp = df[[var, tgt]].copy()
        cuts = np.quantile(df[var].dropna(), q)
        tmp['aggr'] = pd.cut(tmp[var], bins=cuts, duplicates='drop').astype(str)
        a = tmp.groupby('aggr').agg({tgt: 'mean'}).reset_index()
        a['ord'] = a['aggr'].apply(
            lambda x: float(x[1:].split(',')[0]) if x != 'nan' else np.nan
        )
        a.sort_values('ord', inplace=True)
        p = sns.barplot(x='aggr', y=tgt, data=a, ci=None)
        p.axhline(m, ls='--', color='black')

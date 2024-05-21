import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

pd.set_option('display.max_columns', None)

color_palette = ["#1E1A0F", "#3F3128", "#644B35", "#A76F53", "#DCA98E", "#D7C9AC", "#689399", "#575735", "#343D22", "#152411"]

# Define the custom colormap
sns.set_palette(color_palette)
sns.set_style("whitegrid", {"grid.color": "#ffffff", "axes.facecolor": "w", 'figure.facecolor':'white'})

def get_var(df, var_name):
    globals()[var_name] = df
    return df

def cat_value_count(df: pd.DataFrame, 
                    col: str,  
                    reindex: list = None) -> pd.DataFrame:
    
    return (pd
            .DataFrame((df[col].value_counts(normalize=i) for i in [False, True]), index=['abs_count', 'norm_count'])
            .T
            .reindex(reindex)
            .assign(cumsum=lambda df_: df_.norm_count.cumsum(),
                    mean_target=df.groupby(col)["deposit"].mean())
            .pipe(lambda df_: print(f'This categorical predictor has {len(df_)} unique values\n\n', df_))
           )

def outlier_thresholds(df: pd.DataFrame, 
                       col: str, 
                       q1: float = 0.05, 
                       q3: float = 0.95):
    #1.5 as multiplier is a rule of thumb. Generally, the higher the multiplier,
    #the outlier threshold is set farther from the third quartile, allowing fewer data points to be classified as outliers
    
    return (df[col].quantile(q1) - 1.5 * (df[col].quantile(q3) - df[col].quantile(q1)),
            df[col].quantile(q3) + 1.5 * (df[col].quantile(q3) - df[col].quantile(q1)))

def loc_potential_outliers(df: pd.DataFrame,
                           col: str):
    
    low, high = outlier_thresholds(df, col)
    res = df.loc[(df[col] < low) | (df[col] > high)]
    print(f'Detected total of {len(res)} potential outliers based on 1.5xIQR')
    return res

def any_potential_outlier(df: pd.DataFrame, 
                          col: str) -> int:
    
    low, high = outlier_thresholds(df, col)
    if (df
        .loc[(df[col] > high) | (df[col] < low)]
        .any(axis=None)):
        return df.loc[(df[col] > high) | (df[col] < low)].shape[0]
    else:
        return 0
    
def delete_potential_outlier(df: pd.DataFrame,
                             col: str) -> pd.DataFrame:
    
    low, high = outlier_thresholds(df, col)
    df.loc[(df[col]>high) | (df[col]<low),col] = np.nan
    return df

def delete_potential_outlier_list(df: pd.DataFrame,
                                  cols: list) -> pd.DataFrame:

    for item in cols:
        df = delete_potential_outlier(df, item)
    return df

def plot_continuous(df: pd.DataFrame, 
                    col: str, 
                    title: str, 
                    symb: str):
    
    with sns.plotting_context(rc={"font":"Roboto", "palette":color_palette, "grid.linewidth":1.0, "font.size":12.0}):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5),gridspec_kw={"height_ratios": (.2, .8)})

        ax[0].set_title(title,fontsize=18)
        (df
         [[col]]
         .boxplot(ax=ax[0], vert=False))

        ax[0].set(yticks=[])

        (so
         .Plot(df,
               x=col)
         .add(so.Bars(color=color_palette[0]), so.Hist(bins=50))
         .label(x=col,
                y='Frequency',
                color=str.capitalize,)
         .on(ax[1])
         .plot()
        )

        plt.axvline(df[col].mean(), color="#708090", linestyle='--',linewidth=2.2, label='mean=' + str(np.round(df[col].mean(),1)) + symb)
        plt.axvline(df[col].median(), color="#4682B4", linestyle='--',linewidth=2.2, label='median='+ str(np.round(df[col].median(),1)) + symb)
        plt.axvline(df[col].mode()[0], color="#87CEFA", linestyle='--',linewidth=2.2, label='mode='+ str(np.round(df[col].mode()[0],1)) + symb)
        plt.legend(bbox_to_anchor=(1, 1.03), ncol=1, fontsize=12, fancybox=True, shadow=True, frameon=True)
        
        plt.tight_layout()
        plt.show()
        
        plt.close('all')
        
def plot_categorical(df: pd.DataFrame, 
                     col: str, 
                     new_index: list = None) -> so:

    df_to_plot = df[col].value_counts(normalize=True).to_frame().reset_index()

    if new_index is not None:
        df_to_plot = df_to_plot.set_index('index').reindex(new_index).reset_index()

    return (so
            .Plot(df_to_plot, x=col, y='index')
            .add(so.Bar(color=color_palette[8], alpha=.7, edgewidth=0))
            .theme({"axes.facecolor": "w", "grid.color": "#ffffff"})
            .label(x='',
                   y=col,
                   color=str.capitalize,
                   title=f'Normalised Count of {col}')
            .show()
           )

def plot_features_label(df: pd.DataFrame,
                        col: str,
                        new_index: list = None) -> so:
    
    df_to_plot = (df
                   .groupby([col])
                   .deposit #y
                   .value_counts(normalize=False)
                   .unstack(level=1)
                 )
    
    if new_index is not None:
        df_to_plot = df_to_plot.reindex(new_index)
    
    return (so
            .Plot((df_to_plot
                   .reset_index()
                   .rename(columns={0: '0', 1: '1'})
                   .melt(id_vars=col, var_name='decision', value_name='proportion')
                  ),
                  x='proportion',
                  y=col,
                  color='decision'
                 )
            .add(so.Bar(edgewidth=0),
                 so.Dodge())
            .theme({"axes.facecolor": "w", "grid.color": "#ffffff"})
            .label(x='',
                   y=col,
                   color=str.capitalize,
                   title=f'{col} v. Deposit Outcome')
            .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[8], color_palette[3]])})
            .show()
           )

def plot_label_features(df: pd.DataFrame,
                        col: str) -> so:
    
    df_to_plot = (df
                   .groupby([col])
                   .deposit #y
                   .value_counts(normalize=True)
                   .unstack(level=0)
                 )
        
    return (so
            .Plot((df_to_plot
                   .reset_index()
                   .melt(id_vars='deposit', var_name=col, value_name='proportion')
                  ),
                  x='proportion',
                  y='deposit',
                  color=col
                 )
            .add(so.Bar(),
                 so.Dodge(),
                 orient='y')
            .label(x='',
                   y='deposit',
                   color=str.capitalize,
                   title=f'{col} v. Deposit Outcome')
            .scale(y=so.Continuous().tick(at=[0, 1]))
            .theme({"axes.prop_cycle": matplotlib.cycler(color=color_palette + color_palette), "axes.facecolor": "w", "grid.color": "#ffffff"})
            .plot()
           )
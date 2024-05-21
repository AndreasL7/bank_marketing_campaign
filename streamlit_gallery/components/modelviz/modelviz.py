import gc
import streamlit as st
from joblib import load

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.inspection import PartialDependenceDisplay
from sklearn import metrics
import dtreeviz
import scikitplot
import shap

# Load the model
@st.cache_resource 
def load_model():
    primary_path = 'models/best_model_bank_marketing.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        raise Exception("Model not found in both primary and alternative directories!")
    
# Load the pipeline
@st.cache_resource
def load_pipeline():
    primary_path = 'models/best_pipeline_bank_marketing.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        raise Exception("Pipeline not found in both primary and alternative directories!")

# Load the trials object
@st.cache_resource
def load_trials():
    primary_path = 'data/processed/hyperopt_trials.csv'
    
    try:
        return pd.read_csv(primary_path)
    except FileNotFoundError:
        raise Exception("Trials object not found in both primary and alternative directories!")

optimal_threshold = 0.27

@st.cache_data
def read_data(file_name: str):
    
    return (pd
            .read_csv(file_name, sep=';')
            .rename(columns={'y': 'deposit'})
            )

def dataset_split(df):

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['deposit']),
                                                        df[['deposit']].values.ravel(),
                                                        test_size=0.2,
                                                        stratify=df[['deposit']].values.ravel(),
                                                        random_state=42)
    
    def get_X_train():
        return X_train
    
    def get_X_test():
        return X_test
    
    def get_y_train():
        return y_train
    
    def get_y_test():
        return y_test
    
    return get_X_train, get_X_test, get_y_train, get_y_test

def col_trans_feature_names(loaded_pipeline,
                            X_test):
    
    input_features = (loaded_pipeline
                      .named_steps['tweak_bank_marketing']
                      .transform(X_test)
                      .columns
                     )
    
    return (loaded_pipeline
            .named_steps['col_trans']
            .get_feature_names_out(input_features=input_features))
    
def get_selected_features(loaded_pipeline, 
                          X_test):
    
    support_mask = loaded_pipeline.named_steps['rfecv'].get_support()
    
    return np.array(col_trans_feature_names(loaded_pipeline, X_test))[support_mask]
    
def which_tree(loaded_model,
               loaded_pipeline,
               X_train,
               y_train,
               X_test):
    
    gc.enable()
    
    nth_tree = int(st.number_input(label="n-th tree", min_value=1, max_value=126, value=100))
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    class_names_ordered = label_encoder.classes_.astype('<U10')
    X_train_transformed = pd.DataFrame(loaded_pipeline.transform(X_train), columns=get_selected_features(loaded_pipeline, X_test))
    
    viz = dtreeviz.model(load_model(),
                         X_train=X_train_transformed.values,
                         y_train=y_encoded,
                         feature_names=get_selected_features(loaded_pipeline, X_test).tolist(),
                         target_name="deposit",
                         class_names=class_names_ordered,
                         tree_index=nth_tree)
    
    svg_data = viz.view(depth_range_to_display=[0,2], scale=1.5, orientation='TB').svg()
        
    html_data = f"""
    <html>
    <head>
        <title>Decision Tree Visualization</title>
    </head>
    <body>
        {svg_data}
    </body>
    </html>
    """
    del(
        nth_tree,
        label_encoder,
        y_encoded,
        class_names_ordered,
        X_train_transformed,
        viz,
    )
    gc.collect()
    
    return html_data

@st.cache_data
def plot_2d(df: pd.DataFrame, 
            x_col: str, 
            y_col: str, 
            _cm):
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df,
                    x=x_col,
                    y=y_col,
                    hue='tid',
                    palette=_cm,
                    ax=ax)
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.set_title(f"{y_col} vs {x_col}", fontsize=20, fontweight='bold', loc='left')
    
    return fig

@st.cache_data
def plot_3d_mesh(df: pd.DataFrame, 
                 x_col: str, 
                 y_col: str, 
                 z_col: str,
                 color_palette) -> go.Figure:
    
    colorscale = [[i/(len(color_palette)-1), color] for i, color in enumerate(color_palette)]
    
    fig = go.Figure(data=[go.Mesh3d(x=df[x_col], y=df[y_col], z=df[z_col],
                                    intensity=df[z_col]/df[z_col].min(),
                                    colorscale=colorscale,
                                    hovertemplate=f"{z_col}: %{{z}}<br>{x_col}: %{{x}}<br>{y_col}: "
                                                                "%{{y}}<extra></extra>")],
                   )
    
    fig.update_layout(
        title=dict(text=f'{y_col} vs {x_col}'),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col),
        width=350,
        margin=dict(r=20, b=10, l=30, t=50)
    )
    
    return fig

@st.cache_data
def plot_3d_scatter(df: pd.DataFrame, 
                    x_col: str, 
                    y_col: str, 
                    z_col: str, 
                    color_col: str, 
                    color_palette,
                    opacity: float=1,) -> go.Figure:
    
    colorscale = [[i/(len(color_palette)-1), color] for i, color in enumerate(color_palette)]
    fig = px.scatter_3d(data_frame=df, x=x_col, y=y_col, z=z_col, color=color_col, color_continuous_scale=colorscale, opacity=opacity)
    fig.update_layout(
        title=dict(text=f'{y_col} vs {x_col}'),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col),
        width=350,
    )
    
    return fig

def hyperparameter_tuning(loaded_trials, 
                          _cm,
                          color_palette):
    
    gc.enable()
    
    hyper2hr = loaded_trials
    
    st.table(hyper2hr
             .corr(method='spearman', numeric_only=True)
             .style
             .background_gradient(cmap=_cm)
             .set_sticky(axis=0))
    
    st.markdown("""
                Let's visualise using 2D and 3D plots!
                """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis_2d_choice = st.selectbox("x-axis", ["tid", "subsample", "colsample_bytree", "learning_rate", "max_depth", "min_child_weight", "reg_alpha", "reg_lambda", "gamma", "scale_pos_weight", "max_delta_step"], key="x_axis_2d_choice")
        y_axis_2d_choice = st.selectbox("y-axis", ["loss", "subsample", "colsample_bytree", "learning_rate", "max_depth", "min_child_weight", "reg_alpha", "reg_lambda", "gamma", "scale_pos_weight", "max_delta_step"], key="y_axis_2d_choice")
    
    with col2:
        st.pyplot(plot_2d(hyper2hr, x_axis_2d_choice, y_axis_2d_choice, _cm))     
    
    col3, col4 = st.columns(2)
    
    with col3:
        x_axis_3d_choice = st.selectbox("x-axis", ["reg_alpha", "tid", "subsample", "colsample_bytree", "learning_rate", "max_depth", "min_child_weight", "gamma", "reg_lambda", "scale_pos_weight", "max_delta_step"], key="x_axis_3d_choice")
        y_axis_3d_choice = st.selectbox("y-axis", ["reg_lambda", "tid", "subsample", "colsample_bytree", "learning_rate", "max_depth", "min_child_weight", "reg_alpha", "gamma", "scale_pos_weight", "max_delta_step"], key="y_axis_3d_choice")
        z_axis_3d_choice = st.selectbox("z-axis", ["loss", "subsample", "colsample_bytree", "learning_rate", "max_depth", "min_child_weight", "reg_alpha", "reg_lambda", "gamma", "scale_pos_weight", "max_delta_step"], key="z_axis_3d_choice")
        plot_choice = st.selectbox("View plot", ["Mesh", "Scatter"], key="plot_choice")
        
    with col4:
        if plot_choice == "Mesh":
            fig = plot_3d_mesh(hyper2hr,
                               x_axis_3d_choice, 
                               y_axis_3d_choice, 
                               z_axis_3d_choice,
                               color_palette)
            st.plotly_chart(fig)
        else:
            fig = plot_3d_scatter(hyper2hr,
                                  x_axis_3d_choice, 
                                  y_axis_3d_choice, 
                                  z_axis_3d_choice,
                                  color_col='loss',
                                  color_palette=color_palette)
            st.plotly_chart(fig)
            
    with st.expander("Insights"):
        st.markdown("""
                    1. Here, you can visualise how changes in the value of one hyperparameter may affect the other (or the loss).
                    """)
                    
    with st.expander("Click to view code"):
        code = """
        def hyperparameter_tuning(space: Dict[str, Union[float, int]],
                                  X_train: pd.DataFrame, 
                                  y_train: pd.Series,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series,
                                  early_stopping_rounds: int=50,
                                  metric:callable=accuracy_score) -> Dict[str, Any]:
            
            int_vals = ['max_depth', 'reg_alpha']
            space = {k: (int (val) if k in int_vals else val)
                     for k, val in space.items()}
            space['early_stopping_rounds'] = early_stopping_rounds
            
            model = XGBClassifier(**space, 
                                  eval_metric=['logloss'],
                                  n_jobs=-1)
            
            model.fit(X_train, 
                      y_train, 
                      eval_set=[(X_train, y_train),
                                (X_test, y_test)],
                      verbose=False)
            
            pred = model.predict(X_test)
            score = metric(y_test, pred)
            return {'loss': -score, 'status': STATUS_OK, 'model': model}
        
        options = {'max_depth': hp.quniform('max_depth', 3, 8, 1),
                   'min_child_weight': hp.loguniform('min_child_weight', -2, 3),
                   'subsample': hp.uniform('subsample', 0.7, 1),
                   'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                   'reg_alpha': hp.uniform('reg_alpha', 0, 10),
                   'reg_lambda': hp.uniform('reg_lambda', 1, 10),
                   'gamma': hp.loguniform('gamma', -10, 10),
                   'learning_rate': hp.loguniform('learning_rate', -7, 0),
                   'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 10),
                   'max_delta_step': hp.uniform('max_delta_step', 1, 10),
                   'random_state': 42
                  }
        
        trials = Trials()
        best_params_no_oversampler = fmin(fn=lambda space: hyperparameter_tuning(space, 
                                                           pipeline_oob_xgb.transform(X_train), 
                                                           y_train,
                                                           pipeline_oob_xgb.transform(X_test),
                                                           y_test),
                    space=options,
                    algo=tpe.suggest,
                    max_evals=2_000,
                    trials=trials)
        """
        st.code(code, language='python')
        
        plt.close('all')
        
        del(
            hyper2hr,
            col1,
            col2,
            col3,
            col4
        )
        gc.collect()
    
@st.cache_data
def get_tpr_fpr(probs, 
                y_truth):

    gc.enable()

    tp = np.sum((probs == 1) & (y_truth == 1))
    tn = np.sum((probs == 0) & (y_truth == 0))
    fp = np.sum((probs == 1) & (y_truth == 0))
    fn = np.sum((probs == 0) & (y_truth == 1))
    
    # Handle potential zero denominators
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    del tp, tn, fp, fn
    gc.collect()

    return tpr, fpr

@st.cache_data
def threshold_analysis(_loaded_model, _loaded_pipeline, X_test, y_test):

    gc.enable()

    # Convert y_test outside the loop
    y_test = np.where(y_test == 'no', 0, y_test)
    y_test = np.where(y_test == 'yes', 1, y_test)
    y_test = y_test.astype(int)

    vals = []
    for thresh in np.arange(0, 1, step=.05):
        probs = _loaded_model.predict_proba(_loaded_pipeline.transform(X_test))[:, 1]
        predictions = (probs > thresh).astype(int)
        tpr, fpr = get_tpr_fpr(predictions, y_test)
        val = [thresh, tpr, fpr]

        for metric in [metrics.accuracy_score, metrics.precision_score, metrics.recall_score, metrics.f1_score, metrics.roc_auc_score]:
            try:
                val.append(metric(y_test, predictions))
            except ZeroDivisionError:
                val.append(np.nan)  # or some other value indicating an undefined metric

        vals.append(val)

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.DataFrame(vals, columns=['thresh', 'tpr/rec', 'fpr', 'acc', 'prec', 'rec', 'f1', 'auc']).drop(columns='rec').set_index('thresh').plot(ax=ax, title='Threshold Metrics')
    st.pyplot(fig)
    
    with st.expander("Insights"):
        st.markdown("""
                    1. As we can observe above, the sweet spot for the threshold is around 0.27
                    """)
                    
    with st.expander("Click to view code"):
        code = """
        def threshold_analysis(loaded_model,
                               loaded_pipeline,
                               X_test, 
                               y_test):
            
            vals = []
            for thresh in np.arange(0, 1, step=.05):
                probs = loaded_model.predict_proba(loaded_pipeline.transform(X_test))[:, 1]
                tpr, fpr = get_tpr_fpr(probs > thresh, y_test)
                val = [thresh, tpr, fpr]
                y_test = np.where(y_test == 'no', 0, y_test)
                y_test = np.where(y_test == 'yes', 1, y_test)
                y_test = y_test.astype(int)
                for metric in [metrics.accuracy_score, metrics.precision_score,
                               metrics.recall_score, metrics.f1_score,
                               metrics.roc_auc_score]:
                    val.append(metric(y_test, probs > thresh))
                vals.append(val)
                
            fig, ax = plt.subplots(figsize=(8, 4))
            st.pyplot(pd.DataFrame(vals, columns=['thresh', 'tpr/rec', 'fpr', 'acc',
                                         'prec', 'rec', 'f1', 'auc'])
               .drop(columns='rec')
               .set_index('thresh')
               .plot(ax=ax, title='Threshold Metrics')
               .figure
            )
        """
        st.code(code, language='python')
        
    plt.close('all')
    
    del(
        y_test,
        val,
        vals,
        thresh,
        probs,
        predictions,
        tpr,
        fpr,
        fig,
        ax
    )
    gc.collect()

@st.cache_data
def confusion_matrix(_loaded_model, 
                     _loaded_pipeline, 
                     X_test, 
                     y_test,
                     _cm):
    
    gc.enable()
    
    y_test_encoded = LabelEncoder().fit_transform(y_test)
    y_prob = _loaded_model.predict_proba(_loaded_pipeline.transform(X_test))[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    #fig = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test_encoded, y_pred)).plot().ax_.figure
    cm = metrics.confusion_matrix(y_test_encoded, y_pred)
    display = metrics.ConfusionMatrixDisplay(cm)
    plot = display.plot(cmap=_cm)
    
    for row in plot.text_:
        for text in row:
            text.set_color('white')
        
    fig = plot.ax_.figure
    st.pyplot(fig)
    
    accuracy_score = metrics.accuracy_score(y_test_encoded, y_pred)
    precision_score = metrics.precision_score(y_test_encoded, y_pred)
    recall_score = metrics.recall_score(y_test_encoded, y_pred)
    f1_score = metrics.f1_score(y_test_encoded, y_pred)
    roc_auc_score = metrics.roc_auc_score(y_test_encoded, _loaded_model.predict_proba(_loaded_pipeline.transform(X_test))[:, 1])
    
    with st.expander("Insights"):
        st.markdown(f"""
                    1. Accuracy score: {accuracy_score}
                    2. Precision score: {precision_score}
                    3. Recall score: {recall_score}
                    4. F1 score: {f1_score}
                    5. ROC-AUC score: {roc_auc_score}
                    """)
    del(
        y_test_encoded,
        y_prob,
        y_pred,
        fig,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )
    gc.collect()

@st.cache_data
def cumulative_gain_curve(_loaded_model,
                          _loaded_pipeline,
                          X_test, 
                          y_test):
    
    gc.enable()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    y_probs = _loaded_model.predict_proba(_loaded_pipeline.transform(X_test))
    (scikitplot
     .metrics
     .plot_cumulative_gain(y_test, y_probs, ax=ax))
    ax.plot([0, (y_test == 1).mean(), 1], [0, 1, 1], label='Optimal Class 1')
    ax.set_ylim(0, 1.05)
    ax.annotate('Reach 60% of \nClass 1 (Yes) \nby contacting top 18%', xy=(.18, .6),
                xytext=(.7, .5), arrowprops={'color': 'k'})
    ax.legend();
    st.pyplot(fig)
    
    with st.expander("Insights"):
        st.markdown("""
                    1. The straight line from the bottom left to the top right represents the baseline scenario. 
                    This is the "random model" or the scenario where we contact clients without any order or model. 
                    Here, if we contact 20% of clients, we'd expect to reach roughly 20% of the subscribers.\n
                    
                    2. The curve represents the cumulative gain of the predictive model. 
                    The x-axis shows the proportion of clients targeted, starting with the ones 
                    most likely to subscribe to the deposit. The y-axis shows the proportion of 
                    all subscribers targeted up to the given x value.\n
                    
                    3. The line labeled 'Optimal Class 1' represents the best-case scenario. 
                    If we could perfectly rank all subscribers at the top, this is how our curve would look. 
                    In this case, we'd target all subscribers before targeting any non-subscribers.\n
                    
                    4. The annotation indicates a specific point on the cumulative gain curve. 
                    It says that by contacting the top 18% of clients, we would reach 60% of all subscribers. 
                    This highlights the model's value: we can reach a majority of subscribers by only targeting a minority of clients.\n
                    
                    5. The space between the model's cumulative gain curve and the diagonal line represents 
                    the added value from the model. The further away the curve is from the diagonal, 
                    the better our model is at ranking clients by their likelihood to subscribe.
                    """)
                    
    with st.expander("Click to view code"):
        code = """
        def cumulative_gain_curve(loaded_model,
                                  loaded_pipeline,
                                  X_test, 
                                  y_test):
            
            fig, ax = plt.subplots(figsize=(8, 4))
            y_probs = loaded_model.predict_proba(loaded_pipeline.transform(X_test))
            (scikitplot
             .metrics
             .plot_cumulative_gain(y_test, y_probs, ax=ax))
            ax.plot([0, (y_test == 1).mean(), 1], [0, 1, 1], label='Optimal Class 1')
            ax.set_ylim(0, 1.05)
            ax.annotate('Reach 60% of \nClass 1 (Yes) \nby contacting top 18%', xy=(.18, .6),
                        xytext=(.7, .5), arrowprops={'color': 'k'})
            ax.legend();
            st.pyplot(fig)
        """
        st.code(code, language='python')
        
    plt.close('all')
    
    del(
        fig,
        ax,
        y_probs,
    )
    gc.collect()
   
@st.cache_data
def feature_importance(_loaded_model,
                       _loaded_pipeline,
                       X_test, 
                       y_test,
                       color_palette):

    gc.enable()
    
    fig, ax = plt.subplots(figsize=(8, 12))
    st.pyplot(so
              .Plot((pd
                     .DataFrame(_loaded_model.feature_importances_, index=get_selected_features(_loaded_pipeline, X_test))
                     .rename(columns={0: "feature_importance"})
                     .sort_values(by="feature_importance", ascending=False)
                     .iloc[:8, :]
                     .reset_index()),
                    x='feature_importance',
                    y='index'
                   )
              .add(so.Bar(edgewidth=0))
              .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[8]])})
              .on(ax)
              .show()
    )
    
    with st.expander("Insights"):
        st.markdown("""
                    1. Gain measures the improvement in accuracy brought by a feature 
                    to the branches it is on (Average contribution of a feature to the model). 
                    Essentially, it is also the reduction in the training loss that results 
                    from adding a split on the feature.\n

                    2. A higher value of gain for a feature means it is more important for 
                    generating a prediction. It means changes in this feature's values have 
                    a more substantial effect on the output or prediction of the model. 
                    In this case, we have poutcome_success, contact_unknown, month, housing, and loan.
                    """)
                    
    with st.expander("Click to view code"):
        code = """
        def feature_importance(loaded_model,
                               loaded_pipeline,
                               X_test, 
                               y_test):

            fig, ax = plt.subplots(figsize=(8, 12))
            st.pyplot(so
                      .Plot((pd
                             .DataFrame(loaded_model.feature_importances_, index=get_selected_features(loaded_pipeline, X_test))
                             .rename(columns={0: "feature_importance"})
                             .sort_values(by="feature_importance", ascending=False)
                             .iloc[:8, :]
                             .reset_index()),
                            x='feature_importance',
                            y='index'
                           )
                      .add(so.Bar())
                      .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[8]])})
                      .on(ax)
                      .show()
            )
        )
        """
        st.code(code, language='python')
        
    plt.close('all')
    
    del(
        fig,
        ax
    )
    gc.collect()
     
@st.cache_data
def surrogate_models(_loaded_model,
                     _loaded_pipeline,
                     X_train,
                     X_test):
    
    gc.enable()
    
    y = _loaded_model.predict_proba(pd.DataFrame(_loaded_pipeline.transform(X_train), columns=get_selected_features(_loaded_pipeline, X_test)))[:,-1]
    X_train_transformed = pd.DataFrame(_loaded_pipeline.transform(X_train), columns=get_selected_features(_loaded_pipeline, X_test))
    
    sur_reg_sk = tree.DecisionTreeRegressor(max_depth=4)
    sur_reg_sk.fit(pd.DataFrame(X_train_transformed, columns=get_selected_features(_loaded_pipeline, X_test)),
                   y)

    viz = dtreeviz.model(sur_reg_sk,
                         X_train=X_train_transformed.values,
                         y_train=y,
                         feature_names=get_selected_features(_loaded_pipeline, X_test).tolist(),
                         target_name="deposit",)
    
    svg_data = viz.view(depth_range_to_display=[0,3], scale=1.0, orientation='LR').svg()
        
    html_data = f"""
    <html>
    <head>
    <title>Decision Tree Visualization</title>
    </head>
    <body>
    {svg_data}
    </body>
    </html>
    """
    
    with st.expander("Insights"):
        st.markdown("""
                    1. A surrogate model is a simple model to approximate the predictions 
                    of a more complex model. The main reason for using a surrogate model 
                    is to gain insight into the workings of the complex model, 
                    especially when the original model is a black-box (in this case, XGBoost). 
                    Here, we use DecisionTree due to its interpretability.\n

                    2. Surrogate model can also provide insights into interactions. 
                    Nodes that split on a different feature than a parent node often 
                    have an interaction. It looks like contact_unknown and day 
                    might have some interactions.
                    """)
                    
    with st.expander("Click to view code"):
        code = """
        def surrogate_models(loaded_model,
                             loaded_pipeline,
                             X_train,
                             X_test):
            
            sur_reg_sk = tree.DecisionTreeRegressor(max_depth=4)
            sur_reg_sk.fit(pd.DataFrame(loaded_pipeline.transform(X_train), columns=get_selected_features(loaded_pipeline, X_test)),
                           loaded_model.predict_proba(pd.DataFrame(loaded_pipeline.transform(X_train), columns=get_selected_features(loaded_pipeline, X_test)))[:,-1])
            
            dot_data = tree.export_graphviz(sur_reg_sk, out_file=None, 
                                            feature_names=get_selected_features(loaded_pipeline, X_test), 
                                            filled=True, rotate=True,)
            
            graph = graphviz.Source(dot_data)

            png_bytes = graph.pipe(format='png')
            st.image(png_bytes, caption="Decision Tree Surrogate Model")
        """
        st.code(code, language='python')
    
    del(
        y,
        X_train_transformed,
        sur_reg_sk,
        viz
    )
    gc.collect()
        
    return html_data
     
def xgbfir(_loaded_pipeline,
           X_train,
           y_train,
           X_test,
           _cm,
           color_palette):
    
    gc.enable()
    
    xgbfir_choice = st.selectbox("Depth", ["Interaction Depth 0", "Interaction Depth 1", "Interaction Depth 2",], key="xgbfir_choice")
    sort_by = st.selectbox("Sort by", ["Average Rank", "Gain", "FScore", "wFScore", "Average wFScore", "Average Gain", "Expected Gain", "Gain Rank", "FScore Rank", "wFScore Rank", "Avg wFScore Rank", "Avg Gain Rank", "Expected Gain Rank", "Average Rank", "Average Tree Index", "Average Tree Depth"], key="xgbfir_sortby")
    
    if xgbfir_choice == "Interaction Depth 0":   
        xgbfir = pd.read_excel('data/processed/xgbfir.xlsx')
    elif xgbfir_choice == "Interaction Depth 1":
        xgbfir = pd.read_excel('data/processed/xgbfir.xlsx', sheet_name='Interaction Depth 1')
    elif xgbfir_choice == "Interaction Depth 2":
        xgbfir = pd.read_excel('data/processed/xgbfir.xlsx', sheet_name='Interaction Depth 2')
    
    st.table(pd.DataFrame(xgbfir
                          .sort_values(by=sort_by)
                          .round(1)
                         ))
    st.markdown("Let's view the correlation between features to further understand Interaction Depth 1")
    
    post_col_trans = pd.DataFrame((_loaded_pipeline
                                   .named_steps['col_trans']
                                   .transform((_loaded_pipeline
                                               .named_steps['tweak_bank_marketing']
                                               .transform(X_train)
                                               )
                                              )
                                   ), columns=col_trans_feature_names(_loaded_pipeline, X_test)
                                  )
    
    st.table((post_col_trans
              .assign(deposit=y_train)
              .corr(method='spearman', numeric_only=True)
              .loc[:, ['month', 'day', 'balance', 'contact_telephone', 'contact_unknown', 'poutcome_success', 'poutcome_unknown', 'age', 'pdays']]
              .style
              .background_gradient(cmap=_cm)
              .format('{:.2f}')
              .set_sticky(axis=0)
             ))
    
    st.markdown("Let's look at some plots to see if there is any nonlinearity captured by the interaction")
    xgbfir_plot_xaxis = st.selectbox("x-axis", ["month", "day", "balance", "contact_telephone", "contact_unknown", "poutcome_success", "poutcome_unknown", "age", "pdays"], key="xgbfir_plot_xaxis")
    xgbfir_plot_yaxis = st.selectbox("y-axis", ["age", "campaign", "pdays", "previous", "balance", "job_blue-collar", "job_management", "job_self-employed", "job_services", "job_technician", "job_unemployed", "marital_married", "marital_single", "contact_telephone", "contact_unknown", "poutcome_success", "poutcome_unknown", "education", "day", "month", "default", "housing", "loan", "balance_pos", "pdays_contacted", "previous_contacted"], key="xgbfir_plot_yaxis")
    
    st.pyplot((so
               .Plot(post_col_trans.assign(deposit=y_train), x=xgbfir_plot_xaxis, y=xgbfir_plot_yaxis, color='deposit')
               .add(so.Dots(alpha=.9, pointsize=2), so.Jitter(x=.7, y=1))
               .add(so.Line(), so.PolyFit())
               .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[8], color_palette[6]]), "axes.facecolor": "w", "grid.color": "#ffffff"})
               .show()
               )
    )
    
    with st.expander("Insights"):
        st.markdown("""
                    1. Here, the default Sort by dropdown is set to Average Rank, which
                    is a metric that gives a holistic view of a feature's 
                    (or feature pair's) importance across various criteria.
                     
                    2. How to calculate Average Rank? For each feature or feature pair, 
                    the ranks across these different metrics (Gain, FScore, wFScore, etc.) 
                    are averaged to compute the "Average Rank". This provides a unified rank 
                    that takes into account the various ways a feature might be considered "important".
                    
                    3. In short, a lower Average Rank indicates higher importance. 
                    If a feature consistently ranks high (i.e., is of top importance) 
                    across different metrics, its average rank will be lower 
                    (which is better).
                    """)
                    
    with st.expander("Click to view code"):
        code = """
        def xgbfir(loaded_pipeline,
                   X_train,
                   y_train,
                   X_test):
            
            xgbfir_choice = st.selectbox("Depth", ["Interaction Depth 0", "Interaction Depth 1", "Interaction Depth 2",], key="xgbfir_choice")
            sort_by = st.selectbox("Sort by", ["Average Rank", "Gain", "FScore", "wFScore", "Average wFScore", "Average Gain", "Expected Gain", "Gain Rank", "FScore Rank", "wFScore Rank", "Avg wFScore Rank", "Avg Gain Rank", "Expected Gain Rank", "Average Rank", "Average Tree Index", "Average Tree Depth"], key="xgbfir_sortby")
            
            if xgbfir_choice == "Interaction Depth 0":   
                xgbfir = pd.read_excel('xgbfir.xlsx')
            elif xgbfir_choice == "Interaction Depth 1":
                xgbfir = pd.read_excel('xgbfir.xlsx', sheet_name='Interaction Depth 1')
            elif xgbfir_choice == "Interaction Depth 2":
                xgbfir = pd.read_excel('xgbfir.xlsx', sheet_name='Interaction Depth 2')
            
            st.table(pd.DataFrame(xgbfir
                                  .sort_values(by=sort_by)
                                  .round(1)
                                 ))
            st.markdown("Let's view the correlation between features to further understand Interaction Depth 1")
            
            post_col_trans = pd.DataFrame((loaded_pipeline
                                           .named_steps['col_trans']
                                           .transform((loaded_pipeline
                                                       .named_steps['tweak_bank_marketing']
                                                       .transform(X_train)
                                                       )
                                                      )
                                           ), columns=col_trans_feature_names(loaded_pipeline, X_test)
                                          )
            
            st.table((post_col_trans
                      .assign(deposit=y_train)
                      .corr(method='spearman', numeric_only=True)
                      .loc[:, ['month', 'day', 'balance', 'contact_telephone', 'contact_unknown', 'poutcome_success', 'poutcome_unknown', 'age', 'pdays']]
                      .style
                      .background_gradient(cmap=cm)
                      .format('{:.2f}')
                      .set_sticky(axis=0)
                     ))
            
            st.markdown("Let's look at some plots to see if there is any nonlinearity captured by the interaction")
            xgbfir_plot_xaxis = st.selectbox("x-axis", ["month", "day", "balance", "contact_telephone", "contact_unknown", "poutcome_success", "poutcome_unknown", "age", "pdays"], key="xgbfir_plot_xaxis")
            xgbfir_plot_yaxis = st.selectbox("y-axis", ["age", "campaign", "pdays", "previous", "balance", "job_blue-collar", "job_management", "job_self-employed", "job_services", "job_technician", "job_unemployed", "marital_married", "marital_single", "contact_telephone", "contact_unknown", "poutcome_success", "poutcome_unknown", "education", "day", "month", "default", "housing", "loan", "balance_pos", "pdays_contacted", "previous_contacted"], key="xgbfir_plot_yaxis")
            
            st.pyplot((so
                       .Plot(post_col_trans.assign(deposit=y_train), x=xgbfir_plot_xaxis, y=xgbfir_plot_yaxis, color='deposit')
                       .add(so.Dots(alpha=.9, pointsize=2), so.Jitter(x=.7, y=1))
                       .add(so.Line(), so.PolyFit())
                       .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[8], color_palette[6]])})
                       .show()
                       )
            )
        """
        st.code(code, language='python')
        
    del(
        xgbfir_choice,
        sort_by,
        post_col_trans,
        xgbfir_plot_xaxis,
        xgbfir_plot_yaxis
    )
    gc.collect()
 
@st.cache_data
def shapley(_loaded_model, 
            _loaded_pipeline, 
            X_test,
            color_palette):
    
    gc.enable()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    shap_ex = shap.TreeExplainer(_loaded_model)
    
    keys_to_search = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", 
                      "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]
    
    missing_keys = [key for key in keys_to_search if not st.session_state.get(key)]
    
    if missing_keys:
        st.warning(f"Please input the data for: {', '.join(missing_keys)} in the Insight from data page!")
    else:

        keys_to_search = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", 
                          "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]
        
        client_data = {}
        for key in keys_to_search:
            if key in st.session_state:
                client_data[key] = [st.session_state[key]]
            else:
                client_data[key] = [None]
        
        df_client_data = pd.DataFrame(client_data)
        vals = shap_ex(pd.DataFrame(_loaded_pipeline.transform(df_client_data), columns=get_selected_features(_loaded_pipeline, X_test)))
        shap_df = pd.DataFrame(vals.values, columns=get_selected_features(_loaded_pipeline, X_test))
        
        st.table(shap_df)
        
        st.markdown("Waterfall Plot SHAP values")
        
        # Default SHAP colors
        default_pos_color = "#ff0051"
        default_neg_color = "#008bfb"
        
        # Custom colors
        positive_color = color_palette[8]
        negative_color = color_palette[3]
        
        fig, ax = plt.subplots()
        fig = shap.plots.waterfall(vals[0], show=False)
        
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if (isinstance(fcc, matplotlib.patches.FancyArrow)):
                    if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                        fcc.set_facecolor(positive_color)
                    elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                        fcc.set_color(negative_color)
                elif (isinstance(fcc, plt.Text)):
                    if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                        fcc.set_color(positive_color)
                    elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                        fcc.set_color(negative_color)
                    
        st.pyplot(fig)
        
        st.markdown("Force Plot SHAP values")
        def _force_plot_html():
            force_plot = shap.plots.force(base_value=vals.base_values,
                                          shap_values=vals.values[0,:],
                                          features=get_selected_features(_loaded_pipeline, X_test),
                                          matplotlib=False,
                                          show=False,
                                          plot_cmap=[positive_color, negative_color],)
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            return shap_html
        
        shap_html = _force_plot_html()
        
        st.components.v1.html(shap_html, width=800, height=100)
        
        with st.expander("Insights"):
            st.markdown("""
                        1. Take E[f(X)] as the baseline value. \n
                        2. Gradually adds the values on the bar to obtain f(x). \n
                        3. When the value is less than 0, it explains the deposit outcome "No". \n
                        4. You might received the opposite outcome in the prediction page, 
                        but that final outcome is because the decision is already affected 
                        by the threshold value we set for our model. \n
                        5. Force Plot is merely a flattened version of our Waterfall plot.
                        """)
                        
        with st.expander("Click to view code"):
            code = """
            def shapley(loaded_model, 
                        loaded_pipeline, 
                        X_test):
                
                shap_ex = shap.TreeExplainer(loaded_model)
                
                # List of keys you want to search for
                keys_to_search = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", 
                                  "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]
                
                # Check if the specific keys in st.session_state are empty
                missing_keys = [key for key in keys_to_search if not st.session_state.get(key)]
                
                if missing_keys:
                    st.warning(f"Please input the data for: {', '.join(missing_keys)} in the Insight from data page!")
                else:

                    keys_to_search = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", 
                                      "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]
                    
                    client_data = {}
                    for key in keys_to_search:
                        if key in st.session_state:
                            client_data[key] = [st.session_state[key]]
                        else:
                            client_data[key] = [None]
                    
                    df_client_data = pd.DataFrame(client_data)
                    vals = shap_ex(pd.DataFrame(loaded_pipeline.transform(df_client_data), columns=get_selected_features(loaded_pipeline, X_test)))
                    shap_df = pd.DataFrame(vals.values, columns=get_selected_features(loaded_pipeline, X_test))
                    
                    st.table(shap_df)
                    
                    st.markdown("Waterfall Plot SHAP values")
                    
                    # Default SHAP colors
                    default_pos_color = "#ff0051"
                    default_neg_color = "#008bfb"
                    
                    # Custom colors
                    positive_color = color_palette[8]
                    negative_color = color_palette[3]
                    
                    fig, ax = plt.subplots()
                    fig = shap.plots.waterfall(vals[0], show=False)
                    
                    for fc in plt.gcf().get_children():
                        for fcc in fc.get_children():
                            if (isinstance(fcc, matplotlib.patches.FancyArrow)):
                                if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                                    fcc.set_facecolor(positive_color)
                                elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                                    fcc.set_color(negative_color)
                            elif (isinstance(fcc, plt.Text)):
                                if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                                    fcc.set_color(positive_color)
                                elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                                    fcc.set_color(negative_color)
                                
                    st.pyplot(fig)
                    
                    st.markdown("Force Plot SHAP values")
                    def _force_plot_html():
                        force_plot = shap.plots.force(base_value=vals.base_values,
                                                      shap_values=vals.values[0,:],
                                                      features=get_selected_features(loaded_pipeline, X_test),
                                                      matplotlib=False,
                                                      show=False,
                                                      plot_cmap=[positive_color, negative_color],)
                        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                        return shap_html
                    
                    shap_html = _force_plot_html()
                    
                    st.components.v1.html(shap_html, width=800, height=100)
            """
            st.code(code, language='python')
    
    plt.close('all')
    
    gc.collect()
   
@st.cache_data
def beeswarm_plot(_loaded_model,
                  _loaded_pipeline,
                  X_test,
                  _cm):
    
    gc.enable()
    
    shap_ex = shap.TreeExplainer(_loaded_model)
    
    X_test_vals = shap_ex(pd.DataFrame(_loaded_pipeline.transform(X_test), columns=get_selected_features(_loaded_pipeline, X_test)))
    
    fig, ax = plt.subplots()
    fig = shap.plots.beeswarm(X_test_vals, color=_cm)
    st.pyplot(fig)
    
    with st.expander("Insights"):
        st.markdown("""
                    1. The x-axis represents the SHAP value. A SHAP value is a number 
                    that indicates how much a particular feature changed the model's 
                    prediction for an individual data point compared to the model's 
                    baseline prediction. Positive SHAP values push the prediction higher, 
                    while negative values pull it lower.
                    
                    2. The y-axis represents each feature contributing to the prediction,
                    with the most influential feature at the top.
                    
                    3. Each dot in the plot represents a specific data point from the test dataset. 
                    The horizontal position of the dot shows whether that feature increased 
                    (to the right) or decreased (to the left) the prediction for that data point.
                    
                    4. Areas with more dots show where the feature had a similar impact on 
                    many data points. Sparse areas indicate that the feature's influence 
                    was more unique to specific data points.
                    
                    5. For a given feature, if most dots lie to the right of the center, 
                    it means that this feature tends to increase the prediction when present 
                    (or has a high value). Conversely, if dots predominantly lie to the left, 
                    the feature tends to decrease the prediction.
                    """)
                    
    with st.expander("Click to view code"):
        code = """
        def beeswarm_plot(loaded_model,
                          loaded_pipeline,
                          X_test):
            
            shap_ex = shap.TreeExplainer(loaded_model)
            
            X_test_vals = shap_ex(pd.DataFrame(loaded_pipeline.transform(X_test), columns=get_selected_features(loaded_pipeline, X_test)))
            
            fig, ax = plt.subplots()
            fig = shap.plots.beeswarm(X_test_vals, color=cm)
            st.pyplot(fig)
        """
        st.code(code, language='python')
    
    plt.close('all')
    
    del(
        shap_ex,
        X_test_vals,
        fig,
        ax
    )
    gc.collect()
    
def ice_pdp(_loaded_model, 
            _loaded_pipeline, 
            X_train,
            X_test,
            color_palette):
    
    gc.enable()
    
    icepdp_choice_plot1 = st.selectbox("Feature 1", ['housing', 'age', 'campaign', 'pdays', 'previous', 'balance', 'job_blue-collar', 'job_unemployed', 'marital_married', 'marital_single', 'contact_unknown', 'poutcome_success', 'poutcome_unknown', 'day', 'month', 'loan', 'pdays_contacted', 'previous_contacted'], key="icepdp_choice_plot1")
    icepdp_choice_plot2 = st.selectbox("Feature 2", ['loan', 'age', 'campaign', 'pdays', 'previous', 'balance', 'job_blue-collar', 'job_unemployed', 'marital_married', 'marital_single', 'contact_unknown', 'poutcome_success', 'poutcome_unknown', 'day', 'month', 'housing', 'pdays_contacted', 'previous_contacted'], key="icepdp_choice_plot2")
    fig, ax = plt.subplots()
    PartialDependenceDisplay.from_estimator(_loaded_model, 
                                            pd.DataFrame(_loaded_pipeline.transform(X_train), columns=get_selected_features(_loaded_pipeline, X_test)),
                                            features=[icepdp_choice_plot1, icepdp_choice_plot2],
                                            centered=True,
                                            kind='both',
                                            ax=ax,
                                            ice_lines_kw={"color": color_palette[8]},
                                            pd_line_kw={"color": color_palette[3]})
    st.pyplot(fig)
    
    fig.savefig("ice_pdp.png", format="png", dpi=300)
        
    with st.expander("Insights"):
        st.markdown("""
                    1. The bold line represents the PDP. It averages out the individual ICE lines.
                    2. The faint lines in the background represent ICE lines for individual data points.
                    3. For example, above you can gauve how, on average, having housing or personal loans influence the model's prediction.
                    4. Simultaneously, you can see how these factors affect predictions for individual data points and if there's any variability among them.
                    """)
                    
    with st.expander("Click to view code"):
        code = """
        def ice_pdp(loaded_model, 
                    loaded_pipeline, 
                    X_train,
                    X_test):
            
            icepdp_choice_plot1 = st.selectbox("Feature 1", ['housing', 'age', 'campaign', 'pdays', 'previous', 'balance', 'job_blue-collar', 'job_unemployed', 'marital_married', 'marital_single', 'contact_unknown', 'poutcome_success', 'poutcome_unknown', 'day', 'month', 'loan', 'pdays_contacted', 'previous_contacted'], key="icepdp_choice_plot1")
            icepdp_choice_plot2 = st.selectbox("Feature 2", ['loan', 'age', 'campaign', 'pdays', 'previous', 'balance', 'job_blue-collar', 'job_unemployed', 'marital_married', 'marital_single', 'contact_unknown', 'poutcome_success', 'poutcome_unknown', 'day', 'month', 'housing', 'pdays_contacted', 'previous_contacted'], key="icepdp_choice_plot2")
            fig, ax = plt.subplots()
            PartialDependenceDisplay.from_estimator(loaded_model, 
                                                    pd.DataFrame(loaded_pipeline.transform(X_train), columns=get_selected_features(loaded_pipeline, X_test)),
                                                    features=[icepdp_choice_plot1, icepdp_choice_plot2],
                                                    centered=True,
                                                    kind='both',
                                                    ax=ax,
                                                    ice_lines_kw={"color": color_palette[8]},
                                                    pd_line_kw={"color": color_palette[3]})
            st.pyplot(fig)
        """
        st.code(code, language='python')
        
    plt.close('all')
    
    del(
        icepdp_choice_plot1,
        icepdp_choice_plot2,
        fig,
        ax
    )
    gc.collect()

def main():
    
    gc.enable()
    
    matplotlib.font_manager.fontManager.addfont('streamlit_gallery/utils/arial/arial.ttf')
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    shap.initjs()
    
    color_palette = ["#1E1A0F", "#3F3128", "#644B35", "#A76F53", "#DCA98E", "#D7C9AC", "#689399", "#575735", "#343D22", "#152411"]

    # Define the custom colormap
    cmap_name = 'custom_palette'
    cm = plt.cm.colors.LinearSegmentedColormap.from_list(cmap_name, color_palette, N=len(color_palette))
    sns.set_palette(color_palette)
    sns.set_style("white", {"grid.color": "#ffffff", "axes.facecolor": "w", 'figure.facecolor':'white'})
    
    df = read_data("data/raw/bank-full.csv")
    get_X_train, get_X_test, get_y_train, get_y_test = dataset_split(df)
    
    X_train = get_X_train()
    y_train = get_y_train()
    X_test = get_X_test()
    y_test = get_y_test()
    
    st.title("A Peek into the Model")
    st.subheader("XGBoost!")
    
    st.markdown("""
                For predictive analytics on the bank marketing dataset, 
                I gravitated towards XGBoost, an advanced implementation of 
                gradient boosted trees renowned for its speed and performance. 
                This dataset, dotted with categorical features and class imbalances, 
                found a fitting ally in XGBoost, which deftly handles sparse data and 
                offers built-in mechanisms like scale_pos_weight for imbalance.
                
                Beyond its innate ability to manage such challenges, 
                XGBoost's incorporation of L1 and L2 regularization safeguards 
                against overfitting, while its capacity for parallel computing 
                ensures swift model training. Furthermore, XGBoost consistent top-tier 
                performance in various machine learning arenas and competitions underscores its prowess, 
                making it an optimal choice for the bank marketing dataset.
                
                In this section, I will walk you through various processes such as
                hyperparameter tuning, feature importances, interactions, and
                also understand how a particular feature impact the prediction.
                Let's get started!
                """)
        
# =============================================================================
#     st.subheader("Which Tree?")
#     st.markdown("Let’s look at what the 100-th tree looks like")
#     
#     st.components.v1.html(which_tree(load_model(),
#                                       load_pipeline(),
#                                       X_train,
#                                       y_train,
#                                       X_test), 
#                           width=800, 
#                           height=500, 
#                           scrolling=True)
#     
#     st.image("which_tree.svg")
# =============================================================================
    
    st.subheader("Hyperparameter Tuning")
    st.markdown("Correlation Analysis between XGBoost Hyperparameter")
    
    hyperparameter_tuning(load_trials(), 
                          cm,
                          color_palette)
            
    st.subheader("Optimal Threshold")
    st.markdown("Analysing Precision Recall Tradeoff")
    
    threshold_analysis(load_model(), 
                       load_pipeline(), 
                       X_test, 
                       y_test)
    
    st.subheader("Confusion Matrix")
    st.markdown("After adjusting to the optimal threshold value, below is our confusion matrix")
    
    confusion_matrix(load_model(), 
                     load_pipeline(), 
                     X_test, 
                     y_test,
                     cm)
    
    st.subheader("Cumulative Gains Curve")
    st.markdown("This plot visualizes the cumulative gain of a predictive model, in comparison to a random model and an optimal model.")
    
    cumulative_gain_curve(load_model(), 
                          load_pipeline(),
                          X_test,
                          y_test)
    
    st.subheader("Feature Importances")
    st.markdown("Feature Importances help us understand which features are more influential in making a prediction.")
    
    feature_importance(load_model(),
                       load_pipeline(),
                       X_test, 
                       y_test,
                       color_palette)
    
# =============================================================================
#     st.subheader("Surrogate Models")
#     st.markdown("Surrogate models are simplified versions of complex models, designed to be more interpretable.")
#     
#     st.components.v1.html(surrogate_models(load_model(),
#                                            load_pipeline(),
#                                            X_train,
#                                            X_test), 
#                           width=800, 
#                           height=600, 
#                           scrolling=True)
#     
#     st.image("surrogate_model_tree.svg")    
# =============================================================================

    st.subheader("xgbfir (Feature Interactions Reshaped)")
    st.write("xgbfir is a tool that helps us in understanding interaction effects in our XGBoost models. Specifically, it ranks and visualizes feature interactions based on their importance.")
    st.write("Interaction Depth 0 means we're looking at the main effects of individual features, without considering their interactions with other features. Navigate to Interaction Depth 1 to understand how pairs of features interact with each other in the model and influence the model's predictions.")
    
    xgbfir(load_pipeline(),
           X_train,
           y_train,
           X_test,
           cm,
           color_palette)
    
    st.subheader("Waterfall and Force Plot")
    st.write("This section displays the plots for SHAP value specific to your input on **Prediction and Modelling** page.")
    st.write("SHAP breaks down a prediction into parts, each representing a feature (like age, income, or location). It then tells us how much each feature contributed to the prediction, whether it increased or decreased the prediction, and by how much.")
    st.write("In essence, SHAP helps us peek inside the 'black box' of complex models, making them more transparent and understandable.")
    
    shapley(load_model(), 
            load_pipeline(), 
            X_test,
            color_palette)
    
    st.subheader("Beeswarm Plot for Test Data SHAP values")
    st.markdown("""This section displays the plots for SHAP values for our Test Data.
                provides insights into the impact of features on model predictions.
                Specifically, it lets us understand both global (entire model) and 
                local (individual predictions) interpretations simultaneously.""")
                
    beeswarm_plot(load_model(),
                  load_pipeline(),
                  X_test,
                  cm)
    
    st.subheader("ICE and PDP")
    st.write("These plots help us understand the relationship between specific features and model predictions.")
    st.write("The PDP shows the average prediction of the model as a function of specific feature(s), while keeping all other features constant. Meanwhile, the ICE plots show the effect of a feature on the prediction for individual data points.")
    st.write("Each line in an ICE plot represents an individual data point from the dataset. The line tracks how the model's prediction would change for that specific data point as the feature changes. In short, PDP is the average effect of the ICE plots.")
    
# =============================================================================
#     ice_pdp(load_model(), 
#             load_pipeline(), 
#             X_train,
#             X_test,
#             color_palette)
# =============================================================================
    
    st.image("img/ice_pdp.png")
    
    plt.close('all')
    
    del(
        cm,
        cmap_name,
        df,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    gc.collect()
    
if __name__ == "__main__":
    main()
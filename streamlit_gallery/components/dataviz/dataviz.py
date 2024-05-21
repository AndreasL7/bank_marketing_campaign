import requests
import gc
import streamlit as st
from streamlit_lottie import st_lottie

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.preprocessing import LabelEncoder
from scipy.stats.contingency import association

from ...utils.helpers import (
    plot_categorical,
    plot_features_label,
)

import warnings

@st.cache_data
def load_lottie_url(url: str):

    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


@st.cache_data
def read_data(file_name: str):

    return (pd
            .read_csv(file_name, sep=';')
            .rename(columns={'y': 'deposit'}))

def main():

    gc.enable()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    matplotlib.font_manager.fontManager.addfont(
        'streamlit_gallery/utils/arial/arial.ttf')
    plt.rcParams['font.sans-serif'] = ['Arial']

    color_palette = ["#1E1A0F", "#3F3128", "#644B35", "#A76F53",
                     "#DCA98E", "#D7C9AC", "#689399", "#575735", "#343D22", "#152411"]

    # Define the custom colormap
    cmap_name = 'custom_palette'
    cm = plt.cm.colors.LinearSegmentedColormap.from_list(
        cmap_name, color_palette, N=len(color_palette))
    sns.set_palette(color_palette)
    sns.set_style("white", {"grid.color": "#ffffff",
                  "axes.facecolor": "w", 'figure.facecolor': 'white'})

    st.title("Let's Visualise our Data!")
    st.subheader("Meet Miguel and Emma!")

    col1, col2 = st.columns(2)

    with col1:
        # Sample URL, replace with your desired animation
        lottie_url = "https://lottie.host/0db51d3e-e84e-4e5a-8b1e-f73a89a77f65/i1GvROt5y3.json"
        lottie_animation = load_lottie_url(lottie_url)
        st_lottie(lottie_animation, speed=1, width=350, height=350)
        st.markdown(
            "<div style='text-align: center'>Pablo Miguel</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center'>A 58-year-old technician, married, with a secondary education background.</div>", unsafe_allow_html=True)

    with col2:
        # Sample URL, replace with your desired animation
        lottie_url = "https://lottie.host/067bfd39-6ab6-484b-abd1-37451c842fd3/4OhK1ZCsaG.json"
        lottie_animation = load_lottie_url(lottie_url)
        st_lottie(lottie_animation, speed=1, width=350, height=350)
        st.markdown(
            "<div style='text-align: center'>Olivia Emma</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align: center'>A 38-year-old manager, single, with no personal loans.</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("The Question")
    st.success(
        "Which one of them is likely to make a term deposit? Let's dive deep and look at our sample data!")
    df = read_data("data/raw/bank-full.csv")
    st.table((df
              .head()
              ))

    st.divider()
    st.subheader("The Obstacles")
    st.write("Our detectives face a challenge. Only a small percentage of clients subscribe to term deposits. Amidst the vast ocean of data, finding the Golden Client is no easy task.")

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots()

        fig = (so
               .Plot((df
                      .deposit
                      .value_counts(normalize=True)
                      .to_frame()
                      .reset_index()
                      ),
                     x='deposit',
                     y='proportion')
               .add(so.Bar(edgewidth=0),)
               .label(x='Deposit',
                      y='Proportion',
                      color=str.capitalize,
                      title='Proportion of Deposit')
               .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[8]])})
               .on(ax)
               .show()
               )
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close('all')

    with col4:
        with st.expander("Insights", expanded=True):
            st.markdown("""
                        1. Our dataset is relatively imbalanced with the proportion of response (no / yes) of 88.3% to 11.7%
                        2. Additionally, we decide not to include `duration` as a feature in our model as it is highly predictive of the target and could potentially result in data leakage. At the time of performing prediction, this feature would not be available.
                        """)

    with st.expander("Click to view code"):
        code = """
        (so
         .Plot((df
                .deposit
                .value_counts(normalize=True)
                .to_frame()
                .reset_index()
               ),
               x='index',
               y='deposit')
         .add(so.Bar(edgewidth=0),)
         .scale(x=so.Continuous().tick(between=(0, 1), count=2))
         .label(x='Deposit',
                y='Proportion',
                color=str.capitalize,
                title=f'Proportion of Deposit')
         .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[8]])})
        )
        """
        st.code(code, language='python')

    st.divider()
    st.subheader("The Journey Begins")

    st.write("In this section, we will only focus on a few key aspects. For more details, please refer to my GitHub.")

    st.markdown("""
             The dataset contains various attributes related to the clients of the bank. We can categorise them according to:
             **Demographics:** This includes attributes like age, job, marital status, and education. Visualizing these can help us understand the composition of the clients.\n
             **Financial Status:** Attributes like 'balance', 'housing', 'loan', and 'default' fall into this category. It will be interesting to see how these financial attributes correlate with the target variable (i.e., whether a client subscribes to a term deposit or not).\n
             **Campaign Data:** This refers to attributes like 'contact', 'duration', 'campaign', 'pdays', 'previous', and 'poutcome'. Insights from these attributes can help the bank refine its marketing strategies in the future.\n
             """)

    col5, col6 = st.columns(2)

    with col5:
        fig, ax = plt.subplots()
        fig = (so
               .Plot(df
                     .age)
               .add(so.Bars(color=color_palette[8], edgewidth=0),
                    so.Hist("density", bins=30),)
               .add(so.Line(), so.KDE())
               .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[8], color_palette[3]]), "axes.facecolor": "w", "grid.color": "#ffffff"})
               .label(x='',
                      y='Frequency',
                      color=str.capitalize,
                      title='Age Distribution',)
               .on(ax)
               .show()
               )
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        st.pyplot(fig)
        st.pyplot(plot_categorical(df, "job"))
        plt.close('all')

    with col6:
        st.pyplot(plot_categorical(df, "marital"))
        st.pyplot(plot_categorical(df, "education"))
        plt.close('all')

    with st.expander("Insights"):
        st.markdown("""
                    **1. Age Distribution:**\n
    
                    - The majority of the bank's clients are between 30 and 40 years old.\n
                    - There's a noticeable decrease in the number of clients as age increases, with fewer clients in the older age groups.\n
                    **2. Job Type Distribution:**\n
    
                    - The top three job categories among the bank's clients are 'blue-collar', 'management', and 'technician'.\n
                    - Few clients fall under the 'unknown' job type, which might require further investigation or data cleaning.\n
                    **3. Marital Status Distribution:**\n
    
                    - A significant number of clients are married, followed by single and divorced clients.\n
                    **4. Education Level Distribution:**
    
                    - Most clients have received a secondary level of education, followed by tertiary and primary. A small fraction of clients have an 'unknown' education level.\n
                    """)

    with st.expander("Click to view code"):
        code = """
        def plot_categorical(df: pd.DataFrame, 
                             col: str, 
                             new_index: list = None) -> so:

            df_to_plot = (df[col]
                          .value_counts(normalize=True)
                          .to_frame()
                          .reset_index())

            if new_index is not None:
                df_to_plot = (df_to_plot
                              .set_index('index')
                              .reindex(new_index)
                              .reset_index())

            return (so
                    .Plot(df_to_plot, x=col, y='index')
                    .add(so.Bar(color=color_palette[8], edgewidth=0))
                    .label(x='',
                           y=col,
                           color=str.capitalize,
                           title=f'Normalised Count of {col}')
                    .show()
                    )
        """
        st.code(code, language='python')

    st.divider()
    st.subheader("Bivariate Analysis")
    st.markdown("""
                Next, let's investigate how these **demographics** relate to the outcome (whether the client subscribed to a term deposit or not).
                
                """)

    col7, col8 = st.columns(2)

    with col7:
        fig, ax = plt.subplots()
        fig = (so
               .Plot(df,
                     x='age',
                     color='deposit',)
               .add(so.Area(),
                    so.KDE())
               .theme({"axes.facecolor": "w", "grid.color": "#ffffff"})
               .label(x='age',
                      y='Frequency',
                      color=str.capitalize,
                      title='Age Distribution')
               .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[8], color_palette[3]])})
               .show()
               )

        st.pyplot(fig)
        st.pyplot(plot_features_label(df, "job"))
        plt.close('all')

    with col8:
        st.pyplot(plot_features_label(df, "marital"))
        st.pyplot(plot_features_label(df, "education"))
        plt.close('all')

    with st.expander("Insights"):
        st.markdown("""
                    **1. Subscription Rate Across Job Types:**\n
    
                    - While 'blue-collar', 'management', and 'technician' jobs have the highest numbers of clients, it's interesting to observe that the proportion of 'students' and 'retired' individuals who subscribe to term deposits seems to be relatively higher. This suggests that while certain job categories have more clients, the likelihood of subscribing to a term deposit varies across job types.\n
                    **2. Subscription Rate Across Marital Statuses:**\n
                    
                    - Married individuals constitute the highest number of clients, but when it comes to the proportion of subscriptions, singles have a slightly higher propensity to subscribe to term deposits than their married or divorced counterparts.\n
                    
                    **3. Subscription Rate Across Education Levels:**\n
                    
                    - Clients with tertiary education appear more inclined to subscribe to term deposits when compared to those with secondary or primary education. This insight could be valuable for targeting marketing efforts.\n
                    """)

    with st.expander("Click to view code"):
        code = """
        def plot_features_label(df: pd.DataFrame,
                                col: str,
                                new_index: list = None) -> so:
            
            df_to_plot = (df
                          .groupby([col])
                          .deposit #y
                          .value_counts(normalize=True)
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
                    .label(x='',
                           y=col,
                           color=str.capitalize,
                           title=f'{col} v. Deposit Outcome')
                    .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[0], color_palette[3]])})
                    .show()
                   )
        """
        st.code(code, language='python')

    st.markdown("""
                Next, let's delve deeper into the **financial status** of the clients and how it correlates with the decision to subscribe to a term deposit! 
                """)

    col9, col10 = st.columns(2)

    with col9:
        fig, ax = plt.subplots()
        fig = (sns
               .boxplot(df,
                        x='deposit',
                        y='balance',
                        orient='v',
                        boxprops=dict(alpha=0.7),
                        palette=[color_palette[8], color_palette[3]])
               .set_title("balance v. Deposit Outcome")
               .figure
               )
        ax.set_ylim(-5000, 6000)
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        st.pyplot(fig)
        st.pyplot(plot_features_label(df, "housing"))
        plt.close('all')

    with col10:
        st.pyplot(plot_features_label(df, "loan"))
        st.pyplot(plot_features_label(df, "default"))
        plt.close('all')

    with st.expander("Insights"):
        st.markdown("""
                    **1. Balance Distribution by Subscription Status:**\n
    
                    - Clients who subscribed to the term deposit generally have a higher median balance compared to those who didn't. The presence of some outliers, especially for clients with a positive subscription, suggests a few clients with significantly high balances.\n
                    **2. Subscription Rate for Housing Loan:**\n
                    
                    - A larger proportion of clients without a housing loan seem to subscribe to term deposits compared to those with a housing loan. This could suggest that clients without housing loans might have more financial flexibility or willingness to invest in term deposits.\n
                    **3. Subscription Rate for Personal Loan:**\n
                    
                    - Similarly, clients without a personal loan have a higher propensity to subscribe to term deposits. This trend is even more pronounced than with housing loans.
                    **4. Subscription Rate for Credit Default:**\n
                    
                    - As expected, clients with no credit default have a higher rate of subscription to term deposits. Very few clients with credit defaults subscribe to term deposits, which makes sense given their financial history.
    
                    """)

    st.markdown("""
                Next, we can explore the **campaign** data to derive insights on the bank's marketing strategies and how they correlate with subscription rates.
                """)

    col11, col12 = st.columns(2)

    with col11:
        st.pyplot(plot_features_label(df, "contact"))
        # st.pyplot(plot_features_label(df, "poutcome"))
        plt.close('all')

    with col12:
        st.pyplot(plot_features_label(df, "poutcome"))
# =============================================================================
#         fig, ax = plt.subplots()
#         ax = (sns
#                .boxplot(df,
#                         x='deposit',
#                         y='duration',
#                         orient='v',
#                         boxprops=dict(alpha=0.7),
#                         palette=[color_palette[8], color_palette[3]])
#               )
#         ax.set_title("duration v. Deposit Outcome")
#         ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
#         fig = ax.figure
#         st.pyplot(fig)
#
#         fig, ax = plt.subplots()
#         fig = (sns
#                .boxplot(df,
#                         x='deposit',
#                         y='campaign',
#                         orient='v',
#                         boxprops=dict(alpha=0.7),
#                         palette=[color_palette[8], color_palette[3]])
#               )
#         ax.set_title("campaign v. Deposit Outcome")
#         ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
#         fig = ax.figure
#         ax.set_ylim(0,10)
#         st.pyplot(fig)
# =============================================================================
        plt.close('all')

    with st.expander("Insights"):
        st.markdown("""
                    **1. Subscription Rate by Contact Type:**\n
    
                    - Clients contacted via cellular means exhibit a higher subscription rate to term deposits compared to other methods. The 'unknown' contact method also has a significant number of clients, indicating possible areas for data improvement or further investigation.\n
                    **2. Last Contact Duration by Subscription Status:**\n
                    
                    - Clients who subscribed to the term deposit generally had longer last contact durations. This could indicate that more extensive discussions or convincing might lead to successful subscriptions.\n
                    **3. Subscription Rate by Outcome of Previous Marketing Campaign:**\n
                    
                    - Clients with a 'success' outcome in the previous marketing campaign are significantly more likely to subscribe to term deposits in the current campaign. This emphasizes the importance of prior successful interactions with clients.\n
                    **4. Number of Contacts by Subscription Status:**\n
                    
                    - On average, clients who didn't subscribe to the term deposit were contacted slightly more times during this campaign compared to those who subscribed. This could imply that repeated contacts without success might indicate a lower likelihood of conversion.\n
                    """)

    st.markdown("""
                Next, we can explore the distribution of last contact **month and day** and observe if there is any meaningful pattern.
                """)
    col13, col14 = st.columns(2)

    with col13:
        st.pyplot(plot_categorical(df, "month", [
                  'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']))
        plt.close('all')

    with col14:
        st.pyplot((so
                   .Plot((df
                          .day
                          .value_counts()
                          .sort_index()
                          .to_frame()
                          .reset_index()
                          ),
                         x="day",
                         y="count")
                   .add(so.Line(color=color_palette[0]))
                   .theme({"axes.facecolor": "w", "grid.color": "#ffffff"})
                   .label(x="Day",
                          y='Frequency',
                          color=str.capitalize,
                          title='Day of the month Distribution')
                   .show()
                   ))
        plt.close('all')

    with st.expander("Insights"):
        st.markdown("""
                    **1. Last contact Month of Year:**\n
    
                    - May to August period seems to be the period with highest contact with clients.\n
                    **2. Last contact Day of Month:**\n
                    
                    - Day of Month does not seem to have any discernible pattern, although we can observe that frequency of contact peaked at 20th. \n 
                    """)

    st.divider()
    st.subheader("Correlation Analysis")
    st.markdown("""
                Let's also look at the relationship between numerical features.
                """)

    choice = st.selectbox("What do you want to display? (Default: Pearson)", [
                          "Pearson", "Spearman"])

    method = 'spearman' if choice == "Spearman" else "pearson"

    st.table(df
             .select_dtypes('int')
             .corr(method=method, numeric_only=True)
             .style
             .background_gradient(cmap=cm, axis=None)
             .set_sticky(axis="index")
             .set_sticky(axis="columns")
             )

    with st.expander("Click to view code"):
        code = """
        method = 'spearman' if choice == "Spearman" else "Pearson"
            
        st.table(df
                 .select_dtypes('int')
                 .corr(method=method, numeric_only=True)
                 .style
                 .background_gradient(cmap="viridis", axis=None)
                 .set_sticky(axis="index")
                 .set_sticky(axis="columns")
                )
        """
        st.code(code, language='python')

    st.markdown("""
                Let's also look at the relationship between categorical features using Cramér's V.
                """)

    df_cat = pd.concat([df.select_dtypes('object')], axis=1)

    label = LabelEncoder()
    df_encoded = pd.DataFrame()

    for i in df_cat.columns:
        df_encoded[i] = label.fit_transform(df_cat[i])

    def Cramers_V(var1, var2):
        crosstab = np.array(pd.crosstab(index=var1, columns=var2))  # Cross Tab
        # Return Cramer's V
        return (association(crosstab, method='cramer'))

    # Create the dataFrame matrix with the returned Cramer's V
    rows = []

    for var1 in df_encoded:
        col = []

        for var2 in df_encoded:
            # Return Cramer's V
            V = Cramers_V(df_encoded[var1], df_encoded[var2])
            # Store values to subsequent columns
            col.append(V)

        # Store values to subsequent rows
        rows.append(col)

    CramersV_results = np.array(rows)
    CramersV_df = (pd
                   .DataFrame(CramersV_results, columns=df_encoded.columns, index=df_encoded.columns)
                   .style
                   .background_gradient(cmap=cm, axis=None)
                   .set_sticky(axis="index")
                   .set_sticky(axis="columns"))
    st.table(CramersV_df)
    plt.close('all')

    with st.expander("Click to view code"):
        code = """
        df_cat = pd.concat([df.select_dtypes('object')], axis=1)

        label = LabelEncoder()
        df_encoded = pd.DataFrame()
        
        for i in df_cat.columns:
                df_encoded[i] = label.fit_transform(df_cat[i])
            
        def Cramers_V(var1, var2):
            crosstab = np.array(pd.crosstab(index=var1, columns=var2)) 
            return (association(crosstab, method='cramer'))
        
        rows = []
        
        for var1 in df_encoded:
            col = []
        
            for var2 in df_encoded:
                V = Cramers_V(df_encoded[var1], df_encoded[var2])
                col.append(V)                           
                
            rows.append(col)                      
            
        CramersV_results = np.array(rows)
        CramersV_df = (pd
                       .DataFrame(CramersV_results, columns = df_encoded.columns, index = df_encoded.columns)
                       .style
                       .background_gradient(cmap="viridis", axis=None)
                       .set_sticky(axis="index")
                       .set_sticky(axis="columns"))
        st.table(CramersV_df)
        """
        st.code(code, language='python')

    st.divider()
    st.subheader("Multivariate Analysis")
    st.markdown("""
                We are almost there! Let's visualise the relationships between multiple features and target to uncover more insights!
                """)

    st.pyplot((so
               .Plot(df,
                     x='default',
                     y='balance',
                     color='deposit')
               .add(so.Dot(alpha=0.3),
                    so.Dodge(),
                    so.Jitter(.3))
               .theme({"axes.facecolor": "w", "grid.color": "#ffffff"})
               .label(x='default',
                      y='balance',
                      color=str.capitalize,
                      title='Balance / Default / Deposit')
               .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[0], color_palette[3]]), })
               .show()
               )
              )
    plt.close('all')

    with st.expander("Click to view code"):
        code = """
        (so
         .Plot(df, 
               x='default',
               y='balance',
               color='deposit')
         .add(so.Dot(alpha=0.3),
              so.Dodge(),
              so.Jitter(.3))
         .label(x='default',
                y='balance',
                color=str.capitalize,
                title='Balance / Default / Deposit')
         .theme({"axes.prop_cycle": matplotlib.cycler(color=[color_palette[0], color_palette[3]]),})
         .show()
        )
                 
        """
        st.code(code, language='python')

# =============================================================================
#     st.write("Stripplot of Balance v. Job v. Deposit")
#     fig, axes = plt.subplots(figsize=(18, 6))
#     sns.stripplot(data=df,
#                   x='job',
#                   y=df["balance"],
#                   size=2,
#                   hue="deposit",
#                   palette=[color_palette[0], color_palette[3]],
#                   alpha=0.3,
#                   ax=axes,)
#     plt.tight_layout()
#     axes.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
#     st.pyplot(fig)
#     plt.close('all')
#
#     with st.expander("Click to view code"):
#         code = """
#         fig, axes = plt.subplots(figsize=(18, 6))
#         sns.stripplot(data=df,
#                       x='job',
#                       y=df["balance"],
#                       size=2,
#                       hue="deposit",
#                       palette=[color_palette[0], color_palette[3]],
#                       alpha=0.3,
#                       ax=axes,)
#         plt.tight_layout()
#         """
#         st.code(code, language='python')
# =============================================================================

    st.write("More Stripplots")
    n_cols = 3
    n_elements = len(['marital', 'education', 'default',
                     'housing', 'loan', 'contact'])
    n_rows = np.ceil(n_elements / n_cols).astype("int")

    y_value = df["balance"]

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows,
                             figsize=(15, n_rows * 3.5))

    for col, ax in zip(['marital', 'education', 'default', 'housing', 'loan', 'contact'], axes.ravel()):
        sns.stripplot(data=df,
                      x=col,
                      y=y_value,
                      ax=ax,
                      size=1,
                      hue="deposit",
                      palette=[color_palette[0], color_palette[3]],
                      alpha=0.3)
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close('all')

    with st.expander("Click to view code"):
        code = """
        n_cols = 3
        n_elements = len(['marital', 'education', 'default', 'housing', 'loan', 'contact'])
        n_rows = np.ceil(n_elements / n_cols).astype("int")
        
        y_value = df["balance"]
        
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(15, n_rows * 3.5))
        
        for col, ax in zip(['marital', 'education', 'default', 'housing', 'loan', 'contact'], axes.ravel()):
            sns.stripplot(data=df, x=col, y=y_value, ax=ax, size=1, hue="deposit", palette=[color_palette[0], color_palette[3]], alpha=0.3)
        
        plt.tight_layout();
        """
        st.code(code, language='python')

    plt.close('all')

    del(
        fig,
        ax,
        axes,
        code,
        df,
        col1,
        col2,
        col3,
        col4,
        col5,
        col6,
        col7,
        col8,
        col9,
        col10,
        col11,
        col12,
        col13,
        col14,
        n_cols,
        n_elements,
        n_rows,
        y_value
    )
    gc.collect()


if __name__ == "__main__":
    main()

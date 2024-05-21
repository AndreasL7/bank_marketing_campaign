import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TweakBankMarketing(BaseEstimator, TransformerMixin):
    
    def fit(self,
            X: pd.DataFrame,
            y=None):
        return self

    def transform(self,
                  X: pd.DataFrame,
                  y=None):

        job_mapping = {'entrepreneur': 'self-employed',
                       'retired': 'unemployed',
                       'student': 'unemployed',
                       'unknown': 'unemployed',
                       'housemaid': 'services'}

        return (X
                .assign(balance_pos=lambda df_: np.where(df_.balance < 0, 0, 1),
                        pdays_contacted=lambda df_: np.where(
                    df_.pdays < 0, 0, 1),
                    previous_contacted=lambda df_: np.where(
                    df_.previous == 0, 0, 1),
                    job=lambda df_: df_.job.replace(job_mapping),
                    education=lambda df_: np.where(
                    df_.education == "unknown", np.nan, df_.education),
                    default=lambda df_: np.where(
                    df_.default == "no", 0, 1),
                    housing=lambda df_: np.where(
                    df_.housing == "no", 0, 1),
                    loan=lambda df_: np.where(df_.loan == "no", 0, 1),
                    # contact=lambda df_:np.where(df_.contact == "unknown", np.nan, df_.contact),
                    poutcome=lambda df_: np.where(
                    df_.poutcome == "other", "unknown", df_.poutcome),
                )
                .drop(columns=['duration'])
                .astype({**{k: 'int8' for k in ['age', 'default', 'housing', 'loan', 'campaign', 'balance_pos', 'pdays_contacted', 'previous_contacted']},
                         **{k: 'int16' for k in ['pdays', 'previous']},
                         'balance': 'int32',
                         'day': 'category', })
                .pipe(lambda df_: df_.astype({column: 'category' for column in (df_.select_dtypes("object").columns.tolist())}))
                )
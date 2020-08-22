from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE
import numpy as np

# All sklearn Transforms must have the `transform` and `fit` methods

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class RFE_obj(BaseEstimator, TransformerMixin):
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y):
        
        r = RFE(self.model, step=1).fit(X,y)
        cols = r.get_support(indices=True)      
        X_new= X.iloc[:,cols] 
        return X_new
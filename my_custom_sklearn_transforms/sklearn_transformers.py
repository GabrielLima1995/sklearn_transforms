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

class Create_Features(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        nota_humanas = np.median(X[:,[6,7,9]],axis = 1).reshape(-1,1)
        nota_global =  np.median(X[:,[6,7,8,9]],axis=1).reshape(-1,1)
        reprovacao =   np.median(X[:,[2,3,4,5]],axis =1).reshape(-1,1)
        new_x = np.concatenate((X,nota_humanas,nota_global,reprovacao),axis=1)

        return new_x

class RFE_obj(BaseEstimator, TransformerMixin):
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X,y):
        
        r = RFE(self.model, step=1).fit(X,y)
        cols = r.get_support(indices=True)
        X_new= X[:,cols]       
        
        return X_new,y
from sklearn.base import BaseEstimator, TransformerMixin


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
        
        nota_humanas = {'HUMANAS':X.iloc[:,[6,7,9]].median(axis = 1)}
        nh = pd.DataFrame(data = nota_humanas)
        nota_global = {'GLOBAL':X.iloc[:,[6,7,8,9]].median(axis=1)}
        ng = pd.DataFrame(data = nota_global)
        reprovacao = {'REP_COUNT' : X.iloc[:,[2,3,4,5]].median(axis =1)}
        rep = pd.DataFrame(data=reprovacao)
        df_data_4 = pd.concat([X,nh,ng,rep],axis = 1)

        return df_data_4

class RFE_obj(BaseEstimator, TransformerMixin):
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        r = RFE(self.model, step=1).fit(X,y)
        cols = r.get_support(indices=True)
        X_new= X.iloc[:,cols]       
        
        return X_new
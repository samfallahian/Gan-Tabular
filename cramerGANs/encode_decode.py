
###############


import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,  accuracy_score
from sklearn.tree import DecisionTreeClassifier


###############



class Data:

    def __init__(self,file,cat_cols, use_cols=None, nrows=None, maxSamples = None):

        _, file_extension = os.path.splitext(file)
        
        if file_extension == ".csv":
            df = pd.read_csv(file,usecols = use_cols, nrows = nrows)
        elif file_extension == ".json":
            df = pd.read_json(file, lines=True)
            df = df[use_cols]
            if nrows is not None:
                df = df.iloc[1:nrows,:]
        elif file_extension == ".gz":
            import gzip
            with gzip.open(file, "rb") as f:
                df = pd.read_json(f.read().decode("ascii"), lines=True)
        else:
            print("wrong file format: "+ file_extension)
            exit(1)


        self.cenc = CatEncoder(cat_cols)

        self._data = self.cenc.fit_transform(df)

        self.shape = self._data.shape
        
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._data.shape[0]

        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._data = self._data[perm]

    
    def __call__(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
    
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._data = self._data[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end]




class CatEncoder:
    
    def __init__(self,cat_cols):
        self.feature_indices = None
        self.vectorizer = None
        self.cat_cols = cat_cols
        self.cat_card = None
        self.num_cols =  None # set in fit_transform
        self.columns =  None # set in fit_transform to save original column order
        self.mm =  None
        self.le = None
        self.num_cols_with_nulls = []
        

    def transform(self,df,original_cat_cols):

        #leave original 
        df = df.copy()
        
        for c in (set(df.columns) - set(original_cat_cols)):
           if df[c].isnull().any():
               newcol = c+"_isnull"
               df[newcol] = "0"
               df.loc[df[c].isnull().values,newcol] = "1"
               df[c].fillna(df[c].mean(), inplace=True)
        
    
        df[self.cat_cols] = df[self.cat_cols].applymap(str)           
        vectorized = self.vectorizer.transform( df[self.cat_cols].to_dict(orient='records') )

        #compute feature_indices (as in oneHotEncoder, except that it doesn't contain the starting position) for inverse_transform
        columns = [s.split('=')[0] for s in sorted(self.vectorizer.vocabulary_.keys())]
        columns = self.le.transform(columns)   
        
		nummat = df.drop(self.cat_cols, axis=1)
        nummat = self.mm.transform(nummat) 
        
		vectorized = np.concatenate((nummat,vectorized), axis=1)

        return vectorized
    
    
    
    def fit_transform(self,df):
        
        for c in (set(df.columns) - set(self.cat_cols)):
           if df[c].isnull().any():
               print(c+" contains null")
               newcol = c+"_isnull"
               df[newcol] = "0"
               df.loc[df[c].isnull().values,newcol] = "1"
               self.cat_cols.append(newcol)
               df[c].fillna(df[c].mean(), inplace=True)
               self.num_cols_with_nulls.append(c)
        
        self.columns = df.columns    
        df[self.cat_cols] = df[self.cat_cols].applymap(str)
        self.cat_card = df[self.cat_cols].apply(lambda x: np.unique(x).size)    
        self.vectorizer = DictVectorizer( sparse = False )
        vectorized = self.vectorizer.fit_transform( df[self.cat_cols].to_dict(orient='records') )

        
        #compute feature_indices (as in oneHotEncoder, except that it doesn't contain the starting position) for inverse_transform
        columns = [s.split('=')[0] for s in sorted(self.vectorizer.vocabulary_.keys())]
        self.le = LabelEncoder()
        columns = self.le.fit_transform(columns)               
        self.feature_indices = np.where(columns[:-1] !=columns[1:])[0]+1
        self.feature_indices = np.append(self.feature_indices,vectorized.shape[1])
        
		#numerical columns
		nummat = df.drop(self.cat_cols, axis=1)
        self.num_cols = nummat.columns
        self.mm = MinMaxScaler() 
        nummat = self.mm.fit_transform(nummat) 

        vectorized = np.concatenate((nummat,vectorized), axis=1)

        return vectorized

    
    
    #DECODE
    def inverse_transform(self,mat):        
        nummat = mat[:,:len(self.num_cols)]
        ohmat =  mat[:,len(self.num_cols):]
        
        #correct not properly one-hot encoded data by assigning 1 to largest value and 0 to the rest           
        start = 0   
        for end in self.feature_indices:       
            view=ohmat[:,start:end]            
            c_argmax = np.argmax(view,axis=1)[:,np.newaxis]
            b_map1 = c_argmax == np.arange(view.shape[1])        
            view[b_map1] = 1
            view[~b_map1] = 0   
            start = end

        
        print("vectorize inverse_transform step...")
        recovered = self.vectorizer.inverse_transform(ohmat)
        print("dataframe building step...")
        dfRecovered = pd.DataFrame.from_dict([list_to_dict(r.keys()) for r in recovered])   
        print("minmax inverse_transform step...")
        nummat = self.mm.inverse_transform(nummat)
        dfNum = pd.DataFrame(nummat,columns=self.num_cols)    
        ret = pd.concat([dfRecovered,dfNum],axis=1,ignore_index=False)
    
        #reorder
        ret = ret[self.columns]
        for c in self.num_cols_with_nulls:
            newcol =  c+"_isnull"
            ret.loc[ret[newcol]=="1",c] = np.nan 
            ret.drop(newcol, axis=1, inplace=True)
    
        return ret



############



def preprocessData_model(dataIn, cols_featuresNum,  cols_toOneHot, col_target):
       
    data_list=[]
    cols_out=[]
    
    # numerical data
    numEnc=None
    if len(cols_featuresNum)>0:
        data_numerical=dataIn[cols_featuresNum].astype(np.float).values
       
        num_preP_type='StdScaler'
        if num_preP_type=='MinMax':
            numEnc=MinMaxScaler()
        else:
            numEnc=StandardScaler()            
        data_numerical=numEnc.fit_transform(data_numerical)
    
    
        data_list.append(data_numerical)
        cols_out=cols_out+cols_featuresNum
          
       
    #categorical data: one hot
    le_list_OH=[]
    oneHotEnc=None
    colsOH=[]
    if len(cols_toOneHot)>0:
        data_cat=dataIn[cols_toOneHot].astype(np.str).values
        for colN in  range(len(cols_toOneHot)): 
            unique_vals=np.unique(data_cat[:,colN])
            le = LabelEncoder()
            le.fit(unique_vals)
            data_cat[:,colN]=le.transform(data_cat[:,colN])
            le_list_OH.append(le)
            
        
        oneHotEnc = OneHotEncoder(handle_unknown='ignore',sparse=False)#igore if new value in test set
        data_oh=oneHotEnc.fit_transform(data_cat)
        colsOH=[]
        f_index=oneHotEnc.feature_indices_
        for i, colName in enumerate(cols_toOneHot):

            le=le_list_OH[i]
            i_0=f_index[i]
            i_end=f_index[i+1] 

            colsClass=[colName+'_'+str(le.inverse_transform(t)) for t in range(i_end-i_0) ]
            colsOH=colsOH+colsClass
            
        data_list.append(data_oh)   
        cols_out=cols_out+colsOH

   
    data_np=np.concatenate(data_list,axis=1)#by columns
    target_preP=dataIn[col_target].values
    le_target = preprocessing.LabelEncoder()
    target=le_target.fit_transform(target_preP)

    return data_np, target, numEnc, le_list_OH, oneHotEnc, colsOH, le_target
     



def crossEvaluation(df_real,df_final,col_target,cols_toOneHot,cols_featuresNum,classifier,size_test=0.5):

    df_model=pd.concat((df_real,df_final), axis=0,ignore_index=True)
    data_full, target_full, numEnc, le_list_OH, oneHotEnc, colsOH, le_target= preprocessData_model(df_model, cols_featuresNum,  cols_toOneHot, col_target)
	
    indx_initFake=df_real.shape[0]
    x_real=data_full[0:indx_initFake,:]
    target_real=target_full[0:indx_initFake]
    x_fake=data_full[indx_initFake:,:]
    target_fake=target_full[indx_initFake:]

    x_real_train, x_real_test, target_real_train, target_real_test= train_test_split(x_real, target_real, test_size=size_test, random_state=0)
    x_fake_train, x_fake_test, target_fake_train, target_fake_test= train_test_split(x_fake, target_fake, test_size=size_test, random_state=0)
    
    if classifier=='rf':
        clf=RandomForestClassifier(random_state=0)
        clf_fake=RandomForestClassifier(random_state=0)
    elif classifier=='tree':
        clf=DecisionTreeClassifier(max_depth=10,max_features ='auto',random_state=0)
        clf_fake=DecisionTreeClassifier(max_depth=10,max_features ='auto',random_state=0)
    else:
        #classifier=='logreg':
        clf=LogisticRegression(random_state=0)
        clf_fake=LogisticRegression(random_state=0)
    
    

    #train on real, test real
    clf.fit(x_real_train, target_real_train) 
    target_pred_real_real=clf.predict(x_real_test)
    print(classification_report(target_real_test, target_pred_real_real))
    print(confusion_matrix(target_real_test, target_pred_real_real))
    print(accuracy_score(target_real_test, target_pred_real_real))
    
    
    #train on fake test on real
    clf_fake.fit(x_fake_train, target_fake_train) 
    
    #test on real
    target_pred_fake_real=clf_fake.predict(x_real_test)
    print(classification_report(target_real_test, target_pred_fake_real))
    print(confusion_matrix(target_real_test, target_pred_fake_real)) 
    print(accuracy_score(target_real_test, target_pred_fake_real))
     
    
    return None
  
    

def  classifier_filter(df_real,df_final,col_target,cols_toOneHot,cols_featuresNum,classifier,size_test=0.5):
    
    df_model=pd.concat((df_real,df_final), axis=0,ignore_index=True)
    data_full, target_full, numEnc, le_list_OH, oneHotEnc, colsOH, le_target= preprocessData_model(df_model, cols_featuresNum,  cols_toOneHot, col_target)
    data_train, data_test, target_train, target_test, index_train, index_predict = train_test_split(data_full, target_full,df_model.index, test_size=size_test, random_state=0)
    
    
    if classifier=='rf':
        clf=RandomForestClassifier(random_state=0)
        clf.fit(data_train, target_train) 
    elif classifier=='tree':
        clf=DecisionTreeClassifier(max_depth=10,max_features ='auto',random_state=0)#,min_samples_leaf=10)
        clf.fit(data_train, target_train) 
    elif classifier=='logreg':
        clf=LogisticRegression(random_state=0)
        clf.fit(data_train, target_train) 
    else:
        clf = KNeighborsClassifier(n_neighbors=int((data_train.shape[0])**0.5), n_jobs=6)
        clf.fit(data_train, target_train) 

        
    target_pred=clf.predict(data_test)
    print(classification_report(target_test, target_pred))
    print(confusion_matrix(target_test, target_pred))
             
    return None




def list_to_dict(rlist):
    return dict(map(lambda s : s.split('='), rlist))




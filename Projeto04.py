#!/usr/bin/env python
# coding: utf-8

# ### Fazer a análise exploratória, buscando identificar quais variáveis numéricas melhor separam as classes
# ### Criar boxplots com os resultados da validação cruzada de vários algoritmos
# ### Fazer feature selection
# ### Verificar o balanceamento
# ### Codificar variáveis categóricas
# ### Rodar um GridSearch para otimizar os hiperparâmetros

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, RFE, mutual_info_classif, chi2, f_classif


# In[3]:


#!pip install --user imblearn


# In[ ]:


df = pd.read_csv('projeto4_telecom_treino.csv')


# In[6]:


df.head()


# In[7]:


df.shape


# ## A primeira coluna é uma contagem e não tem utilidade

# In[8]:


df = df.iloc[:,1:]


# In[9]:


df.dtypes


# In[10]:


c=0
obj=[]
for x in df.dtypes:
    if x=='object':
        obj.append(c)
    c+=1


# In[11]:


df.iloc[:,obj].nunique()


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


numericas = []
for x in df.columns:
    if df[x].dtype=='int64' or df[x].dtype=='float64':
        numericas.append(x)


# In[15]:


len(numericas)


# In[16]:


i=0
plt.figure(figsize=(10,25))
for col in numericas:
    i+=1
    plt.subplot(8,2,i)
    sns.boxplot(data=df,x=col,y='churn')
    plt.tight_layout()


# ## Os clientes que tem mais tempo de chamada durante o dia (mas não necessariamente fazem mais ligações) e que mais pagam (por esse serviço) são os que mais cancelam a linha
# 
# ## Clientes que cancelam a linha ligam mais para o atendimento ao cliente

# In[17]:


df[['total_day_calls','total_day_minutes']].corr()


# ## Dentre as variáveis numéricas, as que não separam bem as classes são: account_length, total_night_charge, total_night_minutes, total_day_calls, total_eve_calls, total_night_calls

# In[18]:


categoricas = [x for x in df.columns if x not in numericas and x!='churn']


# In[19]:


i=0
plt.figure(figsize=(10,20))
for col in categoricas:
    i+=1
    plt.subplot(len(categoricas),1,i)
    sns.countplot(data=df,x=col,hue='churn')
    plt.tight_layout()


# ## Quase todos os clientes com um plano internacional cancelam a linha

# ## Testando a associação entre as variáveis categóricas e a classe
# ## H0: Não existe associação entre as variáveis

# In[20]:


for x in categoricas:
    contingencia = pd.crosstab(index=df['churn'],columns=df[x])
    print(f'A probabilidade de erro ao rejeitar H0 entre churn e {x} é: {chi2_contingency(contingencia)[1]*100:.2f}%\n')


# ## Conclui-se que todas as variáveis categóricas tem associação com a saída, exceto area_code

# In[21]:


sns.countplot(data=df,x='churn');


# In[22]:


df['churn'].value_counts()


# ## Os dados estão bem desbalanceados, espera-se que a acurácia do modelo seja melhor para a classe "no". Se for o caso, usar técnicas de rebalanceamento

# ## Codificando variáveis categóricas

# In[23]:


onehot = OneHotEncoder(sparse=False)


# In[24]:


categoricas_cod = onehot.fit_transform(df[categoricas])


# In[25]:


numericas = df[numericas].values


# In[26]:


minmax = MinMaxScaler()
numericas_s = minmax.fit_transform(numericas)


# In[27]:


todas_var = np.concatenate([numericas_s,categoricas_cod],axis=1)


# In[28]:


y = df['churn'].values


# ## Testando validação cruzada sem feature selection e rebalanceamento

# In[29]:


y_encod = []
for x in y:
    if x=='yes':
        y_encod.append(1)
    else:
        y_encod.append(0)


# In[30]:


kf = KFold(n_splits=10)
modelos = [('RandomForest',RandomForestClassifier()),('KNN',KNeighborsClassifier()),('Reg. L.',LogisticRegression()),          ('SVM',SVC()),('NB',GaussianNB()),('LDA',LinearDiscriminantAnalysis()),('XGBoost',XGBClassifier(use_label_encoder=False))]


# In[31]:


resultados=[]
for _,modelo in modelos:
    resultados.append(cross_val_score(modelo,todas_var,y_encod,cv=kf,scoring='balanced_accuracy',n_jobs=-1))


# In[32]:


nomes = [x[0] for x in modelos]
plt.boxplot(resultados,labels=nomes);


# ## XGBoost é o que teve o melhor desempenho

# ## Esse algoritmo pede que as classes sejam números inteiros

# ## 1: Yes , 0: No

# In[33]:


x_treino, x_teste, y_treino, y_teste = train_test_split(todas_var,y_encod,test_size=0.3,random_state=1)


# In[34]:


xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(x_treino,y_treino)


# In[35]:


prev_xgb = xgb.predict(x_teste)


# In[36]:


print(classification_report(prev_xgb,y_teste))


# In[37]:


confusao = confusion_matrix(y_teste,prev_xgb,labels=[0,1])
print(f'Acurácia classe 0: {confusao[0,0]/(confusao[0,1]+confusao[0,0])*100:.4f}%')
print(f'Acurácia classe 1: {confusao[1,1]/(confusao[1,0]+confusao[1,1])*100:.4f}%')


# ## A classe 1 tem uma acurácia de 75%, vou tentar melhorar ela com rebalanceamento

# In[38]:


y_treino.count(0)/np.array(y_treino).shape[0]


# ## Antes a classe 0 era 85% dos dados

# In[39]:


smote = SMOTE(sampling_strategy='minority',random_state=1,k_neighbors=9)


# In[40]:


x_treino_over, y_treino_over = smote.fit_resample(x_treino,y_treino)


# In[41]:


y_treino_over.count(0)/np.array(y_treino_over).shape[0]


# ## Depois passou a ser 50%

# In[42]:


xgb = XGBClassifier(use_label_encoder=False)
xgb.fit(x_treino_over,y_treino_over)


# In[43]:


prev_xgb_over = xgb.predict(x_teste)


# In[44]:


print(classification_report(prev_xgb_over,y_teste))


# In[45]:


confusao = confusion_matrix(y_teste,prev_xgb_over,labels=[0,1])
print(f'Acurácia classe 0: {confusao[0,0]/(confusao[0,1]+confusao[0,0])*100:.4f}%')
print(f'Acurácia classe 1: {confusao[1,1]/(confusao[1,0]+confusao[1,1])*100:.4f}%')


# ## Não houve melhora significativa

# ## Testando se a feature selection melhora o modelo:
# ### 1) ExtraTreesClassifier, 2) RFE e 3) SelectKBest

# ## 1) ExtraTreesClassifier

# ## Preciso criar um dataframe sem variáveis dummy para que eu possa saber qual variável é qual na hora de obter o score de cada variável. Uma vez que usei o OneHotEncoder, não posso usar o todas_var para treinar o modelo

# In[46]:


num = []
for x in df.columns:
    if df[x].dtype=='int64' or df[x].dtype=='float64':
        num.append(x)


# In[47]:


lab_enc = LabelEncoder()


# In[48]:


df_extra = pd.DataFrame()


# In[49]:


for x in categoricas:
    lab_enc.fit(df[x].values)
    df_extra[x] = lab_enc.transform(df[x].values)


# In[50]:


for x in num:
    df_extra[x] = df[x]


# In[51]:


extra = ExtraTreesClassifier()


# In[52]:


extra.fit(df_extra.values,y)


# In[53]:


extra.feature_importances_


# ## Recuperando a ordem das colunas que foram usadas para treinar o modelo

# In[54]:


c=0
dic={}
for x in df_extra.columns:
    dic[x]=extra.feature_importances_[c]
    c+=1


# In[55]:


dic


# In[56]:


df_imp = pd.DataFrame(dic.items())


# In[57]:


df_imp.sort_values(by=1,ascending=False,inplace=True)


# In[58]:


sns.barplot(data=df_imp,x=1,y=0);


# ## De acordo com esse algoritmo, as melhores 10 variáveis são:

# In[59]:


df_imp.iloc[0:10,0]


# In[60]:


melhores = list(df_imp.iloc[0:10,0])


# ## Criando um novo dataframe com essas variáveis para fazer o treinamento

# In[61]:


df_melhores_extra = df[melhores]


# In[62]:


cat_extra = [x for x in df_melhores_extra.columns if df[x].dtype=='object']


# In[63]:


num_extra = [x for x in df_melhores_extra.columns if x not in cat_extra]


# In[64]:


min_max_extra = MinMaxScaler()


# In[65]:


num_extra_s = min_max_extra.fit_transform(df[num_extra].values)


# In[66]:


onehot_extra = OneHotEncoder(sparse=False)


# In[67]:


cat_extra_cod = onehot_extra.fit_transform(df[cat_extra].values)


# In[68]:


todas_var_extra = np.concatenate([num_extra_s,cat_extra_cod],axis=1)


# In[69]:


resultado_extra = cross_val_score(estimator=XGBClassifier(use_label_encoder=False,eval_metric='logloss'),X=todas_var_extra,                                  y=y_encod,cv=kf,scoring='balanced_accuracy')


# In[70]:


plt.boxplot([x for x in [resultados[len(resultados)-1],resultado_extra]],labels=['Sem Seleção','Com Seleção']);


# In[71]:


x_treino_extra, x_teste_extra, y_treino_extra, y_teste_extra = train_test_split(todas_var_extra,y_encod,test_size=0.3,random_state=1)


# In[72]:


xgb_extra = XGBClassifier(use_label_encoder=False)
xgb_extra.fit(x_treino_extra,y_treino_extra)


# In[73]:


print(classification_report(y_teste_extra,xgb_extra.predict(x_teste_extra)))


# In[74]:


confusao = confusion_matrix(y_teste_extra,xgb_extra.predict(x_teste_extra),labels=[0,1])
print(f'Acurácia classe 0: {confusao[0,0]/(confusao[0,1]+confusao[0,0])*100:.4f}%')
print(f'Acurácia classe 1: {confusao[1,1]/(confusao[1,0]+confusao[1,1])*100:.4f}%')


# ## O modelo piorou

# ## 2) SelectKBest

# In[75]:


score_funcs = [('f_classif',f_classif),('mutual_info_classif',mutual_info_classif), ('chi2',chi2)]


# In[76]:


resultados_kbest = []
for nome,func in score_funcs:
    kbest = SelectKBest(score_func=func,k=10)
    kbest.fit(df_extra.values,y)
    
    cat_kbest = [x for x in df_extra.iloc[:,kbest.get_support()].columns if df[x].dtype=='object']
    num_kbest = [x for x in df_extra.iloc[:,kbest.get_support()].columns if x not in [x for x in df_extra.iloc[:,kbest.get_support()].columns if df[x].dtype=='object']]
    
    minmax_kbest = MinMaxScaler()
    var_num_kbest_s = minmax_kbest.fit_transform(df[num_kbest].values)
    
    onehot_kbest = OneHotEncoder(sparse=False)
    var_cat_kbest = onehot_kbest.fit_transform(df[cat_kbest].values)
    
    todas_var_kbest = np.concatenate([var_num_kbest_s,var_cat_kbest],axis=1)
    
    resultados_kbest.append(cross_val_score(estimator=XGBClassifier(use_label_encoder=False,eval_metric='logloss'),X=todas_var_kbest,                                  y=y_encod,cv=kf,scoring='balanced_accuracy'))


# In[77]:


nomes_kbest = [x[0] for x in score_funcs]


# In[78]:


kbest_comparacao = [list(x) for x in resultados_kbest]


# In[79]:


kbest_comparacao.append(list(resultados[len(resultados)-1]))


# In[80]:


plt.boxplot([x for x in kbest_comparacao],labels=nomes_kbest+['Sem Seleção']);


# ## Apesar de parecer que o mutual_info_classif foi melhor, ele não foi, analisei caso a caso dividindo em treino e teste e o melhor foi o f_classif

# In[81]:


kbest = SelectKBest(score_func=f_classif,k=10)
kbest.fit(df_extra.values,y)
    
cat_kbest = [x for x in df_extra.iloc[:,kbest.get_support()].columns if df[x].dtype=='object']
num_kbest = [x for x in df_extra.iloc[:,kbest.get_support()].columns if x not in [x for x in df_extra.iloc[:,kbest.get_support()].columns if df[x].dtype=='object']]
    
minmax_kbest = MinMaxScaler()
var_num_kbest_s = minmax_kbest.fit_transform(df[num_kbest].values)
    
onehot_kbest = OneHotEncoder(sparse=False)
var_cat_kbest = onehot_kbest.fit_transform(df[cat_kbest].values)
    
todas_var_kbest = np.concatenate([var_num_kbest_s,var_cat_kbest],axis=1)


# In[82]:


x_treino_kbest, x_teste_kbest, y_treino_kbest, y_teste_kbest = train_test_split(todas_var_kbest,y_encod,test_size=0.3,random_state=1)


# In[83]:


xgb_kbest = XGBClassifier(use_label_encoder=False,eval_metric='logloss')
xgb_kbest.fit(x_treino_kbest,y_treino_kbest)


# In[84]:


print(classification_report(y_teste_kbest,xgb_kbest.predict(x_teste_kbest)))


# In[85]:


confusao = confusion_matrix(y_teste_kbest,xgb_kbest.predict(x_teste_kbest),labels=[0,1])
print(f'Acurácia classe 0: {confusao[0,0]/(confusao[0,1]+confusao[0,0])*100:.4f}%')
print(f'Acurácia classe 1: {confusao[1,1]/(confusao[1,0]+confusao[1,1])*100:.4f}%')


# ## O resultado foi pior que com o ExtraTrees

# ## 3) RFE (Recursive Feature Elimination)

# In[86]:


rfe = RFE(XGBClassifier(use_label_encoder=False,eval_metric='logloss'),n_features_to_select=10)


# In[87]:


rfe.fit(df_extra.values,y_encod)


# In[88]:


rfe.support_


# In[89]:


cat_rfe = [x for x in df_extra.iloc[:,rfe.get_support()].columns if df[x].dtype=='object']
num_rfe = [x for x in df_extra.iloc[:,rfe.get_support()].columns if x not in cat_rfe]


# In[90]:


minmax_rfe = MinMaxScaler()
var_num_rfe_s = minmax_rfe.fit_transform(df[num_rfe].values)
    
onehot_rfe = OneHotEncoder(sparse=False)
var_cat_rfe = onehot_rfe.fit_transform(df[cat_rfe].values)
    
todas_var_rfe = np.concatenate([var_num_rfe_s,var_cat_rfe],axis=1)


# In[91]:


resultado_rfe = cross_val_score(estimator=XGBClassifier(use_label_encoder=False,eval_metric='logloss'),X=todas_var_rfe,                                  y=y_encod,cv=kf,scoring='balanced_accuracy')


# In[92]:


plt.boxplot([x for x in [resultados[len(resultados)-1],resultado_rfe]],labels=['Sem Seleção','Com Seleção RFE']);


# In[93]:


x_treino_rfe, x_teste_rfe, y_treino_rfe, y_teste_rfe = train_test_split(todas_var_rfe,y_encod,test_size=0.3,random_state=1)


# In[94]:


xgb_rfe = XGBClassifier(use_label_encoder=False,eval_metric='logloss')
xgb_rfe.fit(x_treino_rfe,y_treino_rfe)


# In[95]:


print(classification_report(y_teste_rfe,xgb_rfe.predict(x_teste_rfe)))


# In[96]:


confusao = confusion_matrix(y_teste_rfe,xgb_rfe.predict(x_teste_rfe),labels=[0,1])
print(f'Acurácia classe 0: {confusao[0,0]/(confusao[0,1]+confusao[0,0])*100:.4f}%')
print(f'Acurácia classe 1: {confusao[1,1]/(confusao[1,0]+confusao[1,1])*100:.4f}%')


# ## O RFE é bem superior aos demais métodos

# ## As melhores 10 variáveis para este problema são:

# In[97]:


cat_rfe+num_rfe


# ## Otimizando os hiperparâmetros

# In[98]:


parametros = {'max_depth':list(range(3,10)),'n_estimators':[100,250],'learning_rate':[0.1,0.2,0.3],             'colsample_bytree':[0.5,0.6,0.7,0.8,0.9],'subsample':[0.6,0.7,0.8]}


# In[99]:


todas_var_rfe_over, y_encod_over = smote.fit_resample(todas_var_rfe,y_encod)


# In[100]:


rand_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False,eval_metric='logloss'),cv=kf,                                 scoring='balanced_accuracy',n_jobs=-1,param_distributions=parametros,verbose=2,                                n_iter=1000)

rand_search.fit(todas_var_rfe_over,y_encod_over)


# In[101]:


rand_search.best_params_


# ## Preparando os dados de teste

# In[ ]:


df_teste = pd.read_csv('projeto4_telecom_teste.csv')


# In[104]:


y_teste = df_teste['churn']


# In[105]:


y_encod_teste = []
for x in y_teste:
    if x=='yes':
        y_encod_teste.append(1)
    else:
        y_encod_teste.append(0)


# In[106]:


df_teste = df_teste[cat_rfe+num_rfe]


# In[107]:


cat_teste = [x for x in df_teste.columns if df_teste[x].dtype=='object']


# In[108]:


num_teste = [x for x in df_teste.columns if x not in cat_teste]


# In[109]:


cat_teste_cod = onehot_rfe.transform(df_teste[cat_teste].values)


# In[110]:


num_teste_s = minmax_rfe.transform(df_teste[num_teste].values)


# In[111]:


todas_var_teste = np.concatenate([num_teste_s,cat_teste_cod],axis=1)


# In[112]:


print(classification_report(y_encod_teste,rand_search.predict(todas_var_teste)))


# In[113]:


confusao = confusion_matrix(y_encod_teste,rand_search.predict(todas_var_teste),labels=[0,1])


# In[114]:


print(f'Acurácia classe 0: {confusao[0,0]/(confusao[0,1]+confusao[0,0])*100:.4f}%')


# In[115]:


print(f'Acurácia classe 1: {confusao[1,1]/(confusao[1,0]+confusao[1,1])*100:.4f}%')


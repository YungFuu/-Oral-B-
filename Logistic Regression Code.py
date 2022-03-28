# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 01:08:00 2022

@author: hp
"""
import pandas as pd
import numpy as np
import os,spacy,re
from tqdm import tqdm
from collections import defaultdict
from spacy.lang.en.stop_words import STOP_WORDS

#Environment
path=r'C:\Users\hp\Desktop\MFIN7036\DeepDiver'
nlp=spacy.load("en_core_web_sm")

#%%
#Product attribute impact analysis based on logistic regression
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv(path+os.sep+'amazon_oralb.csv')
rating_list,battery,price,function,sound,power,charge,quality,design,brand,delivery,effect,user_experience = ([] for i in range(13))

def dummy_generation(review):
    battery_word=['battery','lithium','batteries']
    price_word=['price','expensive','cheap','cheaply','money', 'inexpensive','budget','costly']
    function_word=['function','functionality','mode','timing','timer','charging light']
    sound_word=['sound','loud','louder','squeaky','noise','noisy']
    power_word=['power','vibration','rotate','oscillation','powered','spinning']
    charge_word=['charger','charge','recharge','recharging', 'rechargeable','volt']
    quality_word=['quality','waterproof']
    design_word=['design','pretty','color','style','compact','size','looking','heavy','lightweight','weight']
    brand_word=['brand','phillip','sonic','sonicare','soniccare','philip' ,'oralb']
    dilivery_word=['deliver','delivery','shipping','ship']
    effect_word=['clear','cleaner','cleanliness','effective', 'effect']
    user_experience_word = ['feel','feels','felt', 'soft','rough','comfortable','uncomfortable','harsh']
    
    
    if True in list(map(lambda x:x in review ,battery_word)):
        battery=1
    else:
        battery=0
        
    if True in list(map(lambda x:x in review ,price_word)):
        price=1
    else:
        price=0
        
    if True in list(map(lambda x:x in review ,function_word)):
        function=1
    else:
        function=0
        
    if True in list(map(lambda x:x in review ,sound_word)):
        sound=1
    else:
        sound=0
        
    if True in list(map(lambda x:x in review ,power_word)):
        power=1
    else:
        power=0
        
    if True in list(map(lambda x:x in review ,charge_word)):
        charge=1
    else:
        charge=0
    
    if True in list(map(lambda x:x in review ,quality_word)):
        quality=1
    else:
        quality=0
    
    if True in list(map(lambda x:x in review ,design_word)):
        design=1
    else:
        design=0
        
    if True in list(map(lambda x:x in review ,brand_word)): 
        brand=1
    else:
        brand=0
        
    if True in list(map(lambda x:x in review ,dilivery_word)):
        delivery=1
    else:
        delivery=0
    
    if True in list(map(lambda x:x in review ,effect_word)):
        effect=1
    else:
        effect=0
    
    if True in list(map(lambda x:x in review ,user_experience_word)):
        user_experience=1
    else:
        user_experience=0
    
    return battery,price,function,sound,power,charge,quality,design,brand,delivery,effect,user_experience

def draw_curve(fpr,tpr,roc_auc,save_name):
###make a plot of roc curve
    plt.figure(dpi=150)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(save_name)
    plt.legend(loc="lower right")
    plt.savefig(path+os.sep+save_name+'.jpg')
    plt.show()
    print('Figure was saved to ' + path)

df['std_length']=preprocessing.scale(df['sentence_length'])

for rating in df['rating']:
    if rating == 4 or rating == 5:
        rating_list.append(1)
    elif rating == 3:
        rating_list.append(0)
    elif rating == 1 or rating == 2:
        rating_list.append(0)
        
for review in tqdm(df['content']):
    review=word_replace(review)
    text = nlp(review)
    #select token
    token_list = []
    for token in text: 
        token_list.append(token.lemma_)

    filtered =[] 
    for word in token_list: 
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False and lexeme.is_punct == False:
            filtered.append(word)
            
    filtered = [re.sub(r"[^A-Za-z@]", "", word) for word in filtered]
    filtered = [word for word in filtered if word!='']
        
    dummy_tuple=dummy_generation(filtered)
    
    for wordlist,dummy in zip([battery,price,function,sound,power,charge,quality,design,brand,delivery,effect,user_experience],dummy_tuple):
        wordlist.append(dummy)
    
df['rating']=rating_list
df['battery']=battery
df['price']=price
df['function']=function
df['sound']=sound
df['power']=power
df['charge']=charge
df['quality']=quality
df['design']=design
df['brand']=brand
df['delivery']=delivery
df['effect']=effect
df['user_experience'] = user_experience


pools=['battery',
       'price',
       'function',
       'sound',
       'charge',
       'brand',
       'effect',
       'user_experience',
       'std_length']

#%%

X=df[pools]
y=df['rating']

LR = LogisticRegression(penalty="l1",solver= 'liblinear',class_weight='balanced',tol=0.008,max_iter=100000)
lr_model=LR.fit(X,y)
lr_model1 = sm.Logit(y,sm.add_constant(X)).fit()
print(lr_model1.summary())
predicted_prob = lr_model.predict_proba(X)
predicted_default_prob= predicted_prob[:,1]
fpr, tpr, _ = roc_curve(y, predicted_default_prob)
roc_auc = auc(fpr, tpr)
print('Variables: ', list(X.columns))
print('No. of Variables: ' , len(X.columns))
print('the AUC Value: ' , roc_auc)
draw_curve(fpr,tpr,roc_auc,'Product Attributes - Scoring Model')

def draw_variablesimportance(save_name="Variances Importances"):
    plt.rcParams['axes.unicode_minus']=False 
    coef_LR = pd.Series(lr_model.coef_[0][:-1].flatten(),index = pools[:-1],name = 'Var')
    plt.figure(figsize=(8,4.5),dpi=150)
    coef_LR.sort_values().plot(kind='barh')
    plt.title("Variances Importances")
    plt.savefig(path+os.sep+save_name+'.png')
    plt.show()


import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns', None)

output1=pd.read_csv('C:/Users/Dr.Farhan/Desktop/CE888/data/Questionnaire_datasetIB.csv',sep = ",", encoding='latin')
output2=pd.read_csv('C:/Users/Dr.Farhan/Desktop/CE888/data/Questionnaire_datasetIA.csv',sep = ",", encoding='latin')

final_score = output1['Total Score original'] + output2['Total Score original']

for n in range(0,60):
    if  final_score[n] > 190:
        final_score[n] = 1
    else:
        final_score[n] = 0


d = {'Participant no': [], 'Recording no': [], 'Pupil diameter succades': [], 'Pupil diameter for fixations': [],'Max Pupil diameter for succades': [],'Max Pupil diameter for fixations': [],'Succade duration': [],'Fixation duration': [], 'Empythy score': []}

df_new_control = pd.DataFrame(data=d)

df_new_test = pd.DataFrame(data=d)
for i in range(1,7):
    if i <= 9:
        data=pd.read_csv('D:/CE888/raw data/raw_data/Participant00' + str(0) + str(i) + '.tsv',sep='\t')
        data1=pd.read_csv('D:/CE888/raw data/raw_data/Participant00' + str(0) + str(i) + '.tsv',sep='\t')
    else:
        data=pd.read_csv('D:/CE888/raw data/raw_data/Participant00' +  str(i) + '.tsv',sep='\t')
        data1=pd.read_csv('D:/CE888/raw data/raw_data/Participant00' +  str(i) + '.tsv',sep='\t')
        
    last_row = data.iloc[-1]
    
    s1=re.sub("Recording","",last_row['Recording name'])
    s1 = int(s1)
    
    
    
    for j in range(1,s1+1):
        
        data2 = data[['Recording name','Pupil diameter left','Pupil diameter right','Gaze event duration','Eye movement type']]
        data3 = data1[['Recording name','Pupil diameter left','Pupil diameter right','Gaze event duration','Eye movement type']]
        
        data2 = data2.dropna(axis=0)
        
        filter = data2["Recording name"]=="Recording"+str(j)+""
        data2.where(filter, inplace = True)
        
        data2 = data2.dropna(axis=0)
        
        filter = data2["Eye movement type"]=="Saccade"
        data2.where(filter, inplace = True)
        
        data2 = data2.dropna(axis=0)
        
        data3 = data3.dropna(axis=0)
        
        filter = data3["Recording name"]=="Recording"+str(j)+""
        data3.where(filter, inplace = True)
        
        data3 = data3.dropna(axis=0)
        
        filter = data3["Eye movement type"]=="Fixation"
        data3.where(filter, inplace = True)
        
        data3 = data3.dropna(axis=0)
        
        data2 = data2[["Pupil diameter left","Pupil diameter right","Gaze event duration"]]
        
        def fix_commas(data):
            for col in data.select_dtypes(include='object'):
                
                data[col] = data[col].str.replace(',', '.').astype(float)
            return data
        
        data2_test = fix_commas(data2)
        
        data3 = data3[["Pupil diameter left","Pupil diameter right","Gaze event duration"]]
        
        data3_test = fix_commas(data3)
        
        gms = data2_test['Gaze event duration'].mean()

        gmf = data3_test['Gaze event duration'].mean()

        num = float(data2_test['Pupil diameter left'].mean())
        num1 = float(data2_test['Pupil diameter right'].mean())
        num_sp = num + num1

        num = float(data3_test['Pupil diameter left'].mean())
        num1 = float(data3_test['Pupil diameter right'].mean())
        num_fp = num + num1
        num_fp

        num = float(data2_test['Pupil diameter left'].max())
        num1 = float(data2_test['Pupil diameter right'].max())
        num_sm = num + num1
        num_sm

        num = float(data3_test['Pupil diameter left'].max())
        num1 = float(data3_test['Pupil diameter right'].max())
        num_fm = num + num1
        num_fm
        
            
            
            
    
        if i%2 == 0:
            df_new_control = df_new_control.append({'Participant no': i, 'Recording no': j,'Pupil diameter succades': gms, 'Pupil diameter for fixations': gmf,'Max Pupil diameter for succades': num_sp, 'Max Pupil diameter for fixations': num_fp,'Succade duration': num_sm, 'Fixation duration': num_fm, 'Empythy score': final_score[i-1]}, ignore_index=True)
        else:
            df_new_test = df_new_test.append({'Participant no': i, 'Recording no': j,'Pupil diameter succades': gms, 'Pupil diameter for fixations': gmf,'Max Pupil diameter for succades': num_sp, 'Max Pupil diameter for fixations': num_fp,'Succade duration': num_sm, 'Fixation duration': num_fm, 'Empythy score': final_score[i-1]}, ignore_index=True)
            
        df_new_control
        df_new_test


X1 = df_new_test[['Pupil diameter succades','Pupil diameter for fixations','Max Pupil diameter for succades','Max Pupil diameter for fixations','Succade duration','Fixation duration' ]]

Y1 = df_new_test['Empythy score']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.20,
                                                    random_state=23)

log_reg1 = LogisticRegression(random_state=0)
log_reg1.fit(X_train1, y_train1)

y_pred1 = log_reg1.predict(X_test1)
  
accuracy1 = accuracy_score(y_test1, y_pred1)
print("Logistic Regression model accuracy (in %) for test group:", accuracy1*100)

X = df_new_control[['Pupil diameter succades','Pupil diameter for fixations','Max Pupil diameter for succades','Max Pupil diameter for fixations','Succade duration','Fixation duration' ]]
Y = df_new_control['Empythy score']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,random_state=23)
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
  
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy (in %) for control group:", accuracy*100)

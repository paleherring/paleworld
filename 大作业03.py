import pandas  as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
a=pd.read_csv("churn-bigml-80.csv").dropna().drop_duplicates()
def special_process(a):
    lab=LabelEncoder()
    a["International plan"]=lab.fit_transform(a["International plan"])
    a["Voice mail plan"]=lab.fit_transform(a["Voice mail plan"])
    train_y=lab.fit_transform(a["Churn"])
    del a["Churn"]
    del a["Account length"]
    
    a["Customer service calls"]=a["Customer service calls"].map(lambda x:a["Customer service calls"].mean() if x>=7 else x)
    a["Total day calls"]=a["Total day calls"].map(lambda x:a["Total day calls"].mean() if x<10 else x)
    a["Total eve calls"]=a["Total eve calls"].map(lambda x:a["Total eve calls"].mean() if x<25 else x)
    a["Total eve charge"]=a["Total eve charge"].map(lambda x:a["Total eve charge"].mean() if x<5 else x)
    
    sta=StandardScaler()
    sta.fit(a)
    a=sta.fit_transform(a)
    train_x=pd.DataFrame(a)
    return train_x,train_y

test=pd.read_csv("churn-bigml-20.csv")
train_x,train_y=special_process(a)
test_x,test_y=special_process(test)
feature_train, feature_test, target_train, target_test =train_test_split(train_x,train_y,test_size=0.3,random_state=0)
dtc=DecisionTreeClassifier()
dtc.fit(feature_train,target_train)
target_produce=dtc.predict(feature_test)
score=accuracy_score(target_test,target_produce)
print(score)            

target_produce=dtc.predict(test_x)
score=accuracy_score(test_y,target_produce)
print(score)


import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv('Admission_Prediction.csv')
df['GRE Score'].fillna(df['GRE Score'].mean(),inplace=True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mean(),inplace=True)
df['University Rating'].fillna(df['University Rating'].mean(),inplace=True)
df.drop(columns=['Serial No.'],inplace=True)

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
sc = StandardScaler()


st.title('Admission Data Science Project')
gre_score = st.number_input('Gre_Score')
TOFEL_SCORE=st.number_input('TOFEL Score')
University_Rating=st.number_input('University Input',min_value=0.00,max_value=5.00)
SOP=st.number_input('SOP',min_value=0,max_value=5)
LOR=st.number_input("LOR",min_value=0,max_value=5)
CGPA=st.number_input("CGPA",max_value=10.00)
Research= st.selectbox('Research',(0,1))
#st.write('The current number is ', number)

user_input=[[gre_score,TOFEL_SCORE,University_Rating,SOP,LOR,CGPA,Research]]
sc.fit(x)
Scaled_user_input=sc.transform(user_input)
st.write(user_input)
st.write(Scaled_user_input)
loaded_model=pickle.load(open('lr_for_admission','rb'))
result=loaded_model.predict(Scaled_user_input)

if st.button("Predict"):
    result_percentage= result*100
    st.header("Percentage of you getting admitted In University is"+str(result_percentage))
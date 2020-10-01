import streamlit as st
import joblib 
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn 
import catboost 


file = open('Financial.joblib','rb')
model = joblib.load(file)

st.title("FINANCIAL INCLUSION IN AFRICA")
st.subheader("Which Individuals are most likely to have or use a Bank Account?")
st.write("Financial Inclusion remains one of the main obstacles to economic and human development in Africa. For example, across Kenya, Rwanda, Tanzania, and Uganda only 9.1 million adults (or 13.9% of the adult population) have access to or use a commercial bank account.  Traditionally, access to bank accounts has been regarded as an indicator of financial inclusion. Despite the proliferation of mobile money in Africa, and the growth of innovative fintech solutions, banks still play a pivotal role in facilitating access to financial services. Access to bank accounts enable households to save and facilitate payments while also helping businesses build up their credit-worthiness and improve their access to other finance services. Therefore, access to bank accounts is an essential contributor to long-term economic growth.")

html_temp = """
    <div style ='background-color: darkkhaki; padding:20px;'>
    <h1><b><center>Streamlit ML Web App</center></b></h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.write('Please provide the following details for an Individual')
country = st.sidebar.selectbox('Country: 0 for Kenya, 1 for Rwanda, 2 for Tanzania, 3 for Uganda',('0','1','2','3'))
year = st.sidebar.selectbox('Year',('2016','2017','2018'))
location_type = st.sidebar.selectbox('Location_type: 0 for Rural, 1 for Urban',('0','1'))
cellphone_access = st.sidebar.selectbox('Cellphone_access: 0 for No, 1 for Yes',('0','1'))
household_size = st.sidebar.slider('Household_size',0,21,1)
age_of_respondent = st.sidebar.slider('Age_of_Respondent',0,120,1)
gender_of_respondent = st.sidebar.selectbox('Gender_of_Respondent: 0 for Male, 1 for Female',('0','1'))
relationship_with_head= st.sidebar.selectbox('Relationship_with_Head: 0 for Head of Household, 1 for Spouse, 2 for Child, 3 for Parent, 4 for Other relative, 5 for Other non-relatives',('0','1','2','3','4','5'))
marital_status = st.sidebar.selectbox('Marital_Status: 0 for Married/Living together, 1 for Single/Never Married, 2 for Widowed, 3 for Divorced/Seperated, 4 for Dont know ',('0','1', '2', '3', '4'))
education_level = st.sidebar.selectbox('Education_level: 0 for Primary education, 1 for No formal education, 2 for Secondary education, 3 for Tertiary education, 4 for Vocational/Specialised training, 5 for Other/Dont know/RTA',('0','1','2','3','4','5'))
job_type = st.sidebar.selectbox('Job_Type: 0 for Self employed, 1 for Informally employed, 2 for Farming and Fishing, 3 for Remittance Dependent, 4 for Other Income, 5 for Formally employed Private, 6 for No Income, 7 for Formally employed Government, 8 for Government Dependent, 9 for Dont Know/Refuse to answer',('0','1','2','3','4','5','6','7','8','9'))


features = {'country':country,
'year':year,
'location_type':location_type,
'cellphone_access':cellphone_access,
'household_size':household_size,
'age_of_respondent':age_of_respondent,
'gender_of_respondent':gender_of_respondent,
'relationship_with_head':relationship_with_head,
'marital_status':marital_status,
'education_level':education_level,
'job_type':job_type
}

if st.button('Submit'):
    data = pd.DataFrame(features,index=[0,1])
    st.write(data)

    prediction = model.predict(data)
    proba = model.predict_proba(data)[1]

    if prediction[0] == 0:
        st.error('Individual do not use or have a bank account')
    else:
        st.success('Individual uses or has a bank account')

    proba_df = pd.DataFrame(proba,columns=['Probability'],index=['Bank_Account:No','Bank_Account:Yes'])
    proba_df.plot(kind='barh')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
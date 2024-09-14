from PIL import Image

import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import shap
import catboost
from catboost import CatBoostClassifier
import plotly.figure_factory as ff
import matplotlib.pyplot as plt


plt.style.use('default')

st.set_page_config(
    page_title = '“Alert level HABs” forecasting Model of Yilong Lake',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)


# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>“Alert level HABs” prediction of Yilong Lake</h1>", unsafe_allow_html=True)



# side-bar
def user_input_features():
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.slider('pH', 7.0, 10.0, 8.0)
    a2 = st.sidebar.slider('Cond', 27.0, 102.0, 50.0)
    a3 = st.sidebar.slider('CODMn', 3.0, 51.0, 30.0)
    a4 = st.sidebar.slider('CODCr', 18.0, 253.0, 100.0)
    a5 = st.sidebar.slider('TP', 0.004, 0.41, 0.5)
    a6 = st.sidebar.slider('TN', 0.5, 9.0, 5.0)
    a7 = st.sidebar.slider('SD', 0.05, 2.0, 1.0)
    a8 = st.sidebar.slider('WV', 1600.0, 11600.0, 5000.0)

    output = [a1, a2, a3, a4, a5, a6, a7, a8]
    return output


outputdf = user_input_features()





import pandas as pd
# 读取数据(csv)
df = pd.read_excel('fraud1.xlsx')


# 需要一个count plot
placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()




shapdatadf =pd.read_excel(r'shapdatadf1.xlsx')
shapvaluedf =pd.read_excel(r'shapvaluedf1.xlsx')




catmodel = CatBoostClassifier()
catmodel.load_model("fraud1")

outputdf = pd.DataFrame([outputdf], columns=shapdatadf.columns)

# st.write('User input parameters below ⬇️')
# st.write(outputdf)
p1 = catmodel.predict(outputdf)[0]
p2 = catmodel.predict_proba(outputdf)


placeholder6 = st.empty()
with placeholder6.container():
    f1, f2 = st.columns(2)
    with f1:
        st.title('')
        st.title('')
        st.title('')
        st.title('')
        st.write('User input parameters below ⬇️')
        st.write(outputdf)
        st.title('')
        st.write('Predicted class Probability')
        st.write('0️⃣ means “Non-Alert level HABs”')
        st.write('1️⃣ means “Alert level HABs”')
        st.write(p2)

    with f2:
        st.title('')
        st.title('')
        st.title('')
        st.title('')
        explainer = shap.Explainer(catmodel)
        shap_values = explainer(outputdf)

        # st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
        shap.plots.waterfall(shap_values[0])
        st.pyplot(bbox_inches='tight')


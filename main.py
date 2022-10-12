import streamlit as st
import pandas as pd


st.title(" Modèle de prédiction des types d'accidents routiers")

st.sidebar.header("Les paramètres d'entrée")

def user_input():
    Mois=st.sidebar.slider("Le Mois de l'accident",1,4,2)
    Jour=st.sidebar.slider("Le jour de l'accident",1,2,1)
    Heure=st.sidebar.slider("L'heure de l'accident",1,4,3)
    Type_route=st.sidebar.slider("Le type de route",1,3,2)
    Vitesse_limite=st.sidebar.slider("La vitesse limite",1,2,1)
    Detail_jonction=st.sidebar.slider("Detail du jonction",1,6,2)
    Control_jonction=st.sidebar.slider("Controle du jonction",1,2,1)
    Condition_eclairage=st.sidebar.slider("Les conditions d'éclairage de la route",1,2,1)
    Condition_meteo=st.sidebar.slider("Les conditions de la météo",1,3,2)
    Condition_surface_routiere=st.sidebar.slider("Les conditions de la surface routière",1,2,1)
    Region=st.sidebar.slider("Lieu de l'accident",1,2,1)

    dictionnaire={'Mois':Mois, 'Jour':Jour, 'Heure':Heure,'Type_route':Type_route,'Vitesse_limite':Vitesse_limite,
     'Detail_jonction':Detail_jonction,'Control_jonction':Control_jonction,'Condition_eclairage':Condition_eclairage,
          'Condition_meteo':Condition_meteo,'Condition_surface_routiere':Condition_surface_routiere,
           'Region': Region   }
    accident_parametres=pd.DataFrame(dictionnaire,index=[0])

    return accident_parametres
df=user_input()
st.subheader("Nous souhaitons connaitre le type de l'accident suivant :")
st.write(df)

import pickle
lr_from_pickle=pickle.load(open("modele_a_rendre.sav","rb"))
y_predd=lr_from_pickle.predict(df)       # La fonction de prediction


st.subheader("Le type de l'accident est: ")
if y_predd==0:
   st.write(y_predd[0])
   st.write("Donc c'est du type fatal/sérieux avec une probabilité de :\n")
   st.write(lr_from_pickle.predict_proba(df)[0][0])
else:
    st.write(y_predd[0])
    st.write("Donc c'est du type léger avec une probabilité de :\n")
    st.write(lr_from_pickle.predict_proba(df)[0][1])
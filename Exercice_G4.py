from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv('covid_19_data.txt', sep=",")

df = load_data()

# 1. Sélection des colonnes numériques
cols = ['Cas_Confirmes', 'Deces', 'Tests_Realises', 'Taux_Occupation_Hosp']
data_to_scale = df[cols]

st.subheader("🛠️ Préparation des données (Scaling)")

option = st.selectbox("Choisis la méthode", ["Données Brutes", "Normalisation (0-1)", "Standardisation (Moyenne 0)"])

if option == "Normalisation (0-1)":
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data_to_scale), columns=cols)
    st.write("Données compressées entre 0 et 1.")
    
elif option == "Standardisation (Moyenne 0)":
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data_to_scale), columns=cols)
    st.write("Données centrées sur 0 avec écart-type de 1.")

else:
    df_scaled = data_to_scale

st.dataframe(df_scaled.head())

# --- VISUALISATION 2 : BOXPLOT (Dispersion) ---
st.write("###  Analyse de la dispersion (Boxplot)")
fig_box, ax_box = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_scaled, ax=ax_box, palette="Set2")
ax_box.set_title(f"Distribution après {option}")
st.pyplot(fig_box)

# --- VISUALISATION 3 : LINE CHART (Évolution) ---
st.write("###  Comparaison de l'évolution (Line Chart)")
# On utilise st.line_chart pour une visualisation interactive rapide
st.line_chart(df_scaled)

# Optionnel : Version Matplotlib pour plus de contrôle sur le design
fig_line, ax_line = plt.subplots(figsize=(10, 5))
df_scaled.plot(ax=ax_line)
ax_line.set_title("Superposition des tendances")
st.pyplot(fig_line)
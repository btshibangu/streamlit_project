
import streamlit as st
import numpy as np
import pandas as pd
import ast
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import FeatureHasher

# --- Stopwords (anglais + français) ---
nltk.download('stopwords')
stop_words_EF = stopwords.words('english') + stopwords.words('french')

# --- Interface Streamlit ---
st.title("🎬 Recommandation de films intelligente")
st.image("findyournextmovie.png")

# --- Chargement du dataset ---
df = pd.read_parquet("df_machine_learning (4).parquet")
df = df.reset_index(drop=True)

# --- Prétraitement identique au code ML initial ---

# Colonnes numériques
df_numeric = df.select_dtypes(include=['number'])
scaler = StandardScaler()
df_numeric_scale = scaler.fit_transform(df_numeric)

# NLP sur overview avec TF-IDF
tfidf = TfidfVectorizer(stop_words=stop_words_EF, max_features=300)
overview_enc = tfidf.fit_transform(df['overview'].fillna(""))

# Encodage des colonnes de type liste
df['genres'] = df['genres'].apply(lambda x: [x] if isinstance(x, str) else x)
df['actors'] = df['actors'].apply(lambda x: [x] if isinstance(x, str) else x)
df['directors'] = df['directors'].apply(lambda x: [x] if isinstance(x, str) else x)
df['production_countries'] = df['production_countries'].apply(ast.literal_eval)


mlb = MultiLabelBinarizer()
genres_enc = mlb.fit_transform(df['genres'])
countries_enc = mlb.fit_transform(df['production_countries'])
hasher_directeurs = FeatureHasher(n_features=128, input_type='string')
director_enc = hasher_directeurs.fit_transform(df['directors'])
hasher_acteurs = FeatureHasher(n_features=256, input_type='string')
actor_enc = hasher_acteurs.fit_transform(df['actors'])


# Fusion des features
data = csr_matrix(hstack([
    overview_enc,
    genres_enc,
    countries_enc,
    director_enc,
    actor_enc,
    df_numeric_scale
]))

# --- Entraînement du modèle Nearest Neighbors ---
NNmodel = NearestNeighbors(n_neighbors=6, metric='minkowski')
NNmodel.fit(data)

# --- Interface utilisateur pour choix de film ---
film_titre = st.selectbox("📽️ Choisissez un film :", df["title"].dropna().sort_values().unique())
film_index = df[df["title"] == film_titre].index[0]

# --- Affichage des infos du film ---
film = df.iloc[film_index]
st.subheader("🎞️ Film sélectionné")
st.write(f"**Titre :** {film['title']}")
st.write(f"**Genres :** {film['genres']}")
st.write(f"**Année :** {film['startYear']}")
st.write(f"**Pays :** {film['production_countries']}")
st.write(f"**Acteurs :** {film['actorsName']}")
st.write("**Synopsis :**", film["overview"])

# --- Recommandation ---
recommandation = NNmodel.kneighbors(data[film_index], return_distance=False)
st.subheader("🎯 Films similaires recommandés")
for i in recommandation[0][1:]:
    reco = df.iloc[i]
    st.markdown(f"**🎬 {reco['title']}** - *{reco['genres']}*")

st.markdown("---")






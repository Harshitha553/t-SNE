import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

st.set_page_config(
    page_title="Spotify t-SNE Analysis",
    layout="wide"
)

st.title("🎵 Spotify Tracks Analysis using t-SNE")

# Load Dataset

DATA_FILE = Path(__file__).resolve().parent / "spotify.csv"

df = pd.read_csv(DATA_FILE)

df = df.dropna()

# Metrics

col1,col2,col3 = st.columns(3)

with col1:
    st.metric(
        "Tracks",
        len(df)
    )

with col2:
    st.metric(
        "Genres",
        df["track_genre"].nunique()
    )

with col3:
    st.metric(
        "Artists",
        df["artists"].nunique()
    )

# Dataset Preview

st.subheader("Dataset Preview")

st.dataframe(df.head())

# Genre Distribution

st.subheader("Top Genres")

top_genres = (
    df["track_genre"]
    .value_counts()
    .head(10)
)

fig, ax = plt.subplots(
    figsize=(10,5)
)

top_genres.plot(
    kind="bar",
    ax=ax
)

st.pyplot(fig)

# Popularity Distribution

st.subheader("Popularity Distribution")

fig, ax = plt.subplots()

sns.histplot(
    df["popularity"],
    kde=True,
    ax=ax
)

st.pyplot(fig)

# Sidebar

st.sidebar.header("t-SNE Settings")

sample_size = st.sidebar.slider(
    "Sample Size",
    1000,
    5000,
    3000
)

perplexity = st.sidebar.slider(
    "Perplexity",
    5,
    50,
    30
)

if st.button("Run t-SNE"):

    features = [
        "popularity",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]

    sample_df = df.sample(
        sample_size,
        random_state=42
    )

    X = sample_df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    with st.spinner(
        "Running t-SNE..."
    ):

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity
        )

        X_tsne = tsne.fit_transform(
            X_scaled
        )

    st.success(
        "t-SNE Completed"
    )

    # t-SNE Plot

    st.subheader(
        "t-SNE Visualization"
    )

    fig, ax = plt.subplots(
        figsize=(10,7)
    )

    scatter = ax.scatter(
        X_tsne[:,0],
        X_tsne[:,1],
        alpha=0.6
    )

    ax.set_title(
        "Spotify Tracks t-SNE"
    )

    st.pyplot(fig)

    # Genre Plot

    st.subheader(
        "Genre Clusters"
    )

    fig, ax = plt.subplots(
        figsize=(12,8)
    )

    sns.scatterplot(
        x=X_tsne[:,0],
        y=X_tsne[:,1],
        hue=sample_df["track_genre"],
        legend=False,
        ax=ax
    )

    st.pyplot(fig)

    # Reduced Dataset

    tsne_df = pd.DataFrame({
        "TSNE1": X_tsne[:,0],
        "TSNE2": X_tsne[:,1],
        "Genre": sample_df[
            "track_genre"
        ].values
    })

    st.subheader(
        "Reduced Dataset"
    )

    st.dataframe(
        tsne_df.head()
    )

    csv = tsne_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download t-SNE Results",
        csv,
        "spotify_tsne.csv",
        "text/csv"
    )
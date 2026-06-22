import pickle
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import streamlit as st
import os
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
import tempfile

st.title("🎵 Audio Fingerprinting & Song Recognition System")
st.markdown("""
This application implements an audio fingerprinting system inspired by modern music recognition platforms.

Features:
- Single Song Identification
- Batch Song Identification
- Spectrogram Visualization
- Constellation Map Generation
- Offset Histogram Analysis
""")

# -------------------------------------------------
# Fingerprint Extraction
# -------------------------------------------------

def extract_fingerprints(song_path):

    y, sr = librosa.load(
        song_path,
        sr=22050,
        duration=20
    )

    S = np.abs(librosa.stft(y))

    local_max = (
        S == maximum_filter(S, size=20)
    )

    threshold = np.percentile(S, 99)

    peaks = local_max & (S > threshold)

    freq_idx, time_idx = np.where(peaks)

    fingerprints = []

    for i in range(len(freq_idx) - 5):

        f1 = int(freq_idx[i])
        t1 = int(time_idx[i])

        for j in range(1, 6):

            f2 = int(freq_idx[i + j])
            t2 = int(time_idx[i + j])

            dt = t2 - t1

            if dt > 0:

                fingerprints.append(
                    (f1, f2, dt)
                )

    return fingerprints

def plot_spectrogram(song_path):

    y, sr = librosa.load(
     song_path,
     sr=22050,
     duration=20
     )

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y)),
        ref=np.max
    )

    fig, ax = plt.subplots(
        figsize=(10,4)
    )

    librosa.display.specshow(
        D,
        sr=sr,
        x_axis='time',
        y_axis='log',
        ax=ax
    )

    ax.set_title(
        "Spectrogram"
    )

    return fig

def plot_constellation(song_path):

    y, sr = librosa.load(
    song_path,
    sr=22050,
    duration=20
    )

    S = np.abs(
        librosa.stft(y)
    )

    local_max = (
        S ==
        maximum_filter(
            S,
            size=20
        )
    )

    threshold = np.percentile(
        S,
        99
    )

    peaks = (
        local_max &
        (S > threshold)
    )

    freq_idx, time_idx = np.where(
        peaks
    )

    fig, ax = plt.subplots(
        figsize=(10,4)
    )

    ax.imshow(
        20*np.log10(S + 1e-6),
        origin='lower',
        aspect='auto'
    )

    ax.scatter(
        time_idx,
        freq_idx,
        s=8
    )

    ax.set_title(
        "Constellation Map"
    )

    return fig

def compute_offsets(
    query_fp,
    db_fp
):

    offsets = []

    db_lookup = {
        fp:i
        for i, fp in enumerate(db_fp)
    }

    for i, fp in enumerate(query_fp):

        if fp in db_lookup:

            offsets.append(
                db_lookup[fp] - i
            )

    return offsets

def plot_offset_histogram(
    offsets
):

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.hist(
        offsets,
        bins=50
    )

    ax.set_title(
        "Offset Histogram"
    )

    ax.set_xlabel(
        "Offset"
    )

    ax.set_ylabel(
        "Count"
    )

    return fig


# -------------------------------------------------
# Build Database
# -------------------------------------------------

@st.cache_resource
def build_database():

    song_database = {}

    songs_folder = "../Songs"

    files = [
        f for f in os.listdir(songs_folder)
        if f.endswith(".mp3")
    ]

    progress = st.progress(0)

    for i, song in enumerate(files):

        path = os.path.join(
            songs_folder,
            song
        )

        song_database[song] = (
            extract_fingerprints(path)
        )
        progress.progress(
            (i + 1) / len(files)
        )
    progress.empty()

    return song_database


# -------------------------------------------------
# Matching Function
# -------------------------------------------------
def match_song(
    query_fingerprints,
    song_database
):

    scores = {}

    query_set = set(
        query_fingerprints
    )

    for song in song_database:

        db_set = set(
            song_database[song]
        )

        matches = len(
            query_set.intersection(
                db_set
            )
        )

        scores[song] = matches

    best_song = max(
        scores,
        key=scores.get
    )

    return best_song, scores

# -------------------------------------------------
# Load Database
# -------------------------------------------------

st.write(
    "Loading Database..."
)

with open(
    "database.pkl",
    "rb"
) as f:

    song_database = pickle.load(f)

st.success(
    f"{len(song_database)} Songs Indexed"
)

st.sidebar.header(
    "System Statistics"
)

st.sidebar.write(
    f"Songs Indexed: {len(song_database)}"
)

st.sidebar.write(
    "Fingerprint Threshold: 99th Percentile"
)
st.sidebar.write(
    "Database Size: 50 Songs"
)

st.sidebar.write(
    "Matching Method: Audio Fingerprinting"
)


# -------------------------------------------------
# Upload Song
# -------------------------------------------------

st.markdown("---")
st.header("Single Song Mode")

uploaded_file = st.file_uploader(
    "Upload MP3 File",
    type=["mp3"]
)


# -------------------------------------------------
# Recognition
# -------------------------------------------------

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        temp_path = tmp.name

    y, sr = librosa.load(
        temp_path,
        sr=22050,
        duration=20
    )

    duration = len(y)/sr

    st.subheader(
        "Song Information"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Sampling Rate",
            sr
        )

    with col2:
        st.metric(
            "Duration (s)",
            f"{duration:.2f}"
        )

    with st.expander(
        "View Spectrogram"
    ):

        st.pyplot(
            plot_spectrogram(
                temp_path
            )
        )

    with st.expander(
        "View Constellation Map"
    ):

        st.pyplot(
            plot_constellation(
                temp_path
            )
        )

    if st.button(
    "Identify Song"
    ):

        query_fp = (
            extract_fingerprints(
                temp_path
            )
        )

        prediction, scores = (
            match_song(
                query_fp,
                song_database
            )
        )

        db_fp = song_database[
            prediction
        ]

        offsets = compute_offsets(
        query_fp,
        db_fp
        )

        clean_prediction = prediction.replace(
            ".mp3",
            ""
        )

        st.success(
            f"Predicted Song: {clean_prediction}"
        )

        st.subheader(
            "Offset Histogram"
        )

        st.pyplot(
            plot_offset_histogram(
                offsets
            )
        )

        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader(
            "Top 5 Matches"
        )

        for song, score in sorted_scores[:5]:

            st.write(
                f"{song} : {score}"
            )

st.markdown("---")

st.header("Batch Mode")

batch_files = st.file_uploader(
    "Upload Multiple MP3 Files",
    type=["mp3"],
    accept_multiple_files=True
    )
if batch_files:

    if st.button("Run Batch Recognition"):

        results = []

        progress = st.progress(0)

        total_files = len(batch_files)

        for idx, uploaded_song in enumerate(batch_files):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp3"
            ) as tmp:

                tmp.write(
                    uploaded_song.read()
                )

                temp_path = tmp.name

            query_fp = extract_fingerprints(
                temp_path
            )

            prediction, scores = match_song(
                query_fp,
                song_database
            )

            prediction = prediction.replace(
                ".mp3",
                ""
            )

            results.append(
                [
                    uploaded_song.name,
                    prediction
                ]
            )

            progress.progress(
                (idx + 1) / total_files
            )

        df = pd.DataFrame(
            results,
            columns=[
                "filename",
                "prediction"
            ]
        )

        csv = df.to_csv(
            index=False
        )

        st.success(
            "Batch Processing Complete"
        )

        st.dataframe(df)

        st.download_button(
            label="Download results.csv",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )
# -------------------------------------------------
# Load Database
# -------------------------------------------------

import os

loading_box = st.empty()

loading_box.info(
    "Loading fingerprint database..."
)

if not os.path.exists("database.pkl"):

    st.error(
        "database.pkl not found."
    )

    st.stop()

with open(
    "database.pkl",
    "rb"
) as f:

    song_database = pickle.load(f)

loading_box.empty()

st.success(
    f"{len(song_database)} Songs Indexed"
)

with st.expander(
    "View Indexed Songs"
):

    for song in sorted(
        song_database.keys()
    ):

        st.write(song)
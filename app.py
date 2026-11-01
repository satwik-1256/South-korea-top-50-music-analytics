# ============================================================
# SOUTH KOREA TOP 50 ANALYTICS DASHBOARD
# PART 1: IMPORTS, CONFIGURATION & PROJECT CONSTANTS
# ============================================================

# STANDARD PYTHON LIBRARIES
import os
import warnings
import textwrap
from pathlib import Path
from datetime import datetime, timedelta

# DATA ANALYSIS LIBRARIES
import pandas as pd
import numpy as np

# DATA VISUALIZATION LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# STREAMLIT
import streamlit as st

# MACHINE LEARNING LIBRARIES
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance

# WARNING CONFIGURATION
warnings.filterwarnings("ignore")

# STREAMLIT PAGE CONFIGURATION
st.set_page_config(
    page_title="South Korea Top 50 Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROJECT INFORMATION
PROJECT_NAME = (
    "Comeback Momentum, Chart Re-Entry, "
    "and Fandom Intensity Analysis"
)

PROJECT_SUBTITLE = "South Korea Top 50 Playlist"

DATASET_NAME = "Atlantic South Korea Top 50"

DEVELOPER_NAME = "Satwik Srivastava"

UNIVERSITY_NAME = "Babu Banarasi Das University, Lucknow"

COURSE_NAME = (
    "BCA — Data Science & Artificial Intelligence"
)

PROJECT_YEAR = "2026"

# COLOR PALETTE
PRIMARY_COLOR = "#172554"
SECONDARY_COLOR = "#312E81"
ACCENT_COLOR = "#4338CA"
HIGHLIGHT_COLOR = "#6366F1"
SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#D97706"
DANGER_COLOR = "#DC2626"

BACKGROUND_COLOR = "#F4F7FC"
CARD_COLOR = "#FFFFFF"
TEXT_COLOR = "#172554"
MUTED_COLOR = "#64748B"

# PLOTLY CONFIGURATION
PLOTLY_TEMPLATE = "plotly_white"

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True
}

# MATPLOTLIB CONFIGURATION
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["font.size"] = 10

# SEABORN CONFIGURATION
sns.set_theme(
    style="whitegrid",
    context="notebook"
)

# APPLICATION CONSTANTS
DATASET_FILENAME = "Atlantic_South_Korea.csv"

EXCLUDED_DATE = pd.Timestamp("2025-03-01")

EXPECTED_CHART_SIZE = 50

# ============================================================
# END OF PART 1
# ============================================================

# ============================================================
# PART 2: PREMIUM LIGHT THEME & DASHBOARD STYLING
# ============================================================

st.markdown(textwrap.dedent("""
<style>

/* GLOBAL APPLICATION */

.stApp {
    background-color: #F4F7FC;
    color: #172554;
    font-family: 'Segoe UI', Arial, sans-serif;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #07162F 0%,
        #10264B 55%,
        #172E53 100%
    );
    border-right: 1px solid #263E65;
}

section[data-testid="stSidebar"] > div {
    padding: 18px 15px;
}

/* SIDEBAR HEADING VISIBILITY */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 700;
}

/* SIDEBAR BRANDING */

.sidebar-brand {
    background: linear-gradient(
        135deg,
        #1D3970,
        #312E81
    );
    border: 1px solid #526BA8;
    border-radius: 16px;
    padding: 22px 12px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.18);
}

.sidebar-brand h2 {
    color: #FFFFFF !important;
    font-size: 23px;
    font-weight: 850;
}

.sidebar-brand p {
    color: #DBEAFE !important;
    font-size: 12px;
}

/* SIDEBAR INPUTS */

section[data-testid="stSidebar"] input {
    border-radius: 9px !important;
}

section[data-testid="stSidebar"] input::placeholder {
    color: #B8C8E3 !important;
}

/* SELECTBOX AND MULTISELECT */

section[data-testid="stSidebar"]
div[data-baseweb="select"] > div {
    background-color: #172E53 !important;
    color: #FFFFFF !important;
    border: 1px solid #526B94 !important;
    border-radius: 9px !important;
    min-height: 44px;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] span {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* MULTISELECT TAGS */

section[data-testid="stSidebar"]
span[data-baseweb="tag"] {
    background-color: #354F7E !important;
    color: #FFFFFF !important;
}

/* DROPDOWN MENUS */

div[data-baseweb="popover"],
div[data-baseweb="menu"] {
    background-color: #172E53 !important;
}

div[data-baseweb="menu"] li {
    color: #FFFFFF !important;
}

/* DATE PICKER */

section[data-testid="stSidebar"]
div[data-testid="stDateInput"] input {
    background-color: #FFFFFF !important;
    color: #003FAD !important;
    -webkit-text-fill-color: #003FAD !important;
    font-weight: 800 !important;
    font-size: 15px !important;
}

section[data-testid="stSidebar"]
div[data-testid="stDateInput"] svg {
    color: #003FAD !important;
    fill: #003FAD !important;
}

/* SLIDERS */

section[data-testid="stSidebar"]
[data-testid="stSlider"] {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"]
[data-testid="stSlider"] label {
    color: #FFFFFF !important;
}

/* RADIO NAVIGATION */

section[data-testid="stSidebar"]
[data-testid="stRadio"] label {
    color: #FFFFFF !important;
    font-weight: 600;
}

/* PROJECT HERO */

.project-hero {
    background: linear-gradient(
        120deg,
        #0B1B35,
        #192F63,
        #3730A3
    );
    padding: 42px 35px;
    border-radius: 22px;
    color: #FFFFFF;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(15,23,42,0.15);
}

.project-hero h1 {
    color: #FFFFFF !important;
    font-size: 34px;
    font-weight: 800;
}

.project-hero p {
    color: #DBEAFE !important;
    font-size: 16px;
}

/* SECTION HEADINGS */

.section-heading {
    background: linear-gradient(
        90deg,
        #172B50,
        #3730A3
    );
    color: #FFFFFF !important;
    padding: 15px 20px;
    border-radius: 12px;
    margin-top: 22px;
    margin-bottom: 18px;
    font-size: 19px;
    font-weight: 700;
}

/* CONTENT CARDS */

.content-card {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 5px 15px rgba(15,23,42,0.05);
    margin-bottom: 18px;
}

/* KPI CARDS */

.kpi-card {
    background: #FFFFFF;
    padding: 22px 18px;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #3730A3;
    box-shadow: 0 5px 16px rgba(15,23,42,0.06);
    min-height: 115px;
    margin-bottom: 15px;
}

.kpi-title {
    color: #64748B;
    font-size: 13px;
    font-weight: 600;
}

.kpi-value {
    color: #172554;
    font-size: 27px;
    font-weight: 800;
}

/* DATAFRAMES */

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
}

/* EXPANDERS */

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    margin-bottom: 12px;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(
        100deg,
        #1E40AF,
        #4338CA
    );
    color: #FFFFFF !important;
    border: 1px solid #647FEA;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 650;
}

.stButton > button:hover {
    background: linear-gradient(
        100deg,
        #1D4ED8,
        #4F46E5
    );
}

/* DOWNLOAD BUTTONS */

.stDownloadButton > button {
    background: linear-gradient(
        100deg,
        #0F766E,
        #0D9488
    );
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-weight: 650;
}

/* PROFESSIONAL MAIN PAGE FOOTER */

.dashboard-footer {
    background: #164DA5 !important;
    color: #FFFFFF !important;
    width: 100%;
    min-height: 280px;
    padding: 42px 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 45px;
    margin-bottom: 10px;
    box-sizing: border-box;
}

.dashboard-footer h1,
.dashboard-footer h2,
.dashboard-footer h3,
.dashboard-footer p {
    color: #FFFFFF !important;
}

.dashboard-footer h1 {
    font-size: clamp(23px, 3vw, 40px);
    font-weight: 800;
    margin-bottom: 30px;
}

.dashboard-footer h2 {
    font-size: 26px;
    font-weight: 750;
}

.dashboard-footer h3 {
    font-size: 18px;
    font-weight: 650;
}

.dashboard-footer p {
    font-size: 16px;
    margin-bottom: 22px;
}

/* RESPONSIVE DESIGN */

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .project-hero {
        padding: 25px 20px;
    }

    .project-hero h1 {
        font-size: 24px;
    }

    .dashboard-footer {
        padding: 30px 15px;
    }

    .dashboard-footer h1 {
        font-size: 25px;
    }
}

</style>
"""), unsafe_allow_html=True)

# ============================================================
# END OF PART 2
# ============================================================

# ============================================================
# PART 3: DATASET LOADING, CLEANING & VALIDATION
# ============================================================


# ------------------------------------------------------------
# 1. REQUIRED DATASET COLUMNS
# ------------------------------------------------------------

REQUIRED_COLUMNS = [
    "date",
    "position",
    "song",
    "artist",
    "popularity",
    "duration_ms",
    "album_type",
    "total_tracks",
    "is_explicit",
    "album_cover_url"
]


# ------------------------------------------------------------
# 2. DATASET FILE LOCATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / DATASET_FILENAME

DATASET_SOURCE = str(DATASET_PATH)

DATA_CONNECTED = False


# ------------------------------------------------------------
# 3. LOAD DATASET DIRECTLY
# ------------------------------------------------------------

@st.cache_data(show_spinner="Loading South Korea Top 50 dataset...")
def load_dataset(dataset_file_path):

    return pd.read_csv(dataset_file_path)


if not DATASET_PATH.is_file():

    st.error(
        "❌ Dataset file not found."
    )

    st.warning(
        f"Expected dataset location: {DATASET_PATH}"
    )

    st.info(
        "Please place Atlantic_South_Korea.csv "
        "in the same folder as app.py."
    )

    st.stop()


try:

    raw_df = load_dataset(str(DATASET_PATH))

    DATA_CONNECTED = True

except Exception as error:

    st.error(
        "Unable to read the South Korea Top 50 dataset."
    )

    st.exception(error)

    st.stop()


# ------------------------------------------------------------
# 4. RAW DATASET INFORMATION
# ------------------------------------------------------------

raw_row_count = len(raw_df)

raw_column_count = len(raw_df.columns)


# ------------------------------------------------------------
# 5. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in raw_df.columns
]

if missing_columns:

    st.error(
        "Dataset structure is incomplete."
    )

    st.write(
        "Missing columns:",
        missing_columns
    )

    st.stop()


# ------------------------------------------------------------
# 6. CREATE CLEANING COPY
# ------------------------------------------------------------

chart_df = raw_df.copy()


# ------------------------------------------------------------
# 7. REMOVE EXACT DUPLICATE RECORDS
# ------------------------------------------------------------

duplicates_before = int(
    chart_df.duplicated().sum()
)

chart_df = (
    chart_df
    .drop_duplicates()
    .copy()
)

duplicates_removed = duplicates_before


# ------------------------------------------------------------
# 8. CONVERT DATE COLUMN
# ------------------------------------------------------------

chart_df["date"] = pd.to_datetime(
    chart_df["date"],
    errors="coerce"
)

invalid_date_count = int(
    chart_df["date"].isna().sum()
)

if invalid_date_count > 0:

    st.warning(
        f"{invalid_date_count} records contain invalid dates. "
        "These records are retained for transparency, "
        "but cannot participate in date-based analysis."
    )


# ------------------------------------------------------------
# 9. CONVERT NUMERIC COLUMNS
# ------------------------------------------------------------

numeric_columns = [
    "position",
    "popularity",
    "duration_ms",
    "total_tracks"
]

for column in numeric_columns:

    chart_df[column] = pd.to_numeric(
        chart_df[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# 10. NORMALIZE TEXT COLUMNS
# ------------------------------------------------------------

text_columns = [
    "song",
    "artist",
    "album_type",
    "album_cover_url"
]

for column in text_columns:

    chart_df[column] = (
        chart_df[column]
        .astype("string")
        .str.strip()
    )


# ------------------------------------------------------------
# 11. NORMALIZE EXPLICIT COLUMN
# ------------------------------------------------------------

def normalize_explicit(value):

    if pd.isna(value):

        return np.nan

    if isinstance(value, (bool, np.bool_)):

        return bool(value)

    normalized = str(value).strip().lower()

    if normalized in [
        "true",
        "1",
        "yes",
        "explicit"
    ]:

        return True

    if normalized in [
        "false",
        "0",
        "no",
        "clean"
    ]:

        return False

    return np.nan


chart_df["is_explicit"] = (
    chart_df["is_explicit"]
    .apply(normalize_explicit)
)


# ------------------------------------------------------------
# 12. EXCLUDE UNRESOLVED SNAPSHOT DATE
# ------------------------------------------------------------

excluded_date_rows = int(
    (
        chart_df["date"].dt.normalize()
        == EXCLUDED_DATE
    ).sum()
)

chart_df = chart_df[
    chart_df["date"].dt.normalize()
    != EXCLUDED_DATE
].copy()


# ------------------------------------------------------------
# 13. CREATE SONG IDENTIFIER
# ------------------------------------------------------------

chart_df["song_id"] = (

    chart_df["song"]
    .fillna("Unknown Song")
    .astype(str)

    + " | "

    + chart_df["artist"]
    .fillna("Unknown Artist")
    .astype(str)

)


# ------------------------------------------------------------
# 14. SORT DATASET
# ------------------------------------------------------------

chart_df = chart_df.sort_values(

    by=[
        "date",
        "position",
        "song"
    ],

    ascending=[
        True,
        True,
        True
    ],

    na_position="last"

).reset_index(drop=True)


# ------------------------------------------------------------
# 15. CREATE MAIN ANALYTICS DATAFRAME
# ------------------------------------------------------------

chart = chart_df.copy()


# ------------------------------------------------------------
# 16. DAILY CHART VALIDATION
# ------------------------------------------------------------

valid_date_df = chart_df[
    chart_df["date"].notna()
].copy()


daily_chart_counts = (
    valid_date_df
    .groupby(
        valid_date_df["date"].dt.normalize()
    )
    .size()
)


daily_unique_songs = (
    valid_date_df
    .groupby(
        valid_date_df["date"].dt.normalize()
    )["song_id"]
    .nunique()
)


daily_unique_positions = (
    valid_date_df
    .groupby(
        valid_date_df["date"].dt.normalize()
    )["position"]
    .nunique()
)


incomplete_dates = daily_chart_counts[
    daily_chart_counts != EXPECTED_CHART_SIZE
]


duplicate_song_dates = daily_unique_songs[
    daily_unique_songs != EXPECTED_CHART_SIZE
]


duplicate_position_dates = daily_unique_positions[
    daily_unique_positions != EXPECTED_CHART_SIZE
]


# ------------------------------------------------------------
# 17. VALIDATION CHECKS
# ------------------------------------------------------------

validation_checks = {

    "Required columns present":
        len(missing_columns) == 0,

    "Dataset successfully loaded":
        DATA_CONNECTED,

    "Exact duplicates removed":
        duplicates_removed >= 0,

    "No invalid dates":
        invalid_date_count == 0,

    "Excluded date removed":
        not (
            chart_df["date"].dt.normalize()
            == EXCLUDED_DATE
        ).any(),

    "All retained dates have 50 records":
        len(incomplete_dates) == 0,

    "All retained dates have 50 unique songs":
        len(duplicate_song_dates) == 0,

    "All retained dates have 50 unique positions":
        len(duplicate_position_dates) == 0,

    "Chart positions are numeric":
        chart_df["position"].notna().all(),

    "Song identifiers created":
        "song_id" in chart_df.columns

}


# ------------------------------------------------------------
# 18. CLEANING SUMMARY
# ------------------------------------------------------------

cleaning_summary = {

    "Raw Rows":
        raw_row_count,

    "Raw Columns":
        raw_column_count,

    "Exact Duplicates Removed":
        duplicates_removed,

    "Rows After Duplicate Removal":
        raw_row_count - duplicates_removed,

    "Excluded Date":
        str(EXCLUDED_DATE.date()),

    "Rows Excluded From Snapshot":
        excluded_date_rows,

    "Final Rows":
        len(chart_df),

    "Final Columns":
        len(chart_df.columns),

    "Unique Chart Dates":
        chart_df["date"].nunique(),

    "Unique Songs":
        chart_df["song_id"].nunique(),

    "Unique Artists":
        chart_df["artist"].nunique(),

    "Invalid Dates":
        invalid_date_count,

    "Incomplete Chart Dates":
        len(incomplete_dates),

    "Duplicate Song Dates":
        len(duplicate_song_dates),

    "Duplicate Position Dates":
        len(duplicate_position_dates)

}


# ------------------------------------------------------------
# 19. DATASET DATE RANGE
# ------------------------------------------------------------

if chart_df["date"].notna().any():

    DATA_START_DATE = (
        chart_df["date"].min().date()
    )

    DATA_END_DATE = (
        chart_df["date"].max().date()
    )

else:

    DATA_START_DATE = None

    DATA_END_DATE = None


# ------------------------------------------------------------
# 13. PROJECT HERO SECTION
# ------------------------------------------------------------

st.html(textwrap.dedent(textwrap.dedent(f"""
        <div class="project-hero">

            <div style="
                font-size:13px;
                letter-spacing:3px;
                font-weight:600;
                margin-bottom:12px;
            ">
                SOUTH KOREA TOP 50 • MUSIC INTELLIGENCE
            </div>

            <h1>
                {PROJECT_NAME}
            </h1>

            <p>
                {PROJECT_SUBTITLE}
            </p>

            <div style="
                margin-top:18px;
                font-size:13px;
                opacity:0.85;
            ">
                An interactive data analytics and
                machine learning project
            </div>

        </div>
    """)))


# ------------------------------------------------------------

# 20. DISPLAY DATASET STATUS
# ------------------------------------------------------------

st.html(textwrap.dedent("""
    <div class="section-heading">
        📊 Dataset Connection Status
    </div>
    """))


status_col1, status_col2, status_col3 = st.columns(3)


with status_col1:

    st.metric(
        "Total Records",
        f"{len(chart_df):,}"
    )


with status_col2:

    st.metric(
        "Unique Songs",
        f"{chart_df['song_id'].nunique():,}"
    )


with status_col3:

    st.metric(
        "Unique Artists",
        f"{chart_df['artist'].nunique():,}"
    )


# ------------------------------------------------------------
# 21. DISPLAY VALIDATION RESULTS
# ------------------------------------------------------------

with st.expander(
    "🔍 View Dataset Validation Report",
    expanded=False
):

    validation_df = pd.DataFrame(

        [
            {
                "Validation Check": check,

                "Status": (
                    "PASS"
                    if result
                    else "WARNING"
                )
            }

            for check, result
            in validation_checks.items()
        ]

    )

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True
    )

    if all(validation_checks.values()):

        st.success(
            "All dataset validation checks passed."
        )

    else:

        st.warning(
            "Some validation checks require attention. "
            "The dataset has not been silently modified "
            "to force these checks to pass."
        )


# ------------------------------------------------------------
# 22. DISPLAY CLEANING SUMMARY
# ------------------------------------------------------------

with st.expander(
    "🧹 View Data Cleaning Summary",
    expanded=False
):

    cleaning_df = pd.DataFrame(

        list(cleaning_summary.items()),

        columns=[
            "Cleaning Metric",
            "Value"
        ]

    )

    st.dataframe(
        cleaning_df,
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------
# 23. DATASET PREVIEW
# ------------------------------------------------------------

with st.expander(
    "👀 Preview Processed Dataset",
    expanded=False
):

    st.dataframe(
        chart_df.head(10),
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------
# 24. FINAL DATASET CONFIRMATION
# ------------------------------------------------------------

if DATA_CONNECTED:

    st.success(
        "✅ South Korea Top 50 dataset loaded successfully."
    )

    if DATA_START_DATE and DATA_END_DATE:

        st.caption(
            f"Available date range: "
            f"{DATA_START_DATE} to {DATA_END_DATE}"
        )


# ============================================================
# END OF PART 3
# ============================================================

# ============================================================
# PART 4: SIDEBAR, NAVIGATION & GLOBAL FILTERS
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT NAVIGATION OPTIONS
# ------------------------------------------------------------

PAGES = [
    "🏠 Executive Dashboard",
    "🎤 Chart Re-Entry",
    "🚀 Momentum Analytics",
    "⏳ Retention Analytics",
    "💜 Fandom Intelligence",
    "🤖 Machine Learning",
    "📊 Executive Insights",
    "🎶 Song Explorer",
    "📁 Project Information"
]


# ------------------------------------------------------------
# 2. INITIALIZE GLOBAL VARIABLES
# ------------------------------------------------------------

selected_dates = ()
selected_artists = []
song_search = "All songs"

selected_position = (1, EXPECTED_CHART_SIZE)
selected_popularity = (0, 100)

filtered = chart_df.copy()

page = PAGES[0]


# ------------------------------------------------------------
# 3. RESET GLOBAL FILTER FUNCTION
# ------------------------------------------------------------

def reset_global_filters():

    reset_keys = [
        "global_date_filter",
        "global_artist_filter",
        "global_song_select",
        "global_position_filter",
        "global_popularity_filter"
    ]

    for key in reset_keys:

        if key in st.session_state:
            del st.session_state[key]


# ------------------------------------------------------------
# 4. PREPARE AVAILABLE FILTER VALUES
# ------------------------------------------------------------

available_dates = (
    chart_df["date"]
    .dropna()
    .dt.normalize()
    .dt.date
)


available_artists = sorted(
    chart_df["artist"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


available_songs = sorted(
    chart_df["song"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


valid_positions = (
    pd.to_numeric(
        chart_df["position"],
        errors="coerce"
    )
    .dropna()
)


valid_popularity = (
    pd.to_numeric(
        chart_df["popularity"],
        errors="coerce"
    )
    .dropna()
)


# ------------------------------------------------------------
# 5. SIDEBAR AND NAVIGATION
# ------------------------------------------------------------

with st.sidebar:

    # SIDEBAR BRANDING

    st.html(textwrap.dedent(textwrap.dedent("""
            <div class="sidebar-brand">
                <div style="font-size:34px;">🎵</div>

                <h2 style="margin:5px 0;">
                    K-POP ANALYTICS
                </h2>

                <p style="margin:0; font-size:12px;">
                    South Korea Top 50 Intelligence
                </p>
            </div>
        """)))

    st.markdown("---")

    # DASHBOARD NAVIGATION

    st.markdown("### 🧭 Dashboard Navigation")

    page = st.radio(
        "Select Dashboard Page",
        options=PAGES,
        index=0,
        key="dashboard_navigation",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # GLOBAL FILTERS

    st.markdown("### 🎛️ Global Filters")

    st.caption(
        "These filters apply across the analytical pages."
    )

    # DATE RANGE FILTER

    st.markdown("#### 📅 Chart Date Range")

    if not available_dates.empty:

        minimum_date = min(available_dates)
        maximum_date = max(available_dates)

        selected_dates = st.date_input(
            "Select starting and ending dates",
            value=(),
            min_value=minimum_date,
            max_value=maximum_date,
            key="global_date_filter",
            help=(
                "Select a starting date and ending date. "
                "Leave empty to include all available dates."
            )
        )

    else:

        selected_dates = ()

        st.warning(
            "No valid chart dates are available."
        )

    # ARTIST FILTER

    st.markdown("#### 🎤 Artist Filter")

    selected_artists = st.multiselect(
        "Select Artists",
        options=available_artists,
        default=[],
        key="global_artist_filter",
        placeholder="All artists"
    )

    # SONG FILTER

    st.markdown("#### 🎵 Song Filter")

    song_options = [
        "All songs"
    ] + available_songs

    song_search = st.selectbox(
        "Search Song",
        options=song_options,
        index=0,
        key="global_song_select",
        help=(
            "Select a song from the complete "
            "South Korea Top 50 dataset."
        )
    )

    # CHART POSITION FILTER

    st.markdown("#### 🏆 Chart Position")

    if not valid_positions.empty:

        minimum_position = int(
            valid_positions.min()
        )

        maximum_position = int(
            valid_positions.max()
        )

        selected_position = st.slider(
            "Select Chart Position",
            min_value=minimum_position,
            max_value=maximum_position,
            value=(
                minimum_position,
                maximum_position
            ),
            key="global_position_filter"
        )

    else:

        selected_position = (
            1,
            EXPECTED_CHART_SIZE
        )

    # POPULARITY FILTER

    st.markdown("#### 🔥 Popularity Score")

    if not valid_popularity.empty:

        minimum_popularity = int(
            np.floor(valid_popularity.min())
        )

        maximum_popularity = int(
            np.ceil(valid_popularity.max())
        )

        selected_popularity = st.slider(
            "Select Popularity Range",
            min_value=minimum_popularity,
            max_value=maximum_popularity,
            value=(
                minimum_popularity,
                maximum_popularity
            ),
            key="global_popularity_filter"
        )

    else:

        selected_popularity = (0, 100)

    # RESET FILTER BUTTON

    st.markdown("---")

    st.button(
        "🔄 Reset All Filters",
        use_container_width=True,
        key="reset_all_filters",
        on_click=reset_global_filters
    )

# ------------------------------------------------------------
# 6. APPLY GLOBAL FILTERS
# ------------------------------------------------------------

filtered = chart_df.copy()


# ------------------------------------------------------------
# 7. APPLY DATE FILTER
# ------------------------------------------------------------

if (
    isinstance(selected_dates, (tuple, list))
    and len(selected_dates) == 2
    and selected_dates[0] is not None
    and selected_dates[1] is not None
):

    start_date = pd.Timestamp(
        selected_dates[0]
    )

    end_date = pd.Timestamp(
        selected_dates[1]
    )

    filtered = filtered[
        (filtered["date"] >= start_date)
        &
        (
            filtered["date"]
            < end_date + pd.Timedelta(days=1)
        )
    ].copy()


# ------------------------------------------------------------
# 8. APPLY ARTIST FILTER
# ------------------------------------------------------------

if selected_artists:

    filtered = filtered[
        filtered["artist"].isin(selected_artists)
    ].copy()


# ------------------------------------------------------------
# 9. APPLY SONG FILTER
# ------------------------------------------------------------

if song_search != "All songs":

    filtered = filtered[
        filtered["song"] == song_search
    ].copy()


# ------------------------------------------------------------
# 10. APPLY CHART POSITION FILTER
# ------------------------------------------------------------

filtered = filtered[
    pd.to_numeric(
        filtered["position"],
        errors="coerce"
    ).between(
        selected_position[0],
        selected_position[1]
    )
].copy()


# ------------------------------------------------------------
# 11. APPLY POPULARITY FILTER
# ------------------------------------------------------------

filtered = filtered[
    pd.to_numeric(
        filtered["popularity"],
        errors="coerce"
    ).between(
        selected_popularity[0],
        selected_popularity[1]
    )
].copy()


# ------------------------------------------------------------
# 12. RESET FILTERED DATAFRAME INDEX
# ------------------------------------------------------------

filtered = filtered.reset_index(drop=True)


# ------------------------------------------------------------
# 14. CURRENT PAGE INFORMATION
# ------------------------------------------------------------

st.html(textwrap.dedent(textwrap.dedent(f"""
        <div style="
            margin-top:10px;
            margin-bottom:15px;
            padding:12px 18px;
            background:white;
            border-radius:12px;
            border:1px solid #E5EAF2;
        ">

            <span style="
                font-size:13px;
                color:#64748B;
            ">
                CURRENT SECTION
            </span>

            <h3 style="
                margin:4px 0 0 0;
                color:#172554;
            ">
                {page}
            </h3>

        </div>
    """)))


# ------------------------------------------------------------
# 15. GLOBAL FILTER STATUS
# ------------------------------------------------------------

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:

    st.metric(
        "Filtered Records",
        f"{len(filtered):,}"
    )

with filter_col2:

    st.metric(
        "Filtered Songs",
        f"{filtered['song_id'].nunique():,}"
    )

with filter_col3:

    st.metric(
        "Filtered Artists",
        f"{filtered['artist'].nunique():,}"
    )


# ------------------------------------------------------------
# 16. FILTER SUMMARY
# ------------------------------------------------------------

with st.expander(
    "🔍 View Active Filter Summary",
    expanded=False
):

    filter_summary = {

        "Selected Page": page,

        "Date Range": (
            f"{selected_dates[0]} to {selected_dates[1]}"
            if (
                isinstance(selected_dates, (tuple, list))
                and len(selected_dates) == 2
                and selected_dates[0] is not None
                and selected_dates[1] is not None
            )
            else "All available dates"
        ),

        "Selected Artists": (
            ", ".join(selected_artists)
            if selected_artists
            else "All artists"
        ),

        "Selected Song": song_search,

        "Chart Position": (
            f"{selected_position[0]} to {selected_position[1]}"
        ),

        "Popularity Range": (
            f"{selected_popularity[0]} to {selected_popularity[1]}"
        ),

        "Filtered Records": len(filtered)

    }

    filter_summary_df = pd.DataFrame(
        list(filter_summary.items()),
        columns=["Filter", "Selection"]
    )

    st.dataframe(
        filter_summary_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# END OF PART 4
# ============================================================

# ============================================================
# PART 5: EXECUTIVE DASHBOARD
# ============================================================

if page == "🏠 Executive Dashboard":

    st.html(textwrap.dedent("""
        <div class="section-heading">
            📊 Executive Overview
        </div>
        <p>
            Explore chart performance, song activity,
            artist dominance and popularity patterns.
        </p>
        """))

    if filtered.empty:

        st.warning("No records match the selected filters.")

    else:

        dashboard_df = filtered.copy()

        total_records = len(dashboard_df)
        unique_songs = dashboard_df["song_id"].nunique()
        unique_artists = dashboard_df["artist"].nunique()

        average_position = dashboard_df["position"].mean()
        average_popularity = dashboard_df["popularity"].mean()

        total_chart_dates = dashboard_df["date"].nunique()

        top_position_count = (
            dashboard_df["position"] == 1
        ).sum()

        # ----------------------------------------------------
        # KPI SECTION
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🎯 Key Performance Indicators</div>'))

        kpi_data = [
            ("Total Chart Records", f"{total_records:,}"),
            ("Unique Songs", f"{unique_songs:,}"),
            ("Unique Artists", f"{unique_artists:,}"),
            ("Average Popularity", f"{average_popularity:.2f}" if pd.notna(average_popularity) else "N/A"),
            ("Average Chart Position", f"{average_position:.2f}" if pd.notna(average_position) else "N/A"),
            ("Chart Dates", f"{total_chart_dates:,}"),
            ("No. 1 Chart Entries", f"{top_position_count:,}")
        ]

        for start in range(0, len(kpi_data), 4):

            cols = st.columns(4)

            for col, item in zip(cols, kpi_data[start:start + 4]):

                with col:

                    st.metric(item[0], item[1])

        # ----------------------------------------------------
        # DAILY CHART ACTIVITY
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📈 Chart Activity Over Time</div>'))

        daily_activity = (
            dashboard_df
            .groupby(dashboard_df["date"].dt.normalize())
            .agg(
                total_entries=("song_id", "count"),
                unique_songs=("song_id", "nunique"),
                average_position=("position", "mean"),
                average_popularity=("popularity", "mean")
            )
            .reset_index()
            .rename(columns={"date": "chart_date"})
        )

        fig_activity = go.Figure()

        fig_activity.add_trace(
            go.Scatter(
                x=daily_activity["chart_date"],
                y=daily_activity["total_entries"],
                mode="lines",
                name="Chart Records",
                line=dict(color=PRIMARY_COLOR, width=2.5),
                fill="tozeroy"
            )
        )

        fig_activity.update_layout(
            title="Daily Chart Records",
            xaxis_title="Chart Date",
            yaxis_title="Number of Records",
            template=PLOTLY_TEMPLATE,
            height=390,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_activity,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # TOP SONGS AND ARTISTS
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🏆 Leading Songs & Artists</div>'))

        song_col, artist_col = st.columns(2)

        with song_col:

            top_songs = (
                dashboard_df
                .groupby(["song", "artist"], as_index=False)
                .agg(
                    chart_appearances=("date", "count"),
                    average_position=("position", "mean"),
                    average_popularity=("popularity", "mean")
                )
                .sort_values(
                    "chart_appearances",
                    ascending=False
                )
                .head(10)
            )

            if not top_songs.empty:

                fig_top_songs = px.bar(
                    top_songs.sort_values(
                        "chart_appearances",
                        ascending=True
                    ),
                    x="chart_appearances",
                    y="song",
                    orientation="h",
                    color="chart_appearances",
                    hover_data=[
                        "artist",
                        "average_position",
                        "average_popularity"
                    ],
                    title="Top 10 Songs by Chart Appearances",
                    color_continuous_scale="Blues"
                )

                fig_top_songs.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=440
                )

                st.plotly_chart(
                    fig_top_songs,
                    use_container_width=True,
                    config=PLOTLY_CONFIG
                )

        with artist_col:

            top_artists = (
                dashboard_df
                .groupby("artist", as_index=False)
                .agg(
                    chart_appearances=("date", "count"),
                    unique_songs=("song_id", "nunique"),
                    average_position=("position", "mean")
                )
                .sort_values(
                    "chart_appearances",
                    ascending=False
                )
                .head(10)
            )

            if not top_artists.empty:

                fig_top_artists = px.bar(
                    top_artists.sort_values(
                        "chart_appearances",
                        ascending=True
                    ),
                    x="chart_appearances",
                    y="artist",
                    orientation="h",
                    color="chart_appearances",
                    hover_data=[
                        "unique_songs",
                        "average_position"
                    ],
                    title="Top 10 Artists by Chart Appearances",
                    color_continuous_scale="Purples"
                )

                fig_top_artists.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=440
                )

                st.plotly_chart(
                    fig_top_artists,
                    use_container_width=True,
                    config=PLOTLY_CONFIG
                )

        # ----------------------------------------------------
        # POPULARITY AND POSITION DISTRIBUTIONS
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🔥 Popularity & Position Analysis</div>'))

        popularity_col, position_col = st.columns(2)

        with popularity_col:

            popularity_data = dashboard_df.dropna(
                subset=["popularity"]
            )

            if not popularity_data.empty:

                fig_popularity = px.histogram(
                    popularity_data,
                    x="popularity",
                    nbins=25,
                    title="Song Popularity Distribution",
                    color_discrete_sequence=[SECONDARY_COLOR]
                )

                fig_popularity.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=370
                )

                st.plotly_chart(
                    fig_popularity,
                    use_container_width=True,
                    config=PLOTLY_CONFIG
                )

        with position_col:

            position_data = dashboard_df.dropna(
                subset=["position"]
            )

            if not position_data.empty:

                fig_position = px.histogram(
                    position_data,
                    x="position",
                    nbins=25,
                    title="Chart Position Distribution",
                    color_discrete_sequence=[ACCENT_COLOR]
                )

                fig_position.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=370
                )

                st.plotly_chart(
                    fig_position,
                    use_container_width=True,
                    config=PLOTLY_CONFIG
                )

                # ----------------------------------------------------
        # ALBUM VS SINGLE
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">💿 Album vs Single Analysis</div>'))

        album_analysis = (
            dashboard_df
            .assign(
                album_category=(
                    dashboard_df["album_type"]
                    .astype(str)
                    .str.strip()
                    .str.title()
                )
            )
            .groupby("album_category", as_index=False)
            .agg(
                total_records=("song_id", "count"),
                unique_songs=("song_id", "nunique"),
                average_popularity=("popularity", "mean"),
                average_position=("position", "mean")
            )
        )

        album_analysis = album_analysis[
            album_analysis["album_category"].isin(
                ["Album", "Single"]
            )
        ]

        if not album_analysis.empty:

            fig_album = px.bar(
                album_analysis,
                x="album_category",
                y="total_records",
                color="album_category",
                text="total_records",
                hover_data=[
                    "unique_songs",
                    "average_popularity",
                    "average_position"
                ],
                title="Chart Records: Albums vs Singles"
            )

            fig_album.update_layout(
                template=PLOTLY_TEMPLATE,
                height=370,
                showlegend=False
            )

            st.plotly_chart(
                fig_album,
                use_container_width=True,
                config=PLOTLY_CONFIG
            )

        # ----------------------------------------------------
        # EXECUTIVE SUMMARY
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📝 Executive Summary</div>'))

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            if not top_songs.empty:

                leading_song = top_songs.iloc[0]

                st.html(textwrap.dedent(f"""
                    <div class="content-card">
                        <h3>🎵 Leading Song</h3>
                        <h2>{leading_song['song']}</h2>
                        <p>Artist: {leading_song['artist']}</p>
                        <p>
                            Chart appearances:
                            <b>{int(leading_song['chart_appearances'])}</b>
                        </p>
                    </div>
                    """))

        with summary_col2:

            if not top_artists.empty:

                leading_artist = top_artists.iloc[0]

                st.html(textwrap.dedent(f"""
                    <div class="content-card">
                        <h3>🎤 Leading Artist</h3>
                        <h2>{leading_artist['artist']}</h2>
                        <p>
                            Chart appearances:
                            <b>{int(leading_artist['chart_appearances'])}</b>
                        </p>
                        <p>
                            Unique songs:
                            <b>{int(leading_artist['unique_songs'])}</b>
                        </p>
                    </div>
                    """))

        st.info(
            "Chart appearances represent observed chart records. "
            "They are not direct measurements of listeners, streams "
            "or audience loyalty."
        )


# ============================================================
# END OF PART 5
# ============================================================

# ============================================================
# PART 6: CHART RE-ENTRY & COMEBACK ANALYTICS
# ============================================================

if page == "🎤 Chart Re-Entry":

    st.html(textwrap.dedent("""
        <div class="section-heading">
            🎤 Chart Re-Entry & Comeback Intelligence
        </div>
        <p>
            Identify songs that return to the South Korea Top 50
            after previously disappearing from the chart.
        </p>
        """))

    # --------------------------------------------------------
    # 1. PREPARE COMPLETE CHART HISTORY
    # --------------------------------------------------------

    reentry_source = chart_df.copy()

    reentry_source["date_key"] = (
        reentry_source["date"].dt.normalize()
    )

    observed_dates = sorted(
        reentry_source["date_key"].dropna().unique()
    )

    observed_date_index = {
        date: index
        for index, date in enumerate(observed_dates)
    }

    # --------------------------------------------------------
    # 2. IDENTIFY COMEBACK EVENTS
    # --------------------------------------------------------

    comeback_events = []

    song_histories = (
        reentry_source
        .sort_values(["song_id", "date_key", "position"])
        .groupby("song_id")
    )

    for song_id, song_history in song_histories:

        song_history = (
            song_history
            .drop_duplicates(subset=["date_key"])
            .sort_values("date_key")
            .reset_index(drop=True)
        )

        if len(song_history) < 2:
            continue

        previous_record = None

        previous_appearances = 0

        for _, current_record in song_history.iterrows():

            current_date = current_record["date_key"]

            if previous_record is not None:

                previous_date = previous_record["date_key"]

                previous_index = observed_date_index[
                    previous_date
                ]

                current_index = observed_date_index[
                    current_date
                ]

                observed_dates_absent = (
                    current_index - previous_index - 1
                )

                if observed_dates_absent > 0:

                    days_absent = (
                        current_date - previous_date
                    ).days

                    comeback_events.append({

                        "song_id": song_id,

                        "song": current_record["song"],

                        "artist": current_record["artist"],

                        "comeback_date": current_date,

                        "previous_chart_date": previous_date,

                        "days_absent": days_absent,

                        "observed_dates_absent":
                            observed_dates_absent,

                        "previous_position":
                            previous_record["position"],

                        "comeback_position":
                            current_record["position"],

                        "position_change": (
                            previous_record["position"]
                            - current_record["position"]
                        ),

                        "previous_popularity":
                            previous_record["popularity"],

                        "popularity":
                            current_record["popularity"],

                        "previous_appearances":
                            previous_appearances,

                        "album_type":
                            current_record["album_type"],

                        "is_explicit":
                            current_record["is_explicit"],

                        "duration_ms":
                            current_record["duration_ms"],

                        "total_tracks":
                            current_record["total_tracks"],

                        "album_cover_url":
                            current_record["album_cover_url"]

                    })

            previous_record = current_record

            previous_appearances += 1

    # --------------------------------------------------------
    # 3. CREATE COMEBACK DATAFRAME
    # --------------------------------------------------------

    reentry_df = pd.DataFrame(comeback_events)

    if not reentry_df.empty:

        reentry_df["comeback_date"] = pd.to_datetime(
            reentry_df["comeback_date"]
        )

        reentry_df["previous_chart_date"] = pd.to_datetime(
            reentry_df["previous_chart_date"]
        )

        reentry_df = reentry_df.sort_values(
            "comeback_date"
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. APPLY GLOBAL FILTERS
    # --------------------------------------------------------

    filtered_reentries = reentry_df.copy()

    if not filtered_reentries.empty:

        if (
            isinstance(selected_dates, (tuple, list))
            and len(selected_dates) == 2
            and selected_dates[0] is not None
            and selected_dates[1] is not None
        ):

            start_date = pd.Timestamp(selected_dates[0])

            end_date = (
                pd.Timestamp(selected_dates[1])
                + pd.Timedelta(days=1)
            )

            filtered_reentries = filtered_reentries[
                (
                    filtered_reentries["comeback_date"]
                    >= start_date
                )
                &
                (
                    filtered_reentries["comeback_date"]
                    < end_date
                )
            ].copy()

        if selected_artists:

            filtered_reentries = filtered_reentries[
                filtered_reentries["artist"].isin(
                    selected_artists
                )
            ].copy()

        if song_search != "All songs":

            filtered_reentries = filtered_reentries[
                filtered_reentries["song"] == song_search
            ].copy()

        filtered_reentries = filtered_reentries[
            filtered_reentries["comeback_position"].between(
                selected_position[0],
                selected_position[1]
            )
        ].copy()

        filtered_reentries = filtered_reentries[
            filtered_reentries["popularity"].between(
                selected_popularity[0],
                selected_popularity[1]
            )
        ].copy()

    # --------------------------------------------------------
    # 5. DISPLAY COMEBACK RESULTS
    # --------------------------------------------------------

    if reentry_df.empty:

        st.warning("No comeback events were identified.")

    elif filtered_reentries.empty:

        st.warning(
            "No comeback events match the selected filters."
        )

    else:

        total_comebacks = len(filtered_reentries)

        comeback_songs = (
            filtered_reentries["song_id"].nunique()
        )

        comeback_artists = (
            filtered_reentries["artist"].nunique()
        )

        average_absence = (
            filtered_reentries["days_absent"].mean()
        )

        average_return_position = (
            filtered_reentries["comeback_position"].mean()
        )

        st.html(textwrap.dedent('<div class="section-heading">🎯 Comeback Performance KPIs</div>'))

        kpi_columns = st.columns(5)

        metrics = [
            ("Comeback Events", f"{total_comebacks:,}"),
            ("Returning Songs", f"{comeback_songs:,}"),
            ("Returning Artists", f"{comeback_artists:,}"),
            ("Average Days Absent", f"{average_absence:.2f}"),
            ("Average Return Position", f"{average_return_position:.2f}")
        ]

        for col, (label, value) in zip(kpi_columns, metrics):

            with col:
                st.metric(label, value)

        # ----------------------------------------------------
        # 6. TOP COMEBACK SONGS
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🏆 Most Frequent Comeback Songs</div>'))

        top_comeback_songs = (
            filtered_reentries
            .groupby(["song", "artist"], as_index=False)
            .agg(
                comeback_count=("song_id", "count"),
                average_days_absent=("days_absent", "mean"),
                average_return_position=("comeback_position", "mean")
            )
            .sort_values(
                "comeback_count",
                ascending=False
            )
            .head(15)
        )

        fig_comeback_songs = px.bar(
            top_comeback_songs.sort_values(
                "comeback_count",
                ascending=True
            ),
            x="comeback_count",
            y="song",
            color="comeback_count",
            orientation="h",
            hover_data=[
                "artist",
                "average_days_absent",
                "average_return_position"
            ],
            title="Top Songs by Comeback Frequency",
            color_continuous_scale="Purples"
        )

        fig_comeback_songs.update_layout(
            template=PLOTLY_TEMPLATE,
            height=520
        )

        st.plotly_chart(
            fig_comeback_songs,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

                # ----------------------------------------------------
        # 7. TOP COMEBACK ARTISTS
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🎤 Artist Comeback Performance</div>'))

        top_comeback_artists = (
            filtered_reentries
            .groupby("artist", as_index=False)
            .agg(
                comeback_count=("song_id", "count"),
                unique_returning_songs=("song_id", "nunique"),
                average_days_absent=("days_absent", "mean")
            )
            .sort_values(
                "comeback_count",
                ascending=False
            )
            .head(15)
        )

        fig_comeback_artists = px.bar(
            top_comeback_artists.sort_values(
                "comeback_count",
                ascending=True
            ),
            x="comeback_count",
            y="artist",
            color="comeback_count",
            orientation="h",
            hover_data=[
                "unique_returning_songs",
                "average_days_absent"
            ],
            title="Artists with the Most Comeback Events",
            color_continuous_scale="Blues"
        )

        fig_comeback_artists.update_layout(
            template=PLOTLY_TEMPLATE,
            height=500
        )

        st.plotly_chart(
            fig_comeback_artists,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 8. ABSENCE DURATION DISTRIBUTION
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">⏳ Comeback Absence Duration</div>'))

        fig_absence = px.histogram(
            filtered_reentries,
            x="days_absent",
            nbins=30,
            title="Distribution of Days Between Chart Appearances",
            color_discrete_sequence=[SECONDARY_COLOR]
        )

        fig_absence.update_layout(
            template=PLOTLY_TEMPLATE,
            height=390,
            xaxis_title="Calendar Days Absent",
            yaxis_title="Comeback Events"
        )

        st.plotly_chart(
            fig_absence,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 9. RETURN POSITION ANALYSIS
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📍 Comeback Return Position</div>'))

        fig_return_position = px.scatter(
            filtered_reentries,
            x="days_absent",
            y="comeback_position",
            color="position_change",
            size="popularity",
            hover_data=[
                "song",
                "artist",
                "comeback_date",
                "previous_position"
            ],
            title="Absence Duration vs Comeback Position",
            color_continuous_scale="RdBu"
        )

        fig_return_position.update_layout(
            template=PLOTLY_TEMPLATE,
            height=450
        )

        st.plotly_chart(
            fig_return_position,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 10. COMEBACK EVENT TABLE
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📋 Detailed Comeback Records</div>'))

        display_columns = [
            "song",
            "artist",
            "comeback_date",
            "days_absent",
            "observed_dates_absent",
            "previous_position",
            "comeback_position",
            "position_change",
            "popularity"
        ]

        st.dataframe(
            filtered_reentries[display_columns],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # 11. DOWNLOAD COMEBACK DATA
        # ----------------------------------------------------

        comeback_csv = filtered_reentries.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Comeback Analysis CSV",
            data=comeback_csv,
            file_name="south_korea_comeback_analysis.csv",
            mime="text/csv"
        )

        # ----------------------------------------------------
        # 12. INTERPRETATION
        # ----------------------------------------------------

        st.info(
            "A comeback is defined here as a song returning after "
            "at least one intervening observed chart date without "
            "an appearance. Calendar days absent and observed chart "
            "dates absent are reported separately."
        )


# ============================================================
# END OF PART 6
# ============================================================

# ============================================================
# PART 7: COMEBACK MOMENTUM ANALYTICS
# ============================================================

if page == "🚀 Momentum Analytics":

    st.html(textwrap.dedent("""
        <div class="section-heading">
            🚀 Comeback Momentum Intelligence
        </div>
        <p>
            Analyse ranking improvements, popularity changes
            and performance patterns when songs return.
        </p>
        """))

    # --------------------------------------------------------
    # 1. RECONSTRUCT COMEBACK HISTORY
    # --------------------------------------------------------

    momentum_source = chart_df.copy()

    momentum_source["date_key"] = (
        momentum_source["date"].dt.normalize()
    )

    momentum_dates = sorted(
        momentum_source["date_key"].dropna().unique()
    )

    momentum_date_index = {
        date: index
        for index, date in enumerate(momentum_dates)
    }

    momentum_events = []

    for song_id, history in momentum_source.groupby("song_id"):

        history = (
            history
            .sort_values(["date_key", "position"])
            .drop_duplicates(subset=["date_key"])
            .reset_index(drop=True)
        )

        if len(history) < 2:
            continue

        previous_record = None

        for _, current_record in history.iterrows():

            if previous_record is not None:

                previous_date = previous_record["date_key"]
                current_date = current_record["date_key"]

                previous_index = momentum_date_index[previous_date]
                current_index = momentum_date_index[current_date]

                observed_absence = (
                    current_index - previous_index - 1
                )

                if observed_absence > 0:

                    rank_improvement = (
                        previous_record["position"]
                        - current_record["position"]
                    )

                    previous_popularity = previous_record["popularity"]
                    current_popularity = current_record["popularity"]

                    popularity_change = (
                        current_popularity - previous_popularity
                        if pd.notna(previous_popularity)
                        and pd.notna(current_popularity)
                        else np.nan
                    )

                    momentum_events.append({

                        "song_id": song_id,
                        "song": current_record["song"],
                        "artist": current_record["artist"],
                        "comeback_date": current_date,
                        "previous_chart_date": previous_date,

                        "days_absent": (
                            current_date - previous_date
                        ).days,

                        "observed_dates_absent": observed_absence,

                        "previous_position":
                            previous_record["position"],

                        "comeback_position":
                            current_record["position"],

                        "rank_improvement": rank_improvement,

                        "previous_popularity":
                            previous_popularity,

                        "comeback_popularity":
                            current_popularity,

                        "popularity_change": popularity_change,

                        "album_type": current_record["album_type"],

                        "is_explicit": current_record["is_explicit"]

                    })

            previous_record = current_record

    momentum_df = pd.DataFrame(momentum_events)

    if not momentum_df.empty:

        momentum_df["comeback_date"] = pd.to_datetime(
            momentum_df["comeback_date"]
        )

    # --------------------------------------------------------
    # 2. APPLY GLOBAL FILTERS
    # --------------------------------------------------------

    filtered_momentum = momentum_df.copy()

    if not filtered_momentum.empty:

        if (
            isinstance(selected_dates, (tuple, list))
            and len(selected_dates) == 2
            and selected_dates[0] is not None
            and selected_dates[1] is not None
        ):

            start_date = pd.Timestamp(selected_dates[0])

            end_date = (
                pd.Timestamp(selected_dates[1])
                + pd.Timedelta(days=1)
            )

            filtered_momentum = filtered_momentum[
                filtered_momentum["comeback_date"].between(
                    start_date,
                    end_date,
                    inclusive="left"
                )
            ]

        if selected_artists:

            filtered_momentum = filtered_momentum[
                filtered_momentum["artist"].isin(selected_artists)
            ]

        if song_search != "All songs":

            filtered_momentum = filtered_momentum[
                filtered_momentum["song"] == song_search
            ]

        filtered_momentum = filtered_momentum[
            filtered_momentum["comeback_position"].between(
                selected_position[0],
                selected_position[1]
            )
        ]

        filtered_momentum = filtered_momentum[
            filtered_momentum["comeback_popularity"].between(
                selected_popularity[0],
                selected_popularity[1]
            )
        ]

    # --------------------------------------------------------
    # 3. DISPLAY RESULTS
    # --------------------------------------------------------

    if filtered_momentum.empty:

        st.warning(
            "No comeback events match the selected filters."
        )

    else:

        average_rank_improvement = (
            filtered_momentum["rank_improvement"].mean()
        )

        average_popularity_change = (
            filtered_momentum["popularity_change"].mean()
        )

        positive_rank_events = (
            filtered_momentum["rank_improvement"] > 0
        ).sum()

        negative_rank_events = (
            filtered_momentum["rank_improvement"] < 0
        ).sum()

        unchanged_rank_events = (
            filtered_momentum["rank_improvement"] == 0
        ).sum()

        st.html(textwrap.dedent('<div class="section-heading">🎯 Momentum Performance Indicators</div>'))

        metrics = [
            ("Comeback Events", f"{len(filtered_momentum):,}"),
            ("Average Rank Improvement", f"{average_rank_improvement:+.2f}"),
            ("Positive Rank Changes", f"{positive_rank_events:,}"),
            (
                "Average Popularity Change",
                f"{average_popularity_change:+.2f}"
                if pd.notna(average_popularity_change)
                else "N/A"
            ),
            ("Unique Comeback Songs", f"{filtered_momentum['song_id'].nunique():,}")
        ]

        cols = st.columns(5)

        for col, (label, value) in zip(cols, metrics):

            with col:
                st.metric(label, value)

        # ----------------------------------------------------
        # 4. RANK IMPROVEMENT DISTRIBUTION
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📊 Ranking Momentum Distribution</div>'))

        fig_rank = px.histogram(
            filtered_momentum,
            x="rank_improvement",
            nbins=30,
            title="Distribution of Comeback Rank Changes",
            color_discrete_sequence=[PRIMARY_COLOR]
        )

        fig_rank.add_vline(
            x=0,
            line_dash="dash",
            line_color="red"
        )

        fig_rank.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400
        )

        st.plotly_chart(
            fig_rank,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

                # ----------------------------------------------------
        # 5. RANK OUTCOMES
        # ----------------------------------------------------

        rank_outcomes = pd.DataFrame({
            "Outcome": ["Improved", "Declined", "Unchanged"],
            "Events": [
                positive_rank_events,
                negative_rank_events,
                unchanged_rank_events
            ]
        })

        fig_outcomes = px.pie(
            rank_outcomes,
            names="Outcome",
            values="Events",
            hole=0.55,
            title="Comeback Ranking Outcomes"
        )

        fig_outcomes.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400
        )

        st.plotly_chart(
            fig_outcomes,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 6. POPULARITY CHANGE ANALYSIS
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🔥 Popularity Change Analysis</div>'))

        popularity_momentum = filtered_momentum.dropna(
            subset=["popularity_change"]
        )

        if not popularity_momentum.empty:

            fig_popularity = px.scatter(
                popularity_momentum,
                x="days_absent",
                y="popularity_change",
                color="rank_improvement",
                size="comeback_popularity",
                hover_data=["song", "artist", "comeback_date"],
                title="Popularity Change vs Days Absent",
                color_continuous_scale="Viridis"
            )

            fig_popularity.add_hline(
                y=0,
                line_dash="dash"
            )

            fig_popularity.update_layout(
                template=PLOTLY_TEMPLATE,
                height=430
            )

            st.plotly_chart(
                fig_popularity,
                use_container_width=True,
                config=PLOTLY_CONFIG
            )

        # ----------------------------------------------------
        # 7. MOMENTUM BY RELEASE TYPE
        # ----------------------------------------------------

        release_momentum = (
            filtered_momentum
            .assign(
                release_type=(
                    filtered_momentum["album_type"]
                    .astype(str)
                    .str.strip()
                    .str.title()
                )
            )
            .groupby("release_type", as_index=False)
            .agg(
                comeback_events=("song_id", "count"),
                unique_songs=("song_id", "nunique"),
                average_rank_improvement=("rank_improvement", "mean"),
                average_popularity_change=("popularity_change", "mean")
            )
        )

        release_momentum = release_momentum[
            release_momentum["release_type"].isin(["Album", "Single"])
        ]

        st.html(textwrap.dedent('<div class="section-heading">💿 Momentum by Release Type</div>'))

        if not release_momentum.empty:

            fig_release = px.bar(
                release_momentum,
                x="release_type",
                y="average_rank_improvement",
                color="release_type",
                hover_data=[
                    "comeback_events",
                    "unique_songs",
                    "average_popularity_change"
                ],
                title="Average Rank Improvement by Release Type"
            )

            fig_release.update_layout(
                template=PLOTLY_TEMPLATE,
                height=400,
                showlegend=False
            )

            st.plotly_chart(
                fig_release,
                use_container_width=True,
                config=PLOTLY_CONFIG
            )

        # ----------------------------------------------------
        # 8. ARTIST MOMENTUM LEADERBOARD
        # ----------------------------------------------------

        artist_momentum = (
            filtered_momentum
            .groupby("artist", as_index=False)
            .agg(
                comeback_events=("song_id", "count"),
                unique_songs=("song_id", "nunique"),
                average_rank_improvement=("rank_improvement", "mean"),
                average_popularity_change=("popularity_change", "mean")
            )
            .sort_values(
                "average_rank_improvement",
                ascending=False
            )
            .head(15)
        )

        st.html(textwrap.dedent('<div class="section-heading">🎤 Artist Momentum Leaderboard</div>'))

        if not artist_momentum.empty:

            fig_artist = px.bar(
                artist_momentum.sort_values(
                    "average_rank_improvement"
                ),
                x="average_rank_improvement",
                y="artist",
                orientation="h",
                color="average_rank_improvement",
                hover_data=[
                    "comeback_events",
                    "unique_songs",
                    "average_popularity_change"
                ],
                title="Artists by Average Rank Improvement",
                color_continuous_scale="Blues"
            )

            fig_artist.update_layout(
                template=PLOTLY_TEMPLATE,
                height=520
            )

            st.plotly_chart(
                fig_artist,
                use_container_width=True,
                config=PLOTLY_CONFIG
            )

        # ----------------------------------------------------
        # 9. ABSENCE VS RANK IMPROVEMENT
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">⏳ Absence Duration vs Momentum</div>'))

        fig_absence = px.scatter(
            filtered_momentum,
            x="days_absent",
            y="rank_improvement",
            color="album_type",
            hover_data=["song", "artist", "comeback_date"],
            title="Absence Duration vs Rank Improvement"
        )

        fig_absence.add_hline(
            y=0,
            line_dash="dash"
        )

        fig_absence.update_layout(
            template=PLOTLY_TEMPLATE,
            height=420
        )

        st.plotly_chart(
            fig_absence,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 10. DETAILED TABLE
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📋 Detailed Momentum Records</div>'))

        momentum_columns = [
            "comeback_date",
            "song",
            "artist",
            "days_absent",
            "previous_position",
            "comeback_position",
            "rank_improvement",
            "previous_popularity",
            "comeback_popularity",
            "popularity_change",
            "album_type"
        ]

        st.dataframe(
            filtered_momentum[momentum_columns].sort_values(
                "comeback_date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Momentum Analysis CSV",
            data=filtered_momentum.to_csv(index=False).encode("utf-8"),
            file_name="South_Korea_Top50_Momentum_Analysis.csv",
            mime="text/csv"
        )

        st.info(
            "A positive rank improvement means the comeback position "
            "is numerically better than the previous position. "
            "This is a chart-ranking measure, not a causal measure "
            "of audience response."
        )


# ============================================================
# END OF PART 7
# ============================================================

# ============================================================
# PART 8: RETENTION ANALYTICS
# ============================================================

if page == "⏳ Retention Analytics":

    st.html(textwrap.dedent("""
        <div class="section-heading">
            ⏳ Post-Comeback Retention Analytics
        </div>
        <p>
            Examine how long returning songs remain on the
            South Korea Top 50 after their comeback.
        </p>
        """))

    # --------------------------------------------------------
    # 1. PREPARE COMPLETE CHART HISTORY
    # --------------------------------------------------------

    retention_source = chart_df.copy()

    retention_source["date_key"] = (
        retention_source["date"].dt.normalize()
    )

    retention_dates = sorted(
        retention_source["date_key"].dropna().unique()
    )

    retention_date_index = {
        date: index
        for index, date in enumerate(retention_dates)
    }

    # --------------------------------------------------------
    # 2. IDENTIFY COMEBACKS AND POST-COMEBACK RETENTION
    # --------------------------------------------------------

    retention_records = []

    for song_id, history in retention_source.groupby("song_id"):

        history = (
            history
            .sort_values(["date_key", "position"])
            .drop_duplicates(subset=["date_key"])
            .reset_index(drop=True)
        )

        if len(history) < 2:
            continue

        for index in range(1, len(history)):

            previous_record = history.iloc[index - 1]
            comeback_record = history.iloc[index]

            previous_date = previous_record["date_key"]
            comeback_date = comeback_record["date_key"]

            previous_index = retention_date_index[previous_date]
            comeback_index = retention_date_index[comeback_date]

            observed_absence = (
                comeback_index - previous_index - 1
            )

            if observed_absence <= 0:
                continue

            # Find the next continuous run of chart appearances.

            run_end_index = comeback_index

            while run_end_index + 1 < len(retention_dates):

                next_observed_date = retention_dates[
                    run_end_index + 1
                ]

                next_song_dates = history["date_key"].values

                if next_observed_date in next_song_dates:

                    run_end_index += 1

                else:

                    break

            # Count consecutive observed chart dates.

            observed_retention = (
                run_end_index - comeback_index + 1
            )

            last_chart_date = retention_dates[
                run_end_index
            ]

            calendar_retention = (
                last_chart_date - comeback_date
            ).days + 1

            is_right_censored = (
                run_end_index == len(retention_dates) - 1
            )

            retention_records.append({

                "song_id": song_id,

                "song": comeback_record["song"],

                "artist": comeback_record["artist"],

                "comeback_date": comeback_date,

                "previous_chart_date": previous_date,

                "days_absent": (
                    comeback_date - previous_date
                ).days,

                "observed_dates_absent": observed_absence,

                "comeback_position":
                    comeback_record["position"],

                "comeback_popularity":
                    comeback_record["popularity"],

                "observed_retention_dates":
                    observed_retention,

                "calendar_retention_days":
                    calendar_retention,

                "last_chart_date": last_chart_date,

                "right_censored": is_right_censored,

                "album_type":
                    comeback_record["album_type"],

                "is_explicit":
                    comeback_record["is_explicit"]

            })

    retention_df = pd.DataFrame(retention_records)

    if not retention_df.empty:

        retention_df["comeback_date"] = pd.to_datetime(
            retention_df["comeback_date"]
        )

        retention_df["last_chart_date"] = pd.to_datetime(
            retention_df["last_chart_date"]
        )

    # --------------------------------------------------------
    # 3. APPLY GLOBAL FILTERS
    # --------------------------------------------------------

    filtered_retention = retention_df.copy()

    if not filtered_retention.empty:

        if (
            isinstance(selected_dates, (tuple, list))
            and len(selected_dates) == 2
            and selected_dates[0] is not None
            and selected_dates[1] is not None
        ):

            start_date = pd.Timestamp(selected_dates[0])

            end_date = (
                pd.Timestamp(selected_dates[1])
                + pd.Timedelta(days=1)
            )

            filtered_retention = filtered_retention[
                filtered_retention["comeback_date"].between(
                    start_date,
                    end_date,
                    inclusive="left"
                )
            ]

        if selected_artists:

            filtered_retention = filtered_retention[
                filtered_retention["artist"].isin(selected_artists)
            ]

        if song_search != "All songs":

            filtered_retention = filtered_retention[
                filtered_retention["song"] == song_search
            ]

        filtered_retention = filtered_retention[
            filtered_retention["comeback_position"].between(
                selected_position[0],
                selected_position[1]
            )
        ]

        filtered_retention = filtered_retention[
            filtered_retention["comeback_popularity"].between(
                selected_popularity[0],
                selected_popularity[1]
            )
        ]

    # --------------------------------------------------------
    # 4. CHECK DATA
    # --------------------------------------------------------

    if filtered_retention.empty:

        st.warning(
            "No retention records match the selected filters."
        )

    else:

        average_retention = (
            filtered_retention["calendar_retention_days"].mean()
        )

        median_retention = (
            filtered_retention["calendar_retention_days"].median()
        )

        longest_retention = (
            filtered_retention["calendar_retention_days"].max()
        )

        unique_songs = (
            filtered_retention["song_id"].nunique()
        )

        censored_count = (
            filtered_retention["right_censored"].sum()
        )

        # ----------------------------------------------------
        # 5. RETENTION KPIs
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🎯 Retention Performance Indicators</div>'))

        metrics = [
            ("Comeback Events", f"{len(filtered_retention):,}"),
            ("Unique Songs", f"{unique_songs:,}"),
            ("Average Retention", f"{average_retention:.2f} days"),
            ("Median Retention", f"{median_retention:.2f} days"),
            ("Right-Censored Events", f"{censored_count:,}")
        ]

        cols = st.columns(5)

        for col, (label, value) in zip(cols, metrics):

            with col:
                st.metric(label, value)

        # ----------------------------------------------------
        # 6. RETENTION DISTRIBUTION
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📊 Retention Duration Distribution</div>'))

        fig_retention = px.histogram(
            filtered_retention,
            x="calendar_retention_days",
            nbins=30,
            title="Post-Comeback Retention Duration",
            color_discrete_sequence=[SECONDARY_COLOR]
        )

        fig_retention.update_layout(
            template=PLOTLY_TEMPLATE,
            height=420,
            xaxis_title="Calendar Retention Days",
            yaxis_title="Comeback Events"
        )

        st.plotly_chart(
            fig_retention,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 7. OBSERVED RETENTION
        # ----------------------------------------------------

        fig_observed = px.histogram(
            filtered_retention,
            x="observed_retention_dates",
            nbins=25,
            title="Retention by Observed Chart Dates",
            color_discrete_sequence=[PRIMARY_COLOR]
        )

        fig_observed.update_layout(
            template=PLOTLY_TEMPLATE,
            height=390,
            xaxis_title="Consecutive Observed Chart Dates",
            yaxis_title="Comeback Events"
        )

        st.plotly_chart(
            fig_observed,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

                # ----------------------------------------------------
        # 8. TOP RETAINED SONGS
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🏆 Songs with the Longest Retention</div>'))

        top_retained = (
            filtered_retention
            .groupby(["song", "artist"], as_index=False)
            .agg(
                average_retention_days=(
                    "calendar_retention_days",
                    "mean"
                ),
                comeback_events=("song_id", "count"),
                average_observed_dates=(
                    "observed_retention_dates",
                    "mean"
                )
            )
            .sort_values(
                "average_retention_days",
                ascending=False
            )
            .head(15)
        )

        fig_top_retained = px.bar(
            top_retained.sort_values(
                "average_retention_days"
            ),
            x="average_retention_days",
            y="song",
            color="average_retention_days",
            orientation="h",
            hover_data=[
                "artist",
                "comeback_events",
                "average_observed_dates"
            ],
            title="Top Songs by Average Post-Comeback Retention",
            color_continuous_scale="Purples"
        )

        fig_top_retained.update_layout(
            template=PLOTLY_TEMPLATE,
            height=520
        )

        st.plotly_chart(
            fig_top_retained,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 9. RETENTION BY ARTIST
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🎤 Artist Retention Performance</div>'))

        artist_retention = (
            filtered_retention
            .groupby("artist", as_index=False)
            .agg(
                average_retention_days=(
                    "calendar_retention_days",
                    "mean"
                ),
                comeback_events=("song_id", "count"),
                unique_songs=("song_id", "nunique")
            )
            .sort_values(
                "average_retention_days",
                ascending=False
            )
            .head(15)
        )

        fig_artist_retention = px.bar(
            artist_retention.sort_values(
                "average_retention_days"
            ),
            x="average_retention_days",
            y="artist",
            orientation="h",
            color="average_retention_days",
            hover_data=[
                "comeback_events",
                "unique_songs"
            ],
            title="Artists by Average Post-Comeback Retention",
            color_continuous_scale="Blues"
        )

        fig_artist_retention.update_layout(
            template=PLOTLY_TEMPLATE,
            height=520
        )

        st.plotly_chart(
            fig_artist_retention,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 10. RETENTION VS COMEBACK POSITION
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📍 Retention vs Comeback Position</div>'))

        fig_retention_position = px.scatter(
            filtered_retention,
            x="comeback_position",
            y="calendar_retention_days",
            color="album_type",
            size="comeback_popularity",
            hover_data=[
                "song",
                "artist",
                "comeback_date"
            ],
            title="Comeback Position vs Retention Duration"
        )

        fig_retention_position.update_layout(
            template=PLOTLY_TEMPLATE,
            height=430
        )

        st.plotly_chart(
            fig_retention_position,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 11. RIGHT-CENSORING INFORMATION
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">⚠️ Retention Measurement Limitations</div>'))

        st.warning(
            f"{int(censored_count)} events in the current selection "
            "reach the final observed chart date while still present. "
            "Their eventual retention duration is unknown. These "
            "observations are right-censored and should not be "
            "interpreted as completed retention periods."
        )

        # ----------------------------------------------------
        # 12. RETENTION TABLE
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📋 Detailed Retention Records</div>'))

        retention_columns = [
            "comeback_date",
            "song",
            "artist",
            "days_absent",
            "observed_dates_absent",
            "observed_retention_dates",
            "calendar_retention_days",
            "right_censored",
            "comeback_position",
            "comeback_popularity"
        ]

        st.dataframe(
            filtered_retention[retention_columns].sort_values(
                "comeback_date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # 13. DOWNLOAD RETENTION DATA
        # ----------------------------------------------------

        st.download_button(
            "⬇️ Download Retention Analysis CSV",
            data=filtered_retention.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="South_Korea_Top50_Retention_Analysis.csv",
            mime="text/csv"
        )


# ============================================================
# END OF PART 8
# ============================================================

# ============================================================
# PART 9: FANDOM INTELLIGENCE
# ============================================================

if page == "💜 Fandom Intelligence":

    st.html(textwrap.dedent("""
        <div class="section-heading">
            💜 Fandom Intensity & Artist Intelligence
        </div>
        <p>
            Explore recurring chart appearances, comeback activity,
            and retention patterns across songs and artists.
        </p>
        """))

    # --------------------------------------------------------
    # 1. SELECT THE ANALYSIS PERIOD
    # --------------------------------------------------------

    fandom_source = chart_df.copy()

    if (
        isinstance(selected_dates, (tuple, list))
        and len(selected_dates) == 2
        and selected_dates[0] is not None
        and selected_dates[1] is not None
    ):

        start_date = pd.Timestamp(selected_dates[0])

        end_date = (
            pd.Timestamp(selected_dates[1])
            + pd.Timedelta(days=1)
        )

        fandom_source = fandom_source[
            fandom_source["date"].between(
                start_date,
                end_date,
                inclusive="left"
            )
        ].copy()

    if selected_artists:

        fandom_source = fandom_source[
            fandom_source["artist"].isin(selected_artists)
        ].copy()

    if song_search != "All songs":

        fandom_source = fandom_source[
            fandom_source["song"] == song_search
        ].copy()

    fandom_source = fandom_source[
        fandom_source["position"].between(
            selected_position[0],
            selected_position[1]
        )
    ].copy()

    fandom_source = fandom_source[
        fandom_source["popularity"].between(
            selected_popularity[0],
            selected_popularity[1]
        )
    ].copy()

    # --------------------------------------------------------
    # 2. CHECK DATA AVAILABILITY
    # --------------------------------------------------------

    if fandom_source.empty:

        st.warning(
            "No chart records match the selected filters."
        )

    else:

        # ----------------------------------------------------
        # 3. SONG-LEVEL FANDOM PROXY METRICS
        # ----------------------------------------------------

        song_fandom = (
            fandom_source
            .groupby(["song_id", "song", "artist"], as_index=False)
            .agg(
                chart_appearances=("date", "count"),
                active_chart_dates=("date", "nunique"),
                average_position=("position", "mean"),
                best_position=("position", "min"),
                average_popularity=("popularity", "mean"),
                first_chart_date=("date", "min"),
                last_chart_date=("date", "max")
            )
        )

        # ----------------------------------------------------
        # 4. ARTIST-LEVEL FANDOM PROXY METRICS
        # ----------------------------------------------------

        artist_fandom = (
            fandom_source
            .groupby("artist", as_index=False)
            .agg(
                chart_appearances=("song_id", "count"),
                unique_songs=("song_id", "nunique"),
                average_position=("position", "mean"),
                average_popularity=("popularity", "mean"),
                active_chart_dates=("date", "nunique")
            )
        )

        artist_fandom = artist_fandom.sort_values(
            "chart_appearances",
            ascending=False
        )

        # ----------------------------------------------------
        # 5. COMEBACK EVENTS WITHIN SELECTED PERIOD
        # ----------------------------------------------------

        fandom_dates = sorted(
            chart_df["date"].dt.normalize().dropna().unique()
        )

        fandom_date_index = {
            date: index
            for index, date in enumerate(fandom_dates)
        }

        fandom_comebacks = []

        for song_id, history in chart_df.groupby("song_id"):

            history = (
                history
                .sort_values(["date", "position"])
                .drop_duplicates(
                    subset=["date"]
                )
                .reset_index(drop=True)
            )

            if len(history) < 2:
                continue

            for index in range(1, len(history)):

                previous_record = history.iloc[index - 1]

                current_record = history.iloc[index]

                previous_date = previous_record["date"].normalize()

                current_date = current_record["date"].normalize()

                previous_index = fandom_date_index.get(
                    previous_date
                )

                current_index = fandom_date_index.get(
                    current_date
                )

                if previous_index is None or current_index is None:
                    continue

                absent_dates = (
                    current_index - previous_index - 1
                )

                if absent_dates <= 0:
                    continue

                fandom_comebacks.append({

                    "song_id": song_id,

                    "song": current_record["song"],

                    "artist": current_record["artist"],

                    "comeback_date": current_date,

                    "days_absent": (
                        current_date - previous_date
                    ).days,

                    "comeback_position":
                        current_record["position"]

                })

        fandom_comeback_df = pd.DataFrame(
            fandom_comebacks
        )

        if not fandom_comeback_df.empty:

            fandom_comeback_df["comeback_date"] = pd.to_datetime(
                fandom_comeback_df["comeback_date"]
            )

            if (
                isinstance(selected_dates, (tuple, list))
                and len(selected_dates) == 2
                and selected_dates[0] is not None
                and selected_dates[1] is not None
            ):

                start_date = pd.Timestamp(selected_dates[0])

                end_date = (
                    pd.Timestamp(selected_dates[1])
                    + pd.Timedelta(days=1)
                )

                fandom_comeback_df = fandom_comeback_df[
                    fandom_comeback_df["comeback_date"].between(
                        start_date,
                        end_date,
                        inclusive="left"
                    )
                ]

            if selected_artists:

                fandom_comeback_df = fandom_comeback_df[
                    fandom_comeback_df["artist"].isin(
                        selected_artists
                    )
                ]

            if song_search != "All songs":

                fandom_comeback_df = fandom_comeback_df[
                    fandom_comeback_df["song"] == song_search
                ]

        # ----------------------------------------------------
        # 6. ADD COMEBACK FREQUENCY
        # ----------------------------------------------------

        if not fandom_comeback_df.empty:

            song_comeback_counts = (
                fandom_comeback_df
                .groupby("song_id")
                .size()
                .rename("comeback_events")
            )

            artist_comeback_counts = (
                fandom_comeback_df
                .groupby("artist")
                .size()
                .rename("comeback_events")
            )

            song_fandom = song_fandom.merge(
                song_comeback_counts,
                on="song_id",
                how="left"
            )

            artist_fandom = artist_fandom.merge(
                artist_comeback_counts,
                on="artist",
                how="left"
            )

        else:

            song_fandom["comeback_events"] = 0

            artist_fandom["comeback_events"] = 0

        song_fandom["comeback_events"] = (
            song_fandom["comeback_events"]
            .fillna(0)
            .astype(int)
        )

        artist_fandom["comeback_events"] = (
            artist_fandom["comeback_events"]
            .fillna(0)
            .astype(int)
        )

        # ----------------------------------------------------
        # 7. KPI SECTION
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🎯 Fandom Intelligence Indicators</div>'))

        total_songs = song_fandom["song_id"].nunique()

        total_artists = artist_fandom["artist"].nunique()

        total_comebacks = int(
            artist_fandom["comeback_events"].sum()
        )

        average_appearances = (
            song_fandom["chart_appearances"].mean()
        )

        average_popularity = (
            song_fandom["average_popularity"].mean()
        )

        metrics = [
            ("Songs Analysed", f"{total_songs:,}"),
            ("Artists Analysed", f"{total_artists:,}"),
            ("Comeback Events", f"{total_comebacks:,}"),
            ("Average Song Appearances", f"{average_appearances:.2f}"),
            (
                "Average Popularity",
                f"{average_popularity:.2f}"
                if pd.notna(average_popularity)
                else "N/A"
            )
        ]

        cols = st.columns(5)

        for col, (label, value) in zip(cols, metrics):

            with col:
                st.metric(label, value)
                        # ----------------------------------------------------
        # 8. SONG COMEBACK LEADERBOARD
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🏆 Song Comeback Leaderboard</div>'))

        top_fandom_songs = (
            song_fandom
            .sort_values(
                ["comeback_events", "chart_appearances"],
                ascending=False
            )
            .head(15)
        )

        fig_fandom_songs = px.bar(
            top_fandom_songs.sort_values(
                "comeback_events",
                ascending=True
            ),
            x="comeback_events",
            y="song",
            orientation="h",
            color="comeback_events",
            hover_data=[
                "artist",
                "chart_appearances",
                "average_popularity"
            ],
            title="Songs with the Most Comeback Events",
            color_continuous_scale="Purples"
        )

        fig_fandom_songs.update_layout(
            template=PLOTLY_TEMPLATE,
            height=520
        )

        st.plotly_chart(
            fig_fandom_songs,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 9. ARTIST FANDOM LEADERBOARD
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🎤 Artist Fandom Leaderboard</div>'))

        top_fandom_artists = (
            artist_fandom
            .sort_values(
                ["comeback_events", "chart_appearances"],
                ascending=False
            )
            .head(15)
        )

        fig_fandom_artists = px.bar(
            top_fandom_artists.sort_values(
                "comeback_events",
                ascending=True
            ),
            x="comeback_events",
            y="artist",
            orientation="h",
            color="comeback_events",
            hover_data=[
                "unique_songs",
                "chart_appearances",
                "average_popularity"
            ],
            title="Artists by Comeback Frequency",
            color_continuous_scale="Blues"
        )

        fig_fandom_artists.update_layout(
            template=PLOTLY_TEMPLATE,
            height=520
        )

        st.plotly_chart(
            fig_fandom_artists,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 10. CHART APPEARANCE VS POPULARITY
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📈 Chart Activity vs Popularity</div>'))

        fig_activity_popularity = px.scatter(
            song_fandom,
            x="chart_appearances",
            y="average_popularity",
            color="comeback_events",
            size="active_chart_dates",
            hover_data=["song", "artist"],
            title="Song Chart Appearances vs Average Popularity",
            color_continuous_scale="Viridis"
        )

        fig_activity_popularity.update_layout(
            template=PLOTLY_TEMPLATE,
            height=450
        )

        st.plotly_chart(
            fig_activity_popularity,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 11. ARTIST SONG PORTFOLIO
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🎶 Artist Song Portfolio</div>'))

        fig_artist_portfolio = px.scatter(
            artist_fandom,
            x="unique_songs",
            y="chart_appearances",
            color="comeback_events",
            size="average_popularity",
            hover_data=["artist", "active_chart_dates"],
            title="Artist Song Count vs Chart Appearances",
            color_continuous_scale="Plasma"
        )

        fig_artist_portfolio.update_layout(
            template=PLOTLY_TEMPLATE,
            height=450
        )

        st.plotly_chart(
            fig_artist_portfolio,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

        # ----------------------------------------------------
        # 12. SONG-LEVEL TABLE
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📋 Song-Level Fandom Metrics</div>'))

        song_display_columns = [
            "song",
            "artist",
            "chart_appearances",
            "active_chart_dates",
            "comeback_events",
            "average_position",
            "best_position",
            "average_popularity",
            "first_chart_date",
            "last_chart_date"
        ]

        st.dataframe(
            song_fandom.sort_values(
                "comeback_events",
                ascending=False
            )[song_display_columns],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # 13. ARTIST-LEVEL TABLE
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">📊 Artist-Level Fandom Metrics</div>'))

        artist_display_columns = [
            "artist",
            "chart_appearances",
            "unique_songs",
            "comeback_events",
            "active_chart_dates",
            "average_position",
            "average_popularity"
        ]

        st.dataframe(
            artist_fandom.sort_values(
                "comeback_events",
                ascending=False
            )[artist_display_columns],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # 14. DOWNLOAD DATA
        # ----------------------------------------------------

        download_col1, download_col2 = st.columns(2)

        with download_col1:

            st.download_button(
                "⬇️ Download Song Fandom Metrics",
                data=song_fandom.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="South_Korea_Song_Fandom_Metrics.csv",
                mime="text/csv",
                use_container_width=True
            )

        with download_col2:

            st.download_button(
                "⬇️ Download Artist Fandom Metrics",
                data=artist_fandom.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="South_Korea_Artist_Fandom_Metrics.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ----------------------------------------------------
        # 15. METHODOLOGY
        # ----------------------------------------------------

        st.html(textwrap.dedent('<div class="section-heading">🧠 Fandom Intelligence Methodology</div>'))

        st.html(textwrap.dedent("""
            <div class="content-card">

                <h3>Chart-Based Fandom Proxies</h3>

                <p>
                    This analysis uses recurring chart appearances,
                    comeback frequency, unique songs and chart
                    retention as observable indicators of repeated
                    chart activity.
                </p>

                <h3>Interpretation</h3>

                <p>
                    Higher comeback frequency indicates more
                    observed returns to the chart during the
                    selected period. Higher chart appearances
                    indicate more recorded presence in the dataset.
                </p>

                <h3>Important Limitation</h3>

                <p>
                    Fandom intensity is a chart-based proxy.
                    The dataset does not directly contain fan
                    membership, individual listener behaviour,
                    social engagement, or streaming-user identity.
                    Therefore, these indicators must not be
                    interpreted as direct measurements of
                    audience loyalty or fandom size.
                </p>

            </div>
            """))


# ============================================================
# END OF PART 9
# ============================================================

# ============================================================
# PART 10: MACHINE LEARNING DASHBOARD
# ============================================================

if page == "🤖 Machine Learning":

    st.html(textwrap.dedent("""
        <div class="section-heading">
            🤖 Machine Learning & Predictive Analytics
        </div>

        <p class="main-subtitle">
            Explore chronological machine learning experiments
            for predicting song comebacks and chart positions.
        </p>
        """))

    st.info(
        "The performance results displayed in this section "
        "are taken from the audited project notebook. "
        "They are not retrained when sidebar filters change."
    )

    # --------------------------------------------------------
    # 1. EXPERIMENT OVERVIEW
    # --------------------------------------------------------

    st.html(textwrap.dedent('<div class="section-heading">🧪 Experiment Overview</div>'))

    exp1, exp2 = st.columns(2)

    with exp1:
        st.html(textwrap.dedent("""
            <div class="content-card">
                <h3>🎯 Classification Problem</h3>
                <p>
                    Predict whether a song will return to
                    the Top 50 within the next seven observed
                    chart dates.
                </p>
                <p><b>Target:</b> comeback_within_7_dates</p>
                <p><b>Models:</b></p>
                <ul>
                    <li>Logistic Regression</li>
                    <li>Random Forest Classifier</li>
                </ul>
            </div>
            """))

    with exp2:
        st.html(textwrap.dedent("""
            <div class="content-card">
                <h3>📈 Regression Problem</h3>
                <p>
                    Predict the song's position on the
                    next observed chart date.
                </p>
                <p><b>Target:</b> Next observed chart position</p>
                <p><b>Models:</b></p>
                <ul>
                    <li>Linear Regression</li>
                    <li>Random Forest Regressor</li>
                </ul>
            </div>
            """))

    # --------------------------------------------------------
    # 2. CLASSIFICATION RESULTS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🎯 Classification Model Performance</div>'))

    classification_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy": [
            0.902133,
            0.972115
        ],
        "Precision": [
            0.101012,
            0.262105
        ],
        "Recall": [
            0.860697,
            0.673300
        ],
        "F1 Score": [
            0.180805,
            0.377323
        ],
        "ROC AUC": [
            0.941202,
            0.944252
        ],
        "PR AUC": [
            0.212719,
            0.352950
        ]
    })

    st.dataframe(
        classification_results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC AUC": "{:.4f}",
            "PR AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # 3. CLASSIFICATION KPI CARDS
    # --------------------------------------------------------

    st.html(textwrap.dedent('<div class="section-heading">📊 Classification Highlights</div>'))

    cls1, cls2, cls3, cls4 = st.columns(4)

    with cls1:
        st.metric("Best Accuracy", "97.21%", "Random Forest")

    with cls2:
        st.metric("Best F1 Score", "0.3773", "Random Forest")

    with cls3:
        st.metric("Best ROC-AUC", "0.9443", "Random Forest")

    with cls4:
        st.metric("Best PR-AUC", "0.3530", "Random Forest")

    # --------------------------------------------------------
    # 4. CLASSIFICATION MODEL COMPARISON
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">📈 Classification Model Comparison</div>'))

    classification_long = classification_results.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    fig_classification = px.bar(
        classification_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".3f",
        title="Classification Model Evaluation"
    )

    fig_classification.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        yaxis_title="Metric Score",
        xaxis_title="Evaluation Metric"
    )

    st.plotly_chart(
        fig_classification,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )

        # --------------------------------------------------------
    # 5. CLASSIFICATION CONFUSION MATRICES
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🔍 Classification Confusion Matrices</div>'))

    confusion_col1, confusion_col2 = st.columns(2)

    logistic_matrix = np.array([
        [42833, 4619],
        [84, 519]
    ])

    random_forest_matrix = np.array([
        [46309, 1143],
        [197, 406]
    ])

    with confusion_col1:

        fig_logistic_cm = go.Figure(
            data=go.Heatmap(
                z=logistic_matrix,
                x=[
                    "Predicted Negative",
                    "Predicted Positive"
                ],
                y=[
                    "Actual Negative",
                    "Actual Positive"
                ],
                text=logistic_matrix,
                texttemplate="%{text:,}",
                colorscale="Blues",
                showscale=False
            )
        )

        fig_logistic_cm.update_layout(
            title="Logistic Regression",
            template=PLOTLY_TEMPLATE,
            height=370
        )

        st.plotly_chart(
            fig_logistic_cm,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

    with confusion_col2:

        fig_rf_cm = go.Figure(
            data=go.Heatmap(
                z=random_forest_matrix,
                x=[
                    "Predicted Negative",
                    "Predicted Positive"
                ],
                y=[
                    "Actual Negative",
                    "Actual Positive"
                ],
                text=random_forest_matrix,
                texttemplate="%{text:,}",
                colorscale="Purples",
                showscale=False
            )
        )

        fig_rf_cm.update_layout(
            title="Random Forest Classifier",
            template=PLOTLY_TEMPLATE,
            height=370
        )

        st.plotly_chart(
            fig_rf_cm,
            use_container_width=True,
            config=PLOTLY_CONFIG
        )

    st.caption(
        "Matrix layout: rows represent actual classes; "
        "columns represent predicted classes."
    )

    # --------------------------------------------------------
    # 6. CLASSIFICATION DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">📂 Classification Dataset</div>'))

    dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

    with dataset_col1:
        st.metric("Total Observations", "137,668")

    with dataset_col2:
        st.metric("Unique Prediction Dates", "546")

    with dataset_col3:
        st.metric("Prediction Horizon", "7 observed dates")

    st.markdown(
        """
        **Chronological train-test split**

        | Dataset | Records | Date range |
        |---|---:|---|
        | Training | 89,613 | 2024-05-19 to 2025-08-01 |
        | Testing | 48,055 | 2025-08-02 to 2025-11-20 |
        """
    )

    st.success(
        "Chronological date separation: PASS. "
        "The training period ends before the testing period begins."
    )

    # --------------------------------------------------------
    # 7. CLASS IMBALANCE
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">⚖️ Classification Target Distribution</div>'))

    class_distribution = pd.DataFrame({
        "Class": [
            "No Comeback",
            "Comeback"
        ],
        "Records": [
            47452,
            603
        ]
    })

    fig_class_distribution = px.pie(
        class_distribution,
        names="Class",
        values="Records",
        hole=0.55,
        title="Test Dataset Class Distribution",
        color_discrete_sequence=[
            PRIMARY_COLOR,
            SECONDARY_COLOR
        ]
    )

    fig_class_distribution.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400
    )

    st.plotly_chart(
        fig_class_distribution,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )

    st.warning(
        "The positive class is highly imbalanced. "
        "Therefore, accuracy alone is not sufficient "
        "for evaluating comeback prediction performance."
    )

    # --------------------------------------------------------
    # 8. CLASSIFICATION FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🔎 Classification Feature Importance</div>'))

    classification_importance = pd.DataFrame({
        "Feature": [
            "days_absent",
            "is_explicit",
            "mean_previous_popularity",
            "popularity_std",
            "duration_ms",
            "previous_position",
            "previous_popularity",
            "mean_previous_position",
            "position_std",
            "album_type",
            "total_tracks",
            "previous_appearances"
        ],
        "Permutation Importance": [
            0.345497,
            0.006979,
            0.004700,
            0.001010,
            -0.000801,
            -0.000920,
            -0.003097,
            -0.005106,
            -0.009599,
            -0.012112,
            -0.017624,
            -0.027359
        ]
    })

    fig_class_importance = px.bar(
        classification_importance.sort_values(
            "Permutation Importance",
            ascending=True
        ),
        x="Permutation Importance",
        y="Feature",
        orientation="h",
        color="Permutation Importance",
        color_continuous_scale="RdBu",
        title="Classification Permutation Importance"
    )

    fig_class_importance.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray"
    )

    fig_class_importance.update_layout(
        template=PLOTLY_TEMPLATE,
        height=550,
        xaxis_title="Change in F1 Score After Permutation",
        yaxis_title="Feature"
    )

    st.plotly_chart(
        fig_class_importance,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )

    st.info(
        "Days absent is the dominant classification feature. "
        "Negative permutation importance values indicate "
        "that shuffling the feature slightly improved the "
        "evaluation score in this experiment."
    )

    # --------------------------------------------------------
    # 9. REGRESSION MODEL PERFORMANCE
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">📈 Regression Model Performance</div>'))

    regression_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest Regressor"
        ],
        "MAE": [
            2.425368,
            2.424530
        ],
        "RMSE": [
            3.796779,
            3.744243
        ],
        "R2 Score": [
            0.927300,
            0.929298
        ]
    })

    st.dataframe(
        regression_results.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # 10. REGRESSION KPI CARDS
    # --------------------------------------------------------

    reg1, reg2, reg3 = st.columns(3)

    with reg1:
        st.metric("Best MAE", "2.4245", "Random Forest")

    with reg2:
        st.metric("Best RMSE", "3.7442", "Random Forest")

    with reg3:
        st.metric("Best R² Score", "0.9293", "Random Forest")

    # --------------------------------------------------------
    # 11. REGRESSION MODEL COMPARISON
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">📊 Regression Model Comparison</div>'))

    regression_long = regression_results.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    fig_regression = px.bar(
        regression_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".3f",
        title="Regression Model Evaluation"
    )

    fig_regression.update_layout(
        template=PLOTLY_TEMPLATE,
        height=430,
        xaxis_title="Evaluation Metric",
        yaxis_title="Metric Value"
    )

    st.plotly_chart(
        fig_regression,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )

    # --------------------------------------------------------
    # 12. REGRESSION DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">📂 Regression Dataset</div>'))

    regression_col1, regression_col2 = st.columns(2)

    with regression_col1:
        st.metric("Regression Observations", "26,289")

    with regression_col2:
        st.metric("Predictive Target", "Next Chart Position")

    st.markdown(
        """
        **Regression features**

        - Current chart position
        - Popularity
        - Duration
        - Total album tracks
        - Album type
        - Explicit-content indicator

        **Target:** Song position on the next observed chart date.
        """
    )

    # --------------------------------------------------------
    # 13. REGRESSION FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🔍 Regression Feature Importance</div>'))

    regression_importance = pd.DataFrame({
        "Feature": [
            "position",
            "popularity",
            "duration_ms",
            "total_tracks",
            "is_explicit",
            "album_type"
        ],
        "Permutation Importance": [
            13.598881,
            0.039187,
            0.029833,
            0.013448,
            0.001949,
            -0.000198
        ]
    })

    fig_reg_importance = px.bar(
        regression_importance.sort_values(
            "Permutation Importance",
            ascending=True
        ),
        x="Permutation Importance",
        y="Feature",
        orientation="h",
        color="Permutation Importance",
        color_continuous_scale="Blues",
        title="Regression Permutation Importance"
    )

    fig_reg_importance.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray"
    )

    fig_reg_importance.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        xaxis_title="Permutation Importance",
        yaxis_title="Feature"
    )

    st.plotly_chart(
        fig_reg_importance,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )

    st.info(
        "Current chart position is the dominant regression "
        "feature. This indicates strong rank persistence "
        "between consecutive chart observations."
    )

    # --------------------------------------------------------
    # 14. MODEL INTERPRETATION
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">💡 Model Interpretation</div>'))

    st.html(textwrap.dedent("""
        <div class="content-card">

            <h3>Classification Findings</h3>

            <p>
                Random Forest achieved the highest F1-score
                among the evaluated classification models.
                Logistic Regression achieved higher recall
                but considerably lower precision.
            </p>

            <h3>Regression Findings</h3>

            <p>
                Random Forest Regressor marginally outperformed
                Linear Regression across MAE, RMSE, and R².
                Current chart position was the dominant
                regression feature.
            </p>

            <h3>Feature Importance</h3>

            <p>
                Days absent was the dominant classification
                feature. Current position was the dominant
                regression feature.
            </p>

            <h3>Important Limitations</h3>

            <p>
                The classification target is imbalanced,
                producing a false-positive burden that must
                be considered when interpreting predictions.
                The regression model's strong performance
                should also be interpreted in light of the
                persistence of chart positions between
                consecutive observations.
            </p>

            <p>
                Feature importance describes predictive
                usefulness within the fitted models. It
                does not establish causation.
            </p>

        </div>
        """))

    # --------------------------------------------------------
    # 15. DOWNLOAD MODEL RESULTS
    # --------------------------------------------------------

    st.markdown("---")

    ml_download_col1, ml_download_col2 = st.columns(2)

    with ml_download_col1:

        st.download_button(
            label="⬇️ Download Classification Results",
            data=classification_results.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="Classification_Model_Results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with ml_download_col2:

        st.download_button(
            label="⬇️ Download Regression Results",
            data=regression_results.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="Regression_Model_Results.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# END OF PART 10
# ============================================================

# ============================================================
# PART 11: EXECUTIVE INSIGHTS
# ============================================================

if page == "📊 Executive Insights":

    st.html(textwrap.dedent("""
        <div class="section-heading">
            📊 Executive Insights & Research Findings
        </div>

        <p class="main-subtitle">
            A consolidated interpretation of the South Korea
            Top 50 analytics project, including statistical
            findings, comeback patterns, and machine learning
            outcomes.
        </p>
        """))

    # --------------------------------------------------------
    # 1. PROJECT AT A GLANCE
    # --------------------------------------------------------

    st.html(textwrap.dedent('<div class="section-heading">🎯 Project at a Glance</div>'))

    overview1, overview2, overview3, overview4 = st.columns(4)

    with overview1:
        st.metric("Final Dataset Records", "27,700")

    with overview2:
        st.metric("Unique Songs", "537")

    with overview3:
        st.metric("Unique Artists", "194")

    with overview4:
        st.metric("Comeback Events", "874")

    # --------------------------------------------------------
    # 2. MAJOR ANALYTICAL FINDINGS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🔎 Major Analytical Findings</div>'))

    findings = [
        {
            "title": "🎤 Comeback Activity",
            "value": "874 comeback events",
            "description": (
                "The audited notebook identified 874 potential "
                "re-entry events across 230 unique songs."
            )
        },
        {
            "title": "🏆 Leading Comeback Song",
            "value": "Our Blues, Our Life",
            "description": (
                "The song by Lim Young Woong recorded "
                "28 re-entry events."
            )
        },
        {
            "title": "⏳ Post-Comeback Retention",
            "value": "8.86 days",
            "description": (
                "The audited notebook reported an average "
                "post-comeback retention of 8.86 calendar days."
            )
        },
        {
            "title": "💿 Release Type Comparison",
            "value": "No significant difference",
            "description": (
                "Singles showed higher average momentum than "
                "albums, but the difference was not statistically "
                "significant."
            )
        },
        {
            "title": "🤖 Classification Model",
            "value": "Random Forest",
            "description": (
                "Random Forest achieved the highest F1-score "
                "among the two classification models."
            )
        },
        {
            "title": "📈 Regression Model",
            "value": "Random Forest Regressor",
            "description": (
                "The model achieved an R² score of approximately "
                "0.9293 in the audited experiment."
            )
        }
    ]

    finding_columns = st.columns(2)

    for index, finding in enumerate(findings):

        with finding_columns[index % 2]:

            st.html(textwrap.dedent(f"""
                <div class="content-card">

                    <h3>{finding['title']}</h3>

                    <h2 style="color:#2563EB;">
                        {finding['value']}
                    </h2>

                    <p>{finding['description']}</p>

                </div>
                """))

    # --------------------------------------------------------
    # 3. ARTIST COMEBACK PERFORMANCE
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🎤 Artist Comeback Performance</div>'))

    artist_findings = pd.DataFrame({
        "Artist": [
            "Lim Young Woong",
            "Jimin",
            "NewJeans",
            "Jung Kook",
            "DAY6"
        ],
        "Comeback Events": [
            159,
            120,
            113,
            63,
            35
        ]
    })

    fig_artist_findings = px.bar(
        artist_findings.sort_values(
            "Comeback Events",
            ascending=True
        ),
        x="Comeback Events",
        y="Artist",
        orientation="h",
        text="Comeback Events",
        color="Comeback Events",
        color_continuous_scale="Purples",
        title="Top Artists by Audited Comeback Events"
    )

    fig_artist_findings.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420,
        xaxis_title="Comeback Events",
        yaxis_title="Artist"
    )

    st.plotly_chart(
        fig_artist_findings,
        use_container_width=True,
        config=PLOTLY_CONFIG
    )

    # --------------------------------------------------------
    # 4. ALBUM VS SINGLE STATISTICAL ANALYSIS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">💿 Album vs Single: Statistical Findings</div>'))

    album_test_results = pd.DataFrame({
        "Release Type": [
            "Album",
            "Single"
        ],
        "Unique Songs": [
            77,
            153
        ],
        "Comeback Events": [
            365,
            509
        ],
        "Average Momentum": [
            3.609,
            4.797
        ]
    })

    st.dataframe(
        album_test_results,
        use_container_width=True,
        hide_index=True
    )

    album_col1, album_col2 = st.columns(2)

    with album_col1:
        st.metric(
            "Mann-Whitney U Statistic",
            "5,659"
        )

    with album_col2:
        st.metric(
            "P-value",
            "0.625480"
        )

    st.info(
        "Interpretation: Although singles recorded a higher "
        "average comeback momentum score, the Mann-Whitney "
        "U test did not identify a statistically significant "
        "difference between the two release types."
    )

    # --------------------------------------------------------
    # 5. EXPLICIT VS CLEAN ANALYSIS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🎵 Explicit vs Clean Song Analysis</div>'))

    explicit_findings = pd.DataFrame({
        "Category": [
            "Clean Songs",
            "Explicit Songs"
        ],
        "Unique Songs": [
            211,
            19
        ],
        "Average Momentum": [
            3.844,
            10.562
        ]
    })

    st.dataframe(
        explicit_findings,
        use_container_width=True,
        hide_index=True
    )

    explicit_col1, explicit_col2 = st.columns(2)

    with explicit_col1:
        st.metric(
            "Mann-Whitney U Statistic",
            "1,288"
        )

    with explicit_col2:
        st.metric(
            "P-value",
            "0.009501"
        )

    st.success(
        "The song-level Mann-Whitney U test identified "
        "a statistically significant difference between "
        "the explicit and clean song groups at the 5% level."
    )

    st.warning(
        "The explicit category contains only 19 unique songs. "
        "The finding should be interpreted cautiously and "
        "does not establish that explicit content causes "
        "higher comeback momentum."
    )

    # --------------------------------------------------------
    # 6. MACHINE LEARNING INSIGHTS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🤖 Machine Learning Insights</div>'))

    ml_findings = pd.DataFrame({
        "Experiment": [
            "Classification",
            "Classification",
            "Regression",
            "Regression"
        ],
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Linear Regression",
            "Random Forest Regressor"
        ],
        "Primary Metric": [
            "F1 Score",
            "F1 Score",
            "R² Score",
            "R² Score"
        ],
        "Result": [
            "0.1808",
            "0.3773",
            "0.9273",
            "0.9293"
        ]
    })

    st.dataframe(
        ml_findings,
        use_container_width=True,
        hide_index=True
    )

    st.html(textwrap.dedent("""
        <div class="content-card">

            <h3>Classification Interpretation</h3>

            <p>
                Random Forest achieved a better F1-score
                than Logistic Regression. However, the
                classification dataset is highly imbalanced,
                and the model's precision remains limited.
            </p>

            <h3>Regression Interpretation</h3>

            <p>
                Random Forest Regressor achieved a slightly
                better R² score than Linear Regression.
                Current chart position was the dominant
                predictive feature.
            </p>

            <h3>Important Consideration</h3>

            <p>
                Predictive performance should be evaluated
                using multiple metrics. High accuracy or R²
                alone does not establish that a model will
                generalize equally well to future chart data.
            </p>

        </div>
        """))

    # --------------------------------------------------------
    # 7. FEATURE IMPORTANCE INSIGHTS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🔍 Most Important Predictive Features</div>'))

    feature_insights = pd.DataFrame({
        "Experiment": [
            "Classification",
            "Regression"
        ],
        "Most Important Feature": [
            "days_absent",
            "position"
        ],
        "Interpretation": [
            "Time absent from the chart was the dominant feature.",
            "Current chart position was the dominant feature."
        ]
    })

    st.dataframe(
        feature_insights,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # 8. KEY BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">💼 Key Business Insights</div>'))

    business_insights = [
        (
            "1. Chart Comebacks",
            "Repeated chart appearances can help identify songs "
            "with recurring chart visibility."
        ),
        (
            "2. Artist Performance",
            "Artist-level comeback patterns help compare the "
            "chart presence of different artists."
        ),
        (
            "3. Release Type",
            "The observed difference between albums and singles "
            "was not statistically significant in the audited test."
        ),
        (
            "4. Popularity Indicators",
            "Popularity values can be examined alongside chart "
            "positions and comeback patterns."
        ),
        (
            "5. Predictive Analytics",
            "Chronological machine learning experiments provide "
            "a framework for studying future comeback patterns."
        ),
        (
            "6. Data-Driven Decisions",
            "Combining descriptive analytics, statistical testing, "
            "and machine learning provides a broader understanding "
            "of chart behaviour."
        )
    ]

    for title, description in business_insights:

        st.html(textwrap.dedent(f"""
            <div class="content-card">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """))

    # --------------------------------------------------------
    # 9. RESEARCH LIMITATIONS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">⚠️ Research Limitations</div>'))

    limitations = [
        "The dataset represents chart snapshots rather than complete streaming histories.",
        "Chart appearances do not directly measure the number of listeners or streams.",
        "Fandom intensity is a chart-based proxy rather than a direct measure of fan engagement.",
        "The explicit-song category has a comparatively small number of unique songs.",
        "Classification performance is affected by substantial class imbalance.",
        "Regression performance is strongly influenced by the current chart position.",
        "Retention observations at the end of the dataset may be right-censored.",
        "The excluded 1 March 2025 snapshot requires caution when interpreting events around that date."
    ]

    for limitation in limitations:
        st.markdown(f"- {limitation}")

    # --------------------------------------------------------
    # 10. FUTURE SCOPE
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🚀 Future Scope & Recommendations</div>'))

    future_scope = [
        "Integrate additional chart datasets to compare multiple countries.",
        "Include verified streaming and listener statistics when available.",
        "Explore survival-analysis methods for retention modelling.",
        "Evaluate additional classification models and threshold strategies.",
        "Use external validation periods to test model generalization.",
        "Explore artist-level and genre-level comparative analysis.",
        "Develop more advanced interactive dashboards for music analytics."
    ]

    for recommendation in future_scope:
        st.markdown(f"- {recommendation}")

    # --------------------------------------------------------
    # 11. FINAL PROJECT CONCLUSION
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🏁 Overall Project Conclusion</div>'))

    st.html(textwrap.dedent("""
        <div class="content-card">

            <h2 style="color:#2563EB;">
                From Chart Data to Analytical Intelligence
            </h2>

            <p>
                This project analyses the South Korea Top 50
                playlist using data cleaning, exploratory
                analysis, comeback identification, momentum
                measurement, retention analysis, statistical
                testing, and machine learning.
            </p>

            <p>
                The findings reveal recurring chart appearances,
                differences in artist-level comeback activity,
                and measurable relationships between chart
                history and future chart behaviour.
            </p>

            <p>
                The project demonstrates how Python-based
                analytics and interactive visualization can
                transform raw chart records into structured
                analytical insights.
            </p>

            <h3>Project Technologies</h3>

            <p>
                Python | Pandas | NumPy | Matplotlib |
                Seaborn | Plotly | Streamlit |
                Scikit-learn | Google Colab
            </p>

        </div>
        """))

    # --------------------------------------------------------
    # 12. DOWNLOAD EXECUTIVE FINDINGS
    # --------------------------------------------------------

    executive_download = pd.DataFrame({
        "Finding": [
            "Final Dataset Records",
            "Unique Songs",
            "Unique Artists",
            "Comeback Events",
            "Songs with Comebacks",
            "Average Post-Comeback Retention",
            "Album vs Single P-value",
            "Explicit vs Clean Song-Level P-value",
            "Best Classification F1 Score",
            "Best Regression R2 Score"
        ],
        "Result": [
            "27,700",
            "537",
            "194",
            "874",
            "230",
            "8.86 days",
            "0.625480",
            "0.009501",
            "0.377323",
            "0.929298"
        ]
    })

    st.download_button(
        label="⬇️ Download Executive Findings CSV",
        data=executive_download.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="South_Korea_Top50_Executive_Findings.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# END OF PART 11
# ============================================================

# ============================================================
# PART 12: SONG EXPLORER
# ============================================================

if page == "🎶 Song Explorer":

    st.html(textwrap.dedent("""
        <div class="section-heading">
            🎶 Interactive Song Explorer
        </div>

        <p class="main-subtitle">
            Explore individual songs, historical chart positions,
            popularity trends, artist information, and comeback
            performance throughout the dataset.
        </p>
        """))

    # --------------------------------------------------------
    # 1. PREPARE SONG DATA
    # --------------------------------------------------------

    explorer_df = chart_df.copy()

    explorer_df["date"] = pd.to_datetime(
        explorer_df["date"],
        errors="coerce"
    )

    explorer_df = explorer_df.dropna(
        subset=["date", "song_id"]
    ).copy()

    explorer_df["song"] = explorer_df["song"].fillna("").astype(str)
    explorer_df["artist"] = explorer_df["artist"].fillna("").astype(str)

    # --------------------------------------------------------
    # 2. SONG SEARCH AND SELECTION
    # --------------------------------------------------------

    st.html(textwrap.dedent('<div class="section-heading">🔎 Find Your Song</div>'))

    search_song = st.text_input(
        "Search by song title or artist",
        placeholder="Type a song or artist name...",
        key="explorer_song_search"
    )

    song_options = (
        explorer_df[
            ["song_id", "song", "artist"]
        ]
        .drop_duplicates(subset=["song_id"])
        .sort_values(["song", "artist"])
        .copy()
    )

    if search_song.strip():

        search_term = search_song.strip().casefold()

        matching_songs = song_options[
            song_options["song"].str.casefold().str.contains(
                search_term,
                na=False,
                regex=False
            )
            |
            song_options["artist"].str.casefold().str.contains(
                search_term,
                na=False,
                regex=False
            )
        ].copy()

    else:
        matching_songs = song_options.copy()

    if matching_songs.empty:

        st.warning(
            "No matching songs found. Try another song "
            "title or artist name."
        )

    else:

        matching_songs["display_label"] = (
            matching_songs["song"]
            + " — "
            + matching_songs["artist"]
        )

        selected_label = st.selectbox(
            "🎵 Select a Song",
            options=matching_songs["display_label"].tolist(),
            key="explorer_selected_song"
        )

        selected_song_id = matching_songs.loc[
            matching_songs["display_label"] == selected_label,
            "song_id"
        ].iloc[0]

        # ----------------------------------------------------
        # 3. EXTRACT SELECTED SONG HISTORY
        # ----------------------------------------------------

        song_history = (
            explorer_df[
                explorer_df["song_id"] == selected_song_id
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if not song_history.empty:

            # -----------------------------------------------
            # 4. SONG INFORMATION
            # -----------------------------------------------

            song_name = song_history["song"].iloc[0]
            artist_name = song_history["artist"].iloc[0]

            album_type_values = song_history["album_type"].mode()

            album_type = (
                album_type_values.iloc[0]
                if not album_type_values.empty
                else "Unknown"
            )

            song_duration = pd.to_numeric(
                song_history["duration_ms"],
                errors="coerce"
            ).dropna()

            song_popularity = pd.to_numeric(
                song_history["popularity"],
                errors="coerce"
            ).dropna()

            explicit_status = song_history[
                "is_explicit"
            ].dropna()

            song_first_date = song_history["date"].min()
            song_last_date = song_history["date"].max()

            chart_appearances = len(song_history)

            unique_appearances = (
                song_history["date"].dt.normalize().nunique()
            )

            average_position = song_history["position"].mean()
            best_position = song_history["position"].min()
            worst_position = song_history["position"].max()

            average_popularity = (
                song_popularity.mean()
                if not song_popularity.empty
                else np.nan
            )

            maximum_popularity = (
                song_popularity.max()
                if not song_popularity.empty
                else np.nan
            )

            # -----------------------------------------------
            # 5. SONG COMEBACK CALCULATION
            # -----------------------------------------------

            song_dates = sorted(
                song_history["date"].dt.normalize().unique()
            )

            explorer_dates = sorted(
                explorer_df["date"].dt.normalize().unique()
            )

            explorer_date_index = {
                date: index
                for index, date in enumerate(explorer_dates)
            }

            song_comebacks = []

            for index in range(1, len(song_dates)):

                previous_date = song_dates[index - 1]
                comeback_date = song_dates[index]

                previous_index = explorer_date_index[previous_date]
                comeback_index = explorer_date_index[comeback_date]

                absent_dates = (
                    comeback_index - previous_index - 1
                )

                if absent_dates > 0:

                    previous_row = song_history[
                        song_history["date"].dt.normalize()
                        == previous_date
                    ].iloc[0]

                    comeback_row = song_history[
                        song_history["date"].dt.normalize()
                        == comeback_date
                    ].iloc[0]

                    previous_position = previous_row["position"]
                    comeback_position = comeback_row["position"]

                    song_comebacks.append({
                        "previous_date": previous_date,
                        "comeback_date": comeback_date,
                        "observed_dates_absent": absent_dates,
                        "days_absent": (
                            comeback_date - previous_date
                        ).days,
                        "previous_position": previous_position,
                        "comeback_position": comeback_position,
                        "rank_improvement": (
                            previous_position - comeback_position
                        )
                    })

            song_comeback_df = pd.DataFrame(song_comebacks)

            # -----------------------------------------------
            # 6. SONG HEADER
            # -----------------------------------------------

            st.markdown("---")

            st.html(textwrap.dedent(f"""
                <div class="content-card">

                    <h2>🎵 {song_name}</h2>

                    <h3 style="color:#6366F1;">
                        {artist_name}
                    </h3>

                    <p>
                        Release Type: <b>{album_type}</b>
                    </p>

                    <p>
                        First Chart Appearance:
                        <b>{song_first_date.strftime('%d %B %Y')}</b>
                    </p>

                    <p>
                        Latest Chart Appearance:
                        <b>{song_last_date.strftime('%d %B %Y')}</b>
                    </p>

                </div>
                """))

            # -----------------------------------------------
            # 7. SONG PERFORMANCE KPIs
            # -----------------------------------------------

            st.html(textwrap.dedent('<div class="section-heading">📊 Song Performance Overview</div>'))

            song_kpi1, song_kpi2, song_kpi3, song_kpi4 = st.columns(4)

            with song_kpi1:
                st.metric("Chart Appearances", f"{chart_appearances:,}")

            with song_kpi2:
                st.metric("Average Position", f"{average_position:.2f}")

            with song_kpi3:
                st.metric("Best Position", f"#{int(best_position)}")

            with song_kpi4:
                st.metric("Comeback Events", f"{len(song_comeback_df):,}")

            song_kpi5, song_kpi6, song_kpi7, song_kpi8 = st.columns(4)

            with song_kpi5:
                st.metric(
                    "Average Popularity",
                    f"{average_popularity:.2f}"
                    if pd.notna(average_popularity)
                    else "N/A"
                )

            with song_kpi6:
                st.metric(
                    "Maximum Popularity",
                    f"{maximum_popularity:.0f}"
                    if pd.notna(maximum_popularity)
                    else "N/A"
                )

            with song_kpi7:
                st.metric("Worst Position", f"#{int(worst_position)}")

            with song_kpi8:
                st.metric("Unique Chart Dates", f"{unique_appearances:,}")

                    # -----------------------------------------------
            # 8. SONG METADATA
            # -----------------------------------------------

            st.markdown("---")

            st.html(textwrap.dedent('<div class="section-heading">ℹ️ Song Metadata</div>'))

            metadata_col1, metadata_col2 = st.columns(2)

            with metadata_col1:

                st.write("**Song Duration**")

                if not song_duration.empty:

                    duration_seconds = song_duration.mean() / 1000
                    duration_minutes = int(duration_seconds // 60)
                    remaining_seconds = int(duration_seconds % 60)

                    st.write(
                        f"{duration_minutes}:{remaining_seconds:02d}"
                    )

                else:
                    st.write("Unavailable")

                st.write("**Explicit Content**")

                if not explicit_status.empty:

                    explicit_value = explicit_status.iloc[0]

                    st.write(
                        "Yes"
                        if bool(explicit_value)
                        else "No"
                    )

                else:
                    st.write("Unknown")

            with metadata_col2:

                st.write("**Album Type**")
                st.write(str(album_type))

                st.write("**Artist**")
                st.write(str(artist_name))

            # -----------------------------------------------
            # 9. HISTORICAL CHART POSITION
            # -----------------------------------------------

            st.markdown("---")

            st.html(textwrap.dedent('<div class="section-heading">📈 Historical Chart Position</div>'))

            fig_song_position = px.line(
                song_history,
                x="date",
                y="position",
                markers=True,
                title=f"Chart Position History — {song_name}",
                hover_data=[
                    "popularity",
                    "artist"
                ]
            )

            fig_song_position.update_yaxes(
                autorange="reversed",
                dtick=5
            )

            fig_song_position.update_layout(
                template=PLOTLY_TEMPLATE,
                height=450,
                xaxis_title="Chart Date",
                yaxis_title="Chart Position"
            )

            st.plotly_chart(
                fig_song_position,
                use_container_width=True,
                config=PLOTLY_CONFIG
            )

            # -----------------------------------------------
            # 10. POPULARITY HISTORY
            # -----------------------------------------------

            st.markdown("---")

            st.html(textwrap.dedent('<div class="section-heading">🔥 Song Popularity History</div>'))

            popularity_history = song_history.dropna(
                subset=["popularity"]
            )

            if not popularity_history.empty:

                fig_song_popularity = px.area(
                    popularity_history,
                    x="date",
                    y="popularity",
                    title=f"Popularity Trend — {song_name}",
                    color_discrete_sequence=[
                        SECONDARY_COLOR
                    ]
                )

                fig_song_popularity.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=380,
                    xaxis_title="Chart Date",
                    yaxis_title="Popularity Score"
                )

                st.plotly_chart(
                    fig_song_popularity,
                    use_container_width=True,
                    config=PLOTLY_CONFIG
                )

            else:

                st.info(
                    "Popularity history is unavailable "
                    "for this song."
                )

            # -----------------------------------------------
            # 11. COMEBACK HISTORY
            # -----------------------------------------------

            st.markdown("---")

            st.html(textwrap.dedent('<div class="section-heading">🎤 Song Comeback History</div>'))

            if not song_comeback_df.empty:

                st.metric(
                    "Total Comeback Events",
                    len(song_comeback_df)
                )

                fig_song_comeback = px.bar(
                    song_comeback_df,
                    x="comeback_date",
                    y="rank_improvement",
                    hover_data=[
                        "days_absent",
                        "observed_dates_absent",
                        "previous_position",
                        "comeback_position"
                    ],
                    title="Rank Improvement During Comebacks",
                    color="rank_improvement",
                    color_continuous_scale="RdBu"
                )

                fig_song_comeback.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray"
                )

                fig_song_comeback.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=400,
                    xaxis_title="Comeback Date",
                    yaxis_title="Rank Improvement"
                )

                st.plotly_chart(
                    fig_song_comeback,
                    use_container_width=True,
                    config=PLOTLY_CONFIG
                )

                st.dataframe(
                    song_comeback_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No comeback events were identified "
                    "for this song in the available chart history."
                )

            # -----------------------------------------------
            # 12. SONG CHART HISTORY TABLE
            # -----------------------------------------------

            st.markdown("---")

            st.html(textwrap.dedent('<div class="section-heading">📋 Complete Song Chart History</div>'))

            history_columns = [
                "date",
                "song",
                "artist",
                "position",
                "popularity",
                "album_type",
                "is_explicit"
            ]

            song_history_display = (
                song_history[history_columns]
                .sort_values(
                    "date",
                    ascending=False
                )
            )

            st.dataframe(
                song_history_display,
                use_container_width=True,
                hide_index=True
            )

            # -----------------------------------------------
            # 13. DOWNLOAD SONG HISTORY
            # -----------------------------------------------

            song_history_csv = song_history.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Selected Song History",
                data=song_history_csv,
                file_name="Selected_Song_Chart_History.csv",
                mime="text/csv",
                use_container_width=True
            )

            # -----------------------------------------------
            # 14. SONG EXPLORER INTERPRETATION
            # -----------------------------------------------

            st.markdown("---")

            st.html(textwrap.dedent('<div class="section-heading">💡 Song Performance Interpretation</div>'))

            st.html(textwrap.dedent("""
                <div class="content-card">

                    <h3>Chart Position</h3>

                    <p>
                        A lower numerical chart position
                        represents a stronger ranking.
                    </p>

                    <h3>Popularity</h3>

                    <p>
                        Popularity values represent the
                        popularity information recorded
                        in the source dataset.
                    </p>

                    <h3>Comeback Activity</h3>

                    <p>
                        A comeback is identified when a song
                        returns after at least one intervening
                        observed chart date on which it was absent.
                    </p>

                    <h3>Interpretation</h3>

                    <p>
                        Song-level results describe the
                        historical chart observations available
                        in this dataset. They should not be
                        interpreted as direct streaming or
                        listener measurements.
                    </p>

                </div>
                """))

        else:

            st.warning("No chart history is available for this song.")


# ============================================================
# END OF PART 12
# ============================================================

# ============================================================
# PART 13: PROJECT INFORMATION & APPLICATION FOOTER
# ============================================================

if page == "📁 Project Information":

    # --------------------------------------------------------
    # 1. PAGE INTRODUCTION
    # --------------------------------------------------------

    st.html(textwrap.dedent("""
        <div class="section-heading">
            📁 Project Information & Documentation
        </div>

        <p class="main-subtitle">
            Explore the project background, research objectives,
            analytical methodology, technologies, and dataset
            documentation.
        </p>
        """))

    # --------------------------------------------------------
    # 2. PROJECT TITLE
    # --------------------------------------------------------

    st.html(textwrap.dedent(f"""
        <div class="content-card">

            <h1 style="color:#2563EB;">
                {PROJECT_NAME}
            </h1>

            <h3>{PROJECT_SUBTITLE}</h3>

            <p>
                A comprehensive music chart analytics project
                combining exploratory data analysis, comeback
                analysis, statistical testing, machine learning,
                and interactive visualization.
            </p>

        </div>
        """))

    # --------------------------------------------------------
    # 3. DEVELOPER INFORMATION
    # --------------------------------------------------------

    st.html(textwrap.dedent('<div class="section-heading">👨‍💻 Developer Information</div>'))

    developer_col1, developer_col2 = st.columns(2)

    with developer_col1:

        st.html(textwrap.dedent(f"""
            <div class="content-card">

                <h3>Developer</h3>

                <h2>{DEVELOPER_NAME}</h2>

                <p>
                    <b>University:</b><br>
                    {UNIVERSITY_NAME}
                </p>

                <p>
                    <b>Programme:</b><br>
                    {COURSE_NAME}
                </p>

            </div>
            """))

    with developer_col2:

        st.html(textwrap.dedent("""
            <div class="content-card">

                <h3>Project Details</h3>

                <p>
                    <b>Project Category:</b><br>
                    Data Analytics & Machine Learning
                </p>

                <p>
                    <b>Domain:</b><br>
                    Music Industry Analytics
                </p>

                <p>
                    <b>Project Year:</b><br>
                    2026
                </p>

            </div>
            """))

    # --------------------------------------------------------
    # 4. PROJECT ABSTRACT
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">📖 Project Abstract</div>'))

    st.html(textwrap.dedent("""
        <div class="content-card">

            <p>
                This project presents an analytical study of
                South Korea's Top 50 music playlist. It examines
                chart performance, song re-entry patterns,
                comeback momentum, post-comeback retention,
                and artist-level chart activity.
            </p>

            <p>
                The analysis uses Python-based data processing
                and visualization techniques to transform raw
                music chart records into structured analytical
                insights.
            </p>

            <p>
                Statistical hypothesis testing is used to
                investigate differences between selected song
                categories. Machine learning experiments are
                conducted to study comeback classification
                and next-chart-position prediction.
            </p>

            <p>
                Finally, an interactive Streamlit dashboard
                brings the analytical findings together through
                visualizations, performance indicators,
                interactive filters, and downloadable reports.
            </p>

        </div>
        """))

    # --------------------------------------------------------
    # 5. PROJECT OBJECTIVES
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🎯 Project Objectives</div>'))

    objectives = [
        "Analyse historical South Korea Top 50 chart performance.",
        "Identify songs that disappear and subsequently return to the chart.",
        "Measure comeback frequency and ranking momentum.",
        "Examine post-comeback chart retention patterns.",
        "Compare artist-level chart performance.",
        "Develop a chart-based fandom intensity proxy.",
        "Perform statistical comparisons between selected song categories.",
        "Build machine learning models for comeback classification and chart-position regression.",
        "Develop an interactive analytics dashboard using Streamlit."
    ]

    for index, objective in enumerate(objectives, start=1):
        st.markdown(f"**{index}.** {objective}")

    # --------------------------------------------------------
    # 6. DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">📂 Dataset Information</div>'))

    dataset_info = pd.DataFrame({
        "Dataset Attribute": [
            "Dataset Name",
            "Original Records",
            "Original Columns",
            "Final Records",
            "Final Columns",
            "Unique Chart Dates",
            "Unique Songs",
            "Unique Artists",
            "Chart Size",
            "Excluded Snapshot"
        ],
        "Description": [
            DATASET_NAME,
            "27,800",
            "10",
            f"{len(chart_df):,}",
            str(len(chart_df.columns)),
            f"{chart_df['date'].nunique():,}",
            f"{chart_df['song_id'].nunique():,}",
            f"{chart_df['artist'].nunique():,}",
            "Top 50",
            "2025-03-01"
        ]
    })

    st.dataframe(
        dataset_info,
        use_container_width=True,
        hide_index=True
    )

    if DATA_START_DATE and DATA_END_DATE:

        st.info(
            f"Available dataset period: "
            f"{DATA_START_DATE} to {DATA_END_DATE}"
        )

    # --------------------------------------------------------
    # 7. DATA DICTIONARY
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">📘 Dataset Data Dictionary</div>'))

    data_dictionary = pd.DataFrame({
        "Column": [
            "date",
            "position",
            "song",
            "artist",
            "popularity",
            "duration_ms",
            "album_type",
            "total_tracks",
            "is_explicit",
            "album_cover_url",
            "song_id"
        ],
        "Description": [
            "Date of the chart observation.",
            "Song's position in the Top 50.",
            "Name of the music track.",
            "Artist associated with the track.",
            "Popularity score available in the dataset.",
            "Song duration measured in milliseconds.",
            "Release classification such as album or single.",
            "Number of tracks associated with the release.",
            "Indicates whether the song contains explicit content.",
            "URL associated with the album cover.",
            "Combined song and artist identifier created during preprocessing."
        ]
    })

    st.dataframe(
        data_dictionary,
        use_container_width=True,
        hide_index=True
    )

        # --------------------------------------------------------
    # 8. TECHNOLOGIES USED
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🛠️ Technologies & Tools</div>'))

    technology_groups = {
        "Programming Language": [
            "Python"
        ],
        "Data Processing": [
            "Pandas",
            "NumPy"
        ],
        "Data Visualization": [
            "Matplotlib",
            "Seaborn",
            "Plotly"
        ],
        "Machine Learning": [
            "Scikit-learn"
        ],
        "Dashboard Development": [
            "Streamlit"
        ],
        "Development Environment": [
            "Google Colab",
            "Visual Studio Code"
        ],
        "Project Management & Version Control": [
            "GitHub"
        ]
    }

    for category, technologies in technology_groups.items():

        st.markdown(f"**{category}**")

        technology_columns = st.columns(
            min(len(technologies), 4)
        )

        for index, technology in enumerate(technologies):

            with technology_columns[index % len(technology_columns)]:

                st.html(textwrap.dedent(f"""
                    <div class="content-card"
                         style="text-align:center; padding:15px;">
                        <b>{technology}</b>
                    </div>
                    """))

    # --------------------------------------------------------
    # 9. ANALYTICAL METHODOLOGY
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">⚙️ Analytical Methodology</div>'))

    methodology_steps = [
        (
            "01. Data Collection",
            "Load historical South Korea Top 50 playlist records."
        ),
        (
            "02. Data Cleaning",
            "Remove exact duplicates, handle invalid values, "
            "and exclude the unresolved chart snapshot."
        ),
        (
            "03. Exploratory Data Analysis",
            "Study chart positions, song popularity, artists, "
            "and release categories."
        ),
        (
            "04. Comeback Identification",
            "Identify songs returning after at least one "
            "intervening observed chart date."
        ),
        (
            "05. Momentum Analysis",
            "Examine chart-position and popularity changes "
            "associated with comeback events."
        ),
        (
            "06. Retention Analysis",
            "Measure consecutive observed chart presence "
            "after a comeback."
        ),
        (
            "07. Statistical Testing",
            "Use non-parametric hypothesis testing to compare "
            "selected song categories."
        ),
        (
            "08. Machine Learning",
            "Evaluate classification and regression models "
            "using chronological train-test separation."
        ),
        (
            "09. Dashboard Development",
            "Present findings through interactive Streamlit "
            "visualizations and analytical reports."
        )
    ]

    for title, description in methodology_steps:

        st.html(textwrap.dedent(f"""
            <div class="content-card">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """))

    # --------------------------------------------------------
    # 10. RESEARCH QUESTIONS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🔬 Research Questions</div>'))

    research_questions = [
        "Which songs demonstrate the highest comeback frequency?",
        "Which artists record the most comeback events?",
        "How does chart position change when songs return?",
        "How long do songs remain on the chart after returning?",
        "Is there a statistically significant difference in comeback momentum between albums and singles?",
        "Does explicit-content classification show a statistical association with comeback momentum?",
        "Can machine learning identify songs likely to return to the chart?",
        "Can a regression model estimate the next observed chart position?"
    ]

    for index, question in enumerate(research_questions, start=1):

        st.markdown(
            f"**Research Question {index}:** {question}"
        )

    # --------------------------------------------------------
    # 11. PROJECT FINDINGS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🏆 Major Project Findings</div>'))

    project_findings = pd.DataFrame({
        "Finding": [
            "Comeback Events",
            "Songs with Comebacks",
            "Average Retention",
            "Top Comeback Song",
            "Top Comeback Artist",
            "Album vs Single P-value",
            "Explicit vs Clean P-value",
            "Classification F1 Score",
            "Regression R² Score"
        ],
        "Result": [
            "874",
            "230",
            "8.86 days",
            "Our Blues, Our Life",
            "Lim Young Woong",
            "0.625480",
            "0.009501",
            "0.377323",
            "0.929298"
        ]
    })

    st.dataframe(
        project_findings,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # 12. RESEARCH LIMITATIONS
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">⚠️ Research Limitations</div>'))

    st.html(textwrap.dedent("""
        <div class="content-card">

            <p>
                The dataset contains chart observations rather
                than complete streaming or audience-engagement
                records.
            </p>

            <p>
                Fandom intensity is a proxy based on chart
                indicators and cannot directly quantify
                fan loyalty or engagement.
            </p>

            <p>
                The explicit-song group has a relatively
                small number of unique songs.
            </p>

            <p>
                The classification dataset has substantial
                class imbalance, which affects precision
                and prediction reliability.
            </p>

            <p>
                Retention observations at the end of the
                dataset may be right-censored.
            </p>

        </div>
        """))

    # --------------------------------------------------------
    # 13. FUTURE SCOPE
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">🚀 Future Scope</div>'))

    future_scope_items = [
        "Expand the analysis to additional international music charts.",
        "Integrate verified streaming and listener statistics.",
        "Explore survival analysis for chart-retention modelling.",
        "Investigate more advanced predictive models.",
        "Compare artist performance across multiple countries.",
        "Develop additional interactive reporting capabilities."
    ]

    for item in future_scope_items:
        st.markdown(f"- {item}")

    # --------------------------------------------------------
    # 14. PROJECT DATA DOWNLOAD
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent('<div class="section-heading">⬇️ Download Project Data</div>'))

    project_csv = chart_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Cleaned South Korea Dataset",
        data=project_csv,
        file_name="South_Korea_Top50_Cleaned_Dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    # --------------------------------------------------------
    # 15. PROJECT ACKNOWLEDGEMENT
    # --------------------------------------------------------

    st.markdown("---")

    st.html(textwrap.dedent(f"""
        <div class="content-card"
             style="text-align:center; padding:30px;">

            <h2 style="color:#2563EB;">
                Thank You
            </h2>

            <h3>{DEVELOPER_NAME}</h3>

            <p>Data Analytics Intern | Unified Mentor</p>

            <p>South Korea Top 50 Music Analytics Project</p>

            <p>{PROJECT_YEAR}</p>

        </div>
        """))


# ============================================================
# GLOBAL APPLICATION FOOTER
# ============================================================

st.html(textwrap.dedent(f"""
    <div class="dashboard-footer">

        <h4>
            🎵 {PROJECT_SUBTITLE} Analytics
        </h4>

        <p>{PROJECT_NAME}</p>

        <p>Developed by {DEVELOPER_NAME}</p>

        <p>Data Analytics Intern | Unified Mentor</p>

        <p style="font-size:11px; opacity:0.7;">
            Python | Pandas | Plotly | Scikit-learn | Streamlit
        </p>

        <p style="font-size:11px; opacity:0.7;">
            © 2026 | All Rights Reserved
        </p>

    </div>
    """))


# ============================================================
# END OF PART 13
# ============================================================
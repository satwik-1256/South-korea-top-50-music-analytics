# South Korea Top 50 Music Analytics

An end-to-end music analytics project studying chart re-entries, comeback momentum, and chart retention in South Korea's Top 50 playlist data. It covers data preparation, exploratory analysis, statistical testing, machine learning, and an interactive Streamlit dashboard.

## Overview

This project analyzes historical chart observations to explore how songs return to the chart, how their chart positions and popularity change around a comeback, and how long they remain on the chart after returning. The work brings together data cleaning, exploratory data analysis, statistical comparisons, machine-learning analyses, and an interactive dashboard for exploring the results.

## Project components

- **Data preparation:** validation, cleaning, duplicate handling, and processed-data review.
- **Exploratory analysis:** chart positions, popularity, songs, artists, and release categories.
- **Comeback analysis:** song re-entry frequency, chart-position and popularity changes, and post-comeback retention.
- **Statistical analysis:** non-parametric comparisons across selected song categories.
- **Machine learning:** classification and regression workflows with evaluation summaries.
- **Interactive presentation:** a Streamlit dashboard with filters, visualizations, project documentation, and cleaned-data download.

## Analyses

The application identifies a comeback when a song returns after at least one intervening observed chart date. It examines chart-position and popularity changes around comeback events and measures consecutive observed chart presence after a return. It also compares selected categories using non-parametric statistical tests and presents classification and regression analyses.

The analyses describe the supplied chart data. Chart records do not directly measure streaming activity, listener engagement, or fan loyalty; the app treats fandom intensity as a proxy based on chart indicators. The classification data is imbalanced, and retention observations near the end of the dataset may be right-censored.

## Requirements

- Python 3.10 or later
- The dataset file `Atlantic_South_Korea.csv`

Python packages:

- Streamlit
- pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- scikit-learn

## Setup and run

1. Clone or download this repository.
2. Place `Atlantic_South_Korea.csv` in the same directory as `app.py`.
3. Create and activate a virtual environment (recommended).
4. Install the packages:

   ```bash
   python -m pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn
   ```

5. Start the interactive project explorer:

   ```bash
   streamlit run app.py
   ```

The app reads `Atlantic_South_Korea.csv` from the directory containing `app.py`. The dataset is not included in this repository; make sure you have permission to use and share it before adding it to a public repository.

## Dataset fields

The dashboard's data dictionary describes fields including chart date and position, track and artist names, popularity, duration, release type, track count, explicit-content flag, album-cover URL, and a processed song identifier. The exact columns required depend on the supplied CSV; use the app's data validation report to inspect the loaded file.

## Technology

- Python for analysis and application logic
- pandas and NumPy for data preparation
- Matplotlib, Seaborn, and Plotly for visualization
- scikit-learn for machine-learning workflows
- Streamlit for the interactive dashboard

## Author

**Satwik Srivastava**  
Data Analytics Intern | Unified Mentor

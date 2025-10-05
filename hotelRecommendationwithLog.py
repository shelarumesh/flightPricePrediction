import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings
import logging
import os
import pickle

warnings.filterwarnings('ignore')

# --- 1. LOGGING SETUP ---
# Determine the absolute path for the log file (in the same directory as this script)
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recommendation_app.log")

# Configure logging to write to a file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Using the absolute path ensures the file is created next to the script
        logging.FileHandler(LOG_FILE_PATH), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. DATA AND MODEL PREPARATION ---

path = 'D:/AlmaBetter/P01_travelPrice/data/hotels.csv'
try:
    # Load the dataset
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded successfully from: {path}")
except FileNotFoundError:
    logger.error(f"FATAL ERROR: hotels.csv not found at {path}. Exiting script.")
    st.error(f"FATAL ERROR: hotels.csv not found. Check path: {path}")
    # Since the app can't run without data, we stop execution or use placeholder data
    df = pd.DataFrame({'days': [], 'place': [], 'price': [], 'total': [], 'name': [], 'travelCode': []}) 
    # Use a small sample size to avoid memory issues during similarity calculation
    data = df[:5000] if not df.empty else df
    data_sampled = data.copy()


# Ensure data is processed only if DataFrame is not empty
if not df.empty:
    data = df[:5000]
    N_SAMPLES = 5000 
    data_sampled = data.sample(n=min(N_SAMPLES, len(data)), random_state=42).reset_index(drop=True)
    logger.info(f"Data sampled to {len(data_sampled)} rows for cosine similarity calculation.")

    # Data Preprocessing on Sample
    data_sampled['Hotel_Info'] = data_sampled['name'].astype(str).str.cat(data_sampled['place'].astype(str), sep='|')
    
    # Create and fit TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_sampled = tfidf_vectorizer.fit_transform(data_sampled['Hotel_Info'])
    logger.info("TF-IDF matrix created successfully.")

    # Compute the cosine similarity on the smaller matrix
    cosine_sim = linear_kernel(tfidf_matrix_sampled, tfidf_matrix_sampled)
    logger.info(f"Cosine similarity matrix shape: {cosine_sim.shape}")
else:
    # Set placeholders if data loading failed
    cosine_sim = np.array([[]])
    data_sampled = df

# --- 3. RECOMMENDATION LOGIC ---

def get_hotel_recommendations(place, days, price, total, cosine_sim=cosine_sim):
    logger.info(f"Recommendation request received: Place={place}, Days={days}, Price={price}, Total={total}")

    # Ensure data is filtered against the full dataset if possible, but use the sampled data structure for similarity lookup
    
    # Filter the sampled dataset based on the given criteria
    filtered_data = data_sampled[
        (data_sampled['place'] == place) &
        (data_sampled['days'] <= days) &
        (data_sampled['price'] <= price) &
        (data_sampled['total'] <= total)
    ]

    if filtered_data.empty:
        logger.warning("No matching hotels found after initial filtering criteria.")
        return "No matching hotels found."

    logger.info(f"Found {len(filtered_data)} hotels matching basic criteria.")
    
    # Get the indices of the filtered hotels relative to the sampled matrix
    # The index used here must match the index used when cosine_sim was created
    hotel_indices = filtered_data.index.tolist()

    # Calculate the average cosine similarity score for each hotel
    avg_similarity_scores = []
    try:
        for idx in hotel_indices:
            # We use the index relative to the sampled data for cosine_sim lookup
            avg_score = cosine_sim[idx].sum() / len(cosine_sim[idx])
            avg_similarity_scores.append(avg_score)
    except IndexError as e:
        logger.error(f"IndexError during similarity calculation: {e}. Check index alignment.")
        return "Internal error during similarity calculation."

    # Create a DataFrame to store the filtered hotels and their average similarity scores
    recommended_hotels_df = pd.DataFrame({
        'Hotel Details': filtered_data['name'],
        'Avg Similarity Score': avg_similarity_scores
    })

    # Sort and return the top results
    recommended_hotels_df = recommended_hotels_df.sort_values(by='Avg Similarity Score', ascending=False)
    
    logger.info(f"Returning top {len(recommended_hotels_df)} recommended hotels.")
    return recommended_hotels_df['Hotel Details'].value_counts()


# --- 4. STREAMLIT APP LAYOUT ---

st.title("Hotel Recommendation Web App : ")
st.write(
    "This is Travel hotels recommendation web app where user will "
    "interact and get best choice of hotel based on its selected features : "
)

if not df.empty:
    days_options = sorted(df['days'].unique().tolist())
    places_options = sorted(df['place'].unique().tolist())
    prices_max = int(df['price'].max())
    totals_max = int(df['total'].max())

    days = st.selectbox("Select Number of Days : ", days_options)
    place = st.selectbox("Select Place : ", places_options)
    price = st.slider("Select Price : ", min_value=0, max_value=prices_max)
    total = st.slider("Select Total : ", min_value=0, max_value=totals_max)

    if st.button("Show Recommended Hotels"):
        logger.info("Show Recommended Hotels button pressed.")
        recommended_hotels = get_hotel_recommendations(place, days, price, total)
        st.write(recommended_hotels)
else:
    st.error("Cannot run app: Data failed to load or is empty.")

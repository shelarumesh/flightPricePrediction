from flask import Flask, request,render_template
import pandas as pd
from sklearn.compose import ColumnTransformer
import numpy as np
import pickle
import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

# Load the dataset (assuming 'df' is available)
path = 'D:/AlmaBetter/P01_travelPrice/data/hotels.csv'
df = pd.read_csv(path)
df.head()
data = df[:5000]

# Reduce the size of your recommendation set significantly to fit in RAM.
# Adjust N based on your available memory (try 5000 or less first).
N_SAMPLES = 5000 
data_sampled = data.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)

# Data Preprocessing on Sample
data_sampled['Hotel_Info'] = data_sampled['name'].astype(str).str.cat(data_sampled['place'].astype(str), sep='|')

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the vectorizer on the sampled data
# The resulting matrix size will now be N_SAMPLES x Features
tfidf_matrix_sampled = tfidf_vectorizer.fit_transform(data_sampled['Hotel_Info'])

# Compute the cosine similarity on the smaller matrix
# This matrix will be N_SAMPLES x N_SAMPLES (much smaller than 40k x 40k)
cosine_sim = linear_kernel(tfidf_matrix_sampled, tfidf_matrix_sampled)

print(f"New similarity matrix shape: {cosine_sim.shape}")
print("Successfully created similarity matrix without MemoryError.")

# Note: You must now update your get_hotel_recommendations function 
# to use data_sampled instead of the original 'data' 
# and use the indices of the sampled dataset.





# Function to get hotel recommendations based on Package Type, Start City, Price, and Destination
def get_hotel_recommendations(place, days, price, total, cosine_sim=cosine_sim):
    # Filter the dataset based on the given criteria
    filtered_data = data[(data['place'] == place) &
                         (data['days'] <= days) &
                         (data['price'] <= price) &
                         (data['total'] <= total)]

    if filtered_data.empty:
        return "No matching hotels found."

    # Get the indices of the filtered hotels
    hotel_indices = filtered_data.index[:5000]

    # Calculate the average cosine similarity score for each hotel with the filtered hotels
    avg_similarity_scores = []
    for idx in hotel_indices[:5000]:
        avg_score = sum(cosine_sim[idx]) / len(cosine_sim[idx])
        avg_similarity_scores.append(avg_score)

    # Create a DataFrame to store the filtered hotels and their average similarity scores
    recommended_hotels_df = pd.DataFrame({'travelCode': filtered_data['travelCode'],
                                          'Hotel Details': filtered_data['name'],
                                          'Avg Similarity Score': avg_similarity_scores})

    # Sort the hotels by average similarity score in descending order
    recommended_hotels_df = recommended_hotels_df.sort_values(by='Avg Similarity Score', ascending=False)

    # Return the recommended hotel details
    return recommended_hotels_df[['travelCode', 'Hotel Details']]

# Example usage: Get hotel recommendations based on input criteria
days = 3
place = 'Salvador (BH)'
price = 300  # Specify your desired price
total = 1200

recommended_hotels = get_hotel_recommendations(place, days, price, total)
print(recommended_hotels)


# Streamlit Web App title
st.title("Hotel Recommendation Web App : ")
# user need to select information about data
st.write("This is Travel hotels recommendation web app where user will " \
"interact and get best choice of hotel based on its selected features : ")

days = df['days'].unique().tolist()
places = df['place'].unique().tolist()
prices_max = int(df['price'].max())
totals_max = int(df['total'].max())


days = st.selectbox("Select Number of Days : ", days)
place = st.selectbox("Select Place : ", places)
price = st.slider("Select Price : ", min_value=0, max_value=prices_max)
total = st.slider("Select Total : ", min_value=0, max_value=totals_max)

if st.button("Show Recommended Hotels"):
    recommended_hotels = get_hotel_recommendations(place, days, price, total)
    st.write(recommended_hotels)



# User inputs for Package Type, Start City, Price, and Destination
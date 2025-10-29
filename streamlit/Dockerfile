# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY ./app_st.py /app/app_st.py
COPY ./save_model/transformer.pkl /app/save_model/
COPY ./save_model/decision_tree_regressor_model.pkl /app/save_model/

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "/app/app_st.py", "--server.port=8501", "--server.address=0.0.0.0"]

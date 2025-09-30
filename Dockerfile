# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to the absolute path /app
WORKDIR /app

# 1. Copy requirements.txt and install dependencies FIRST
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy application source code and assets
# Note: The "save_model" folder is now correctly placed under /app/save_model/
RUN mkdir -p /app/save_model  # Ensure the directory exists

COPY ./app.py /app/app.py
COPY ./templates/index.html /app/templates/

# Copy the models into the correct subdirectory
COPY ./save_model/transformer.pkl /app/save_model/
COPY ./save_model/decision_tree_regressor_model.pkl /app/save_model/

# Expose the port the app runs on (Informational for Docker)
EXPOSE 5000

# Run the application (This keeps the container running)
CMD ["python", "app.py"]
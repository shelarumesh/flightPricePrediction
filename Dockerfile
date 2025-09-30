# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR app/

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /app/requirements.txt
COPY ./save_model/decision_tree_regressor_model.pkl /app/save_model/decision_tree_regressor_model.pkl
COPY ./save_model/transformer.pkl /app/save_model/transformer.pkl
COPY ./templates/index.html /app/templates/index.html
COPY ./app.py /app/app.py
RUN pip install -r requirements.txt

# Copy application files
COPY . /app

#AlmaBetter/P01_travelPrice/save_model/decision_tree_regressor_model.pkl
# Expose the port the app runs on (change if needed)
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
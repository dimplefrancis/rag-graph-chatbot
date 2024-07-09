# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY streamlit_app.py .
COPY images ./images

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run streamlit when the container launches
CMD ["streamlit", "run", "streamlit_app.py"]
# Use the official Python image as a base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
     git \
     build-essential \
     cmake && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Dash listens on
EXPOSE 8050

# Set the environment variable to ensure Python output is displayed in Docker logs
ENV PYTHONUNBUFFERED=1

# Command to run the app
CMD ["python", "app.py"]

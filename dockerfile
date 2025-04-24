# Use official Python runtime as the base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file first (optimization for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 5000 (default Flask port)
EXPOSE 5000

# Command to run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

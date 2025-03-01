# Use Python 3.10 as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run FastAPI using Uvicorn with correct host and port settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

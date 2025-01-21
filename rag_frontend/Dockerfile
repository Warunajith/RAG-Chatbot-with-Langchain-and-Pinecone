# Streamlit Dockerfile

# Use the official Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port (e.g., 8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "rag_fe.py", "--server.port", "8501", "--server.enableCORS", "false"]

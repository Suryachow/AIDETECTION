# Use an official Python image
FROM python:3.11

# Install Java (for language-tool-python)
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy project files to the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Run the app using the startup script
CMD ["bash", "run.sh"]

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Start with shell to expand environment variables
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT app:app"]

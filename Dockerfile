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

# Expose port (Railway will use PORT environment variable)
EXPOSE 8000

# Start command - use shell form to expand $PORT variable
# For Flask with Gunicorn (RECOMMENDED):
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} app:app

# Or for Flask without Gunicorn - add this to your app.py:
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8000))
#     app.run(host='0.0.0.0', port=port)

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

# Start command - adjust based on your app
# For Flask:
CMD ["python", "app.py"]

# Or for Django:
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# Or with Gunicorn (recommended for production):
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
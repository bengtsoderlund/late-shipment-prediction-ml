# Use slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install curl for downloading model files
RUN apt-get update && apt-get install -y curl && apt-get clean

# Make sure models folder exists
RUN mkdir -p models

# Download model files from Dropbox
RUN curl -L -o models/late_model.pkl https://www.dropbox.com/scl/fi/9k6heuy8wuh9rufpf19wl/late_model.pkl?rlkey=gsqy6gtzfgsvfmyi7iv7oo0ob\&st=m7z6koyf\&dl=1 && \
    curl -L -o models/onehot_encoder.pkl https://www.dropbox.com/scl/fi/hiy15evl6z7o7m1n57f1z/onehot_encoder.pkl?rlkey=vh611s4sp1ov9ocecnwn29v1l\&st=qst1croi\&dl=1 && \
    curl -L -o models/ordinal_encoder.pkl https://www.dropbox.com/scl/fi/7xvafcp1ifg58cy1j91rb/ordinal_encoder.pkl?rlkey=e348szxwmx1m2iywqx3s911mu\&st=ai7siwl2\&dl=1 && \
    curl -L -o models/scaler.pkl https://www.dropbox.com/scl/fi/5jxlfpf92uhc7nlm4uxvz/scaler.pkl?rlkey=7x3fy88p5hnnnxgwrammont74\&st=4alwa13n\&dl=1 && \
    curl -L -o models/very_late_model.pkl https://www.dropbox.com/scl/fi/4q84fuo0axx6g8ewwrhd8/very_late_model.pkl?rlkey=cbeywkhrrea6hjwphwtcrnoeu\&st=26ywnsv5\&dl=1

# Copy app code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

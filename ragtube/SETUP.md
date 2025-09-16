# RAGTube Setup Guide

This document provides a step-by-step guide to get RAGTube up and running.

## Prerequisites

- Python 3.11 or higher
- Pinecone account and API key
- (Optional) Cohere account for enhanced reranking
- (Optional) YouTube Data API key for channel ingestion

## Quick Setup

### 1. Install Dependencies

```bash
cd ragtube
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp env_example.txt .env

# Edit .env with your API keys
nano .env
```

Required configuration:
```bash
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=ragtube-hybrid
PINECONE_ENV=us-east-1
```

Optional configuration:
```bash
COHERE_API_KEY=your_cohere_api_key_here
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### 3. Verify Setup

```bash
python run.py check-env
```

### 4. Start the Server

```bash
python run.py serve
```

The server will start at `http://localhost:8000` with docs at `http://localhost:8000/docs`.

## First Steps

### 1. Ingest Sample Videos

```bash
# Using specific video IDs
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"video_ids": ["dQw4w9WgXcQ"]}'

# Or using a channel (requires YouTube API key)
curl -X POST "http://localhost:8000/ingest/channel" \
  -H "Content-Type: application/json" \
  -d '{"channel_id": "UCBJycsmduvYEL83R_U4JriQ", "max_videos": 5}'
```

### 2. Test Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"q": "test query", "k": 3}'
```

### 3. Run Evaluation

```bash
# Create sample evaluation set
python run.py create-evalset

# Run evaluation
python run.py eval --evalset evalset.yaml
```

## Development Commands

### Run Tests

```bash
# All tests
python run.py test

# With coverage
python run.py test --coverage

# Verbose output
python run.py test --verbose
```

### Run Evaluation

```bash
# Basic evaluation
python run.py eval

# Comparative evaluation
python run.py eval --comparative

# Custom parameters
python run.py eval --alpha 0.7 --top-k 3 --no-rerank
```

### Server Options

```bash
# Development server (auto-reload)
python run.py serve

# Production-like server
python run.py serve --no-reload --host 0.0.0.0 --port 8080
```

## Troubleshooting

### Common Issues

1. **"No module named 'config'"**
   - Make sure you're running commands from the `ragtube/` directory
   - Check that `__init__.py` files exist in all subdirectories

2. **"Pinecone API key is required"**
   - Verify your `.env` file contains `PINECONE_API_KEY`
   - Check that the API key is valid

3. **"Failed to load dense encoder model"**
   - First run may take time to download models
   - Ensure stable internet connection
   - Check available disk space

4. **"No transcript available for video"**
   - Not all YouTube videos have transcripts
   - Try different video IDs
   - Check video privacy settings

### Logs and Debugging

```bash
# Check server logs
tail -f logs/ragtube.log

# Enable debug logging
export LOG_LEVEL=DEBUG
python run.py serve
```

### Performance Optimization

For better performance:

1. **Use GPU if available:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Increase batch sizes** (in config.py):
   ```python
   BATCH_SIZE = 64  # Increase if you have more memory
   ```

3. **Use Cohere reranking** (faster than local BGE):
   ```bash
   export COHERE_API_KEY=your_key_here
   ```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run.py", "serve", "--no-reload", "--host", "0.0.0.0"]
```

### Environment Variables

```bash
# Production environment
PINECONE_API_KEY=prod_key
PINECONE_INDEX=ragtube-prod
COHERE_API_KEY=prod_cohere_key
LOG_LEVEL=INFO
MAX_WORKERS=4
```

### Health Monitoring

```bash
# Health check endpoint
curl http://localhost:8000/healthz

# Expected response
{
  "status": "healthy",
  "timestamp": 1640995200.0,
  "components": {
    "pinecone": {"available": true},
    "dense_encoder": {"available": true},
    "sparse_encoder": {"available": true},
    "reranker": {"available": true, "type": "cohere"}
  }
}
```

## Next Steps

1. **Index your content**: Start with a small set of videos to test
2. **Customize evaluation**: Create your own evaluation set based on your use case
3. **Tune parameters**: Experiment with different alpha values and chunk sizes
4. **Monitor performance**: Use the evaluation harness to track improvements
5. **Scale up**: Add more videos and optimize for your specific domain

## Support

- Check the README.md for detailed documentation
- Run tests to verify your setup: `python run.py test`
- Use the evaluation harness to benchmark performance
- Monitor logs for errors and performance metrics

Happy querying! 🚀

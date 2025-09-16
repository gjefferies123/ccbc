# 🚀 Deploy Biblical Wisdom App

## Quick Deploy Options

### Option 1: Railway (Recommended)

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy from GitHub**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will automatically detect it's a Python app

3. **Set Environment Variables**
   - Go to your project → Variables tab
   - Add these variables:
     ```
     PINECONE_API_KEY=your_pinecone_key
     COHERE_API_KEY=your_cohere_key
     PINECONE_INDEX=christ-chapel-sermons
     FLASK_ENV=production
     ```

4. **Deploy**
   - Railway will automatically build and deploy
   - Get your live URL!

### Option 2: Render

1. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Create Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repo
   - Choose "Python" as environment

3. **Configure Service**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python final_wisdom_web_app.py`
   - **Python Version**: 3.11

4. **Set Environment Variables**
   - Go to Environment tab
   - Add the same variables as Railway

5. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy automatically

### Option 3: Heroku

1. **Install Heroku CLI**
   - Download from [heroku.com](https://heroku.com)

2. **Login and Create App**
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set PINECONE_API_KEY=your_key
   heroku config:set COHERE_API_KEY=your_key
   heroku config:set PINECONE_INDEX=christ-chapel-sermons
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

## Environment Variables Required

```
PINECONE_API_KEY=your_pinecone_api_key
COHERE_API_KEY=your_cohere_api_key  
PINECONE_INDEX=christ-chapel-sermons
FLASK_ENV=production
```

## Files Included for Deployment

- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Heroku process file
- ✅ `runtime.txt` - Python version
- ✅ `final_wisdom_web_app.py` - Main app
- ✅ `enhanced_wisdom_app.py` - Enhanced logic
- ✅ `christ_chapel_search.py` - Search engine
- ✅ `templates/wisdom_app.html` - Web interface

## Post-Deployment

1. **Test the live site**
2. **Check environment variables** are set correctly
3. **Monitor logs** for any errors
4. **Set up custom domain** (optional)

## Troubleshooting

- **Import errors**: Check all dependencies in requirements.txt
- **Environment variables**: Verify all are set correctly
- **Pinecone connection**: Check API key and index name
- **Cohere API**: Verify API key is valid

## Cost

- **Railway**: Free tier (500 hours/month)
- **Render**: Free tier (750 hours/month)  
- **Heroku**: Free tier (limited)

All options have generous free tiers for personal use!

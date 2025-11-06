# Deploying to Render

This guide will help you deploy your Fraud Detection Flask app to Render.

## Prerequisites

1. A Render account (sign up at [render.com](https://render.com))
2. Your code pushed to GitHub, GitLab, or Bitbucket

## Deployment Steps

### Option 1: Using render.yaml (Recommended)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Add Render configuration"
   git push origin main
   ```

2. **Deploy on Render**
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`
   - Click "Apply" to deploy

### Option 2: Manual Setup

1. **Go to Render Dashboard**
   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" → "Web Service"

2. **Connect Repository**
   - Connect your GitHub/GitLab/Bitbucket account
   - Select your repository: `-AI-Driven-Financial-Fraud-Anomaly-Detection-Platform`

3. **Configure Settings**
   - **Name**: `fraud-detection-app` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
   - **Plan**: Free (or choose a paid plan for better performance)

4. **Environment Variables**
   - Click "Advanced" → "Add Environment Variable"
   - Add `SECRET_KEY` with a secure random string (or let Render generate it)
   - Add `RENDER=true` (optional, helps with path detection)

5. **Deploy**
   - Click "Create Web Service"
   - Wait for the build to complete (5-10 minutes for first build)

## Important Notes

### Database Storage
- SQLite database is stored in the project directory on Render
- **Free tier**: Database persists but may be slower
- **For production**: Consider upgrading to Render Postgres (paid)

### Build Time
- First build takes 5-10 minutes (PyTorch is large ~500MB)
- Subsequent builds are faster due to caching
- Free tier has build time limits

### Model Files
Make sure these files are in your repository:
- `fraud_model_state.pth`
- `scaler.pkl`

### Port Configuration
- Render automatically sets the `$PORT` environment variable
- The app uses `gunicorn` to bind to this port
- No manual port configuration needed

## Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

### App Doesn't Start
- Check runtime logs in Render dashboard
- Verify `gunicorn` is in `requirements.txt`
- Ensure `Procfile` or start command is correct

### Database Issues
- Check that database path is correct
- Verify write permissions
- Consider using Render Postgres for production

### Memory Issues
- Free tier has 512MB RAM limit
- PyTorch may use significant memory
- Consider upgrading to paid plan if needed

## Updating Your App

1. Make changes to your code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update app"
   git push
   ```
3. Render automatically redeploys on push

## Environment Variables

Set these in Render Dashboard → Environment:

- `SECRET_KEY`: Flask secret key (auto-generated if using render.yaml)
- `RENDER`: Set to `true` (optional, for path detection)
- `PYTHON_VERSION`: `3.11.0` (specified in render.yaml)

## Free Tier Limitations

- 750 hours/month (enough for always-on)
- 512MB RAM
- Slower cold starts
- Build time limits

For production use, consider upgrading to a paid plan.

## Support

- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com)


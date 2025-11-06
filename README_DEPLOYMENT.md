# Quick Deployment Guide

## Deploy to Vercel

### Method 1: Using Vercel Dashboard (Easiest)

1. **Prepare your repository**
   - Make sure all files are committed to Git
   - Push to GitHub, GitLab, or Bitbucket

2. **Deploy on Vercel**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import your Git repository
   - Vercel will auto-detect the configuration
   - Click "Deploy"

3. **Wait for deployment**
   - Build typically takes 2-5 minutes
   - You'll get a URL when complete

### Method 2: Using Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel

# For production
vercel --prod
```

## Important Notes

⚠️ **File Size Warning**: PyTorch is large (~500MB). If deployment fails due to size:
- Consider using a lighter ML framework
- Or use external model hosting (AWS S3, Cloudflare R2)
- Or switch to a platform with larger limits (Railway, Render)

⚠️ **Database Warning**: SQLite in `/tmp` is **ephemeral** (resets on each deployment)
- For production, use Vercel Postgres or external database
- User accounts will be lost on each deployment

## Troubleshooting

**Build fails?**
- Check build logs in Vercel dashboard
- Ensure all files are committed
- Verify `requirements.txt` is correct

**App doesn't work?**
- Check function logs in Vercel dashboard
- Verify model files (`fraud_model_state.pth`, `scaler.pkl`) are included
- Check environment variables

## Alternative Platforms

If Vercel doesn't work due to size limits, consider:
- **Railway** - Better for Python apps with large dependencies
- **Render** - Good free tier, supports Flask
- **Fly.io** - Great for Python applications
- **Heroku** - Classic option (paid)


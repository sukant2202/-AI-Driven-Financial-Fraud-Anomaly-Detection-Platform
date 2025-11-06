# Deploying to Vercel

This guide will help you deploy your Fraud Detection Flask app to Vercel.

## Prerequisites

1. A Vercel account (sign up at [vercel.com](https://vercel.com))
2. Vercel CLI installed (optional, for CLI deployment)
3. Git repository (recommended)

## Deployment Steps

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to GitHub/GitLab/Bitbucket**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Import Project in Vercel**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import your Git repository
   - Vercel will automatically detect the `vercel.json` configuration

3. **Configure Environment Variables** (if needed)
   - Go to Project Settings → Environment Variables
   - Add `SECRET_KEY` with a secure random string (optional, defaults provided)

4. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```
   
   For production deployment:
   ```bash
   vercel --prod
   ```

## Important Notes

### Database Storage
- The SQLite database is stored in `/tmp` on Vercel
- **Important**: `/tmp` is ephemeral and resets on each deployment
- For production, consider using:
  - Vercel Postgres (recommended)
  - External database service (MongoDB Atlas, Supabase, etc.)
  - Vercel KV for session storage

### File Size Limits
- Vercel has a 50MB limit for serverless functions
- Your model files (`fraud_model_state.pth`, `scaler.pkl`) must be under this limit
- If files are too large, consider:
  - Using Vercel Blob Storage
  - Hosting model files on external storage (S3, Cloudflare R2)
  - Loading models from external URLs

### Model Files
Make sure these files are included in your deployment:
- `fraud_model_state.pth`
- `scaler.pkl`

They should be in the root directory alongside `app.py`.

## Troubleshooting

### Build Errors
- Check that all dependencies are in `requirements.txt`
- Ensure Python version is compatible (Vercel uses Python 3.9+)
- Check build logs in Vercel dashboard

### Runtime Errors
- Check function logs in Vercel dashboard
- Verify model files are present and accessible
- Ensure database path is correct (`/tmp/users.db`)

### Database Issues
- Remember that `/tmp` is ephemeral
- Consider migrating to a persistent database for production

## Updating Requirements

If you need to update dependencies, edit `requirements.txt` and redeploy:

```bash
git add requirements.txt
git commit -m "Update dependencies"
git push
```

Vercel will automatically rebuild on push.

## Environment Variables

Set these in Vercel Dashboard → Settings → Environment Variables:

- `SECRET_KEY`: Flask secret key (optional, has default)

## Support

For issues specific to Vercel deployment, check:
- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Python Runtime](https://vercel.com/docs/runtimes#official-runtimes/python)


# DermAI — Complete Setup Guide (Windows)
### From Zero → Localhost → Deployed Live URL

---

## 📁 Project Structure

```
dermai/
├── frontend/          ← React + Vite (what users see)
│   ├── src/
│   │   ├── App.jsx         main app shell
│   │   ├── api.js          Gemini + Supabase + ML backend calls
│   │   ├── supabase.js     Supabase client
│   │   ├── Phase1.jsx      AI Engine results
│   │   ├── Phase2.jsx      Virtual Biopsy results
│   │   ├── Phase3.jsx      Digital Twin results
│   │   └── History.jsx     Past sessions viewer
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example   ← copy to .env and fill in keys
│
├── backend/           ← FastAPI Python (ML model server)
│   ├── server.py           EfficientNet inference API
│   ├── requirements.txt
│   └── models/        ← put .pth file here after training
│
├── database/
│   └── schema.sql     ← run once in Supabase SQL editor
│
└── SETUP.md           ← you are here
```

---

## STAGE 1 — Install Prerequisites (Do Once)

### 1A. Install Node.js
1. Go to https://nodejs.org
2. Download **LTS version** (the left green button)
3. Run the installer — click Next through everything
4. Open a new terminal and verify:
   ```
   node --version     ← should show v18 or higher
   npm --version      ← should show a number
   ```

### 1B. Install Python
1. Go to https://python.org/downloads
2. Download **Python 3.11** (recommended)
3. ⚠️ IMPORTANT: On the first installer screen, check **"Add Python to PATH"**
4. Click Install Now
5. Verify:
   ```
   python --version   ← should show 3.11.x
   pip --version      ← should show a number
   ```

### 1C. Install Git (if not already)
1. Go to https://git-scm.com/download/win
2. Download and install — all defaults are fine
3. Verify:
   ```
   git --version
   ```

---

## STAGE 2 — Get Your API Keys

### 2A. Gemini API Key (FREE)
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key — looks like: `AIzaSy...`

### 2B. Supabase (FREE)
1. Go to https://supabase.com → Sign Up (free)
2. Click **"New Project"**
3. Give it a name: `dermai-db`
4. Set a database password (save it somewhere)
5. Choose a region close to you
6. Wait ~2 minutes for it to create
7. Go to **Settings → API**
8. Copy two values:
   - **Project URL** → looks like `https://abcxyz.supabase.co`
   - **anon public key** → long string starting with `eyJ...`

### 2C. Set Up the Database
1. In your Supabase project, click **"SQL Editor"** in the left sidebar
2. Click **"New query"**
3. Open the file `database/schema.sql` from this project
4. Copy everything in that file
5. Paste it into the SQL editor
6. Click **"Run"**
7. You should see "Success" — your tables are created ✅

---

## STAGE 3 — Run the Frontend Locally

### 3A. Create your .env file
1. Open the `frontend/` folder
2. Find the file `.env.example`
3. Make a copy of it and rename the copy to `.env`
4. Open `.env` and fill in your keys:
   ```
   VITE_GEMINI_API_KEY=AIzaSy...your key here...
   VITE_SUPABASE_URL=https://your-project.supabase.co
   VITE_SUPABASE_ANON_KEY=eyJ...your key here...
   VITE_ML_BACKEND_URL=http://localhost:8000
   ```

### 3B. Install and run
Open a terminal, navigate to the frontend folder:
```bash
cd path\to\dermai\frontend

npm install        ← downloads all dependencies (~1 min)
npm run dev        ← starts the dev server
```

You should see:
```
  VITE v5.x.x  ready in 500ms
  ➜  Local:   http://localhost:5173/
```

5. Open your browser and go to: **http://localhost:5173**
6. You should see the DermAI app! 🎉

### 3C. Test it without the ML backend
- Fill in age + gender (required)
- Optionally upload a skin lesion image
- Click **Run Full Analysis**
- All 3 phases should run using Gemini AI
- Results auto-save to Supabase

---

## STAGE 4 — Run the Backend Locally

> Do this after the frontend is working. The backend is optional
> until you train your EfficientNet model.

### 4A. Create a virtual environment (good practice)
```bash
cd path\to\dermai\backend

python -m venv venv
venv\Scripts\activate       ← you'll see (venv) in your terminal
```

### 4B. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ PyTorch is large (~2GB). This will take 5-10 minutes.

### 4C. Start the backend
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
⚠️  Model file not found — running in DEMO mode
```

Demo mode means the backend works but returns simulated scores.
Real scores come after you train the EfficientNet model (Stage 6).

### 4D. Test the backend
Open your browser: **http://localhost:8000/health**
Should show: `{"status":"ok","model_loaded":false,"demo_mode":true}`

---

## STAGE 5 — Train the EfficientNet Model (Google Colab)

> You need a Google account for this.

1. Go to https://colab.research.google.com
2. Upload the file `DermAI_EfficientNet_Training.ipynb`
   (from the `dermai-ml-pipeline.zip` you received earlier)
3. Click **Runtime → Change runtime type → T4 GPU → Save**
4. Get your Kaggle API key:
   - Go to https://kaggle.com → Account → API → Create New Token
   - It downloads `kaggle.json` — open it, copy username and key
5. In the notebook Step 2, paste your Kaggle username and key
6. Click **Runtime → Run All**
7. Training takes ~45-60 minutes (runs itself, you can close the tab)
8. At the end, it auto-downloads two files:
   - `dermai_model_full.pth`
   - `model_meta.json`
9. Copy both files into: `backend/models/`
10. Restart the backend — it will now load the real model ✅

---

## STAGE 6 — Deploy to the Web

### 6A. Push to GitHub
```bash
cd path\to\dermai

git init
git add .
git commit -m "DermAI full stack"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dermai.git
git push -u origin main
```

### 6B. Deploy Frontend → Netlify (free)
1. Go to https://netlify.com → Sign up with GitHub
2. Click **Add new site → Import an existing project → GitHub**
3. Select your `dermai` repo
4. Set build settings:
   - **Base directory:** `frontend`
   - **Build command:** `npm run build`
   - **Publish directory:** `frontend/dist`
5. Click **Deploy site**
6. Go to **Site settings → Environment variables** → Add:
   ```
   VITE_GEMINI_API_KEY        = your key
   VITE_SUPABASE_URL          = your supabase url
   VITE_SUPABASE_ANON_KEY     = your anon key
   VITE_ML_BACKEND_URL        = https://your-railway-url.up.railway.app
   ```
7. **Deploys → Trigger deploy** → Your site is live! 🚀

### 6C. Deploy Backend → Railway (free)
1. Go to https://railway.app → Sign up with GitHub
2. Click **New Project → Deploy from GitHub repo**
3. Select your `dermai` repo
4. Click **Add service → select the repo**
5. Set the root directory to `backend`
6. Set start command:
   ```
   uvicorn server:app --host 0.0.0.0 --port $PORT
   ```
7. Your backend gets a URL like: `https://dermai-backend.up.railway.app`
8. Copy this URL into Netlify's `VITE_ML_BACKEND_URL` env variable
9. Redeploy Netlify — everything is connected ✅

---

## ✅ You're Done!

Your live app URL from Netlify is what you share with professors and teammates.

```
Frontend (Netlify) ──► https://dermai-app.netlify.app
Backend (Railway)  ──► https://dermai-backend.up.railway.app
Database (Supabase)──► auto-connected via env keys
```

---

## ❓ Common Issues

| Problem | Fix |
|---|---|
| `npm: command not found` | Restart terminal after installing Node.js |
| `python: command not found` | Reinstall Python and check "Add to PATH" |
| `MISSING_GEMINI_KEY` error | Check your `.env` file has no spaces around `=` |
| Gemini returns 429 error | Free tier rate limit hit, wait 1 minute |
| Backend shows demo mode | Normal until you add the `.pth` model file |
| Supabase save fails | Check your anon key and URL in `.env` |

---

## 📋 Continuation Prompt (for new chat)

If you start a new chat with Claude, paste this:

```
I'm building DermAI — an AI skin cancer diagnostic platform.
3-phase system: Phase 1 = EfficientNet-B4 on HAM10000
(real probability scores) → Phase 2 = Gemini predicts DNA
mutations → Phase 3 = Digital Twin simulates drug responses.

Stack: React+Vite frontend, FastAPI backend, Gemini API (free),
Supabase (database + image storage), Railway (backend deploy),
Netlify (frontend deploy).

OS: Windows. New to ML. GitHub account only.
All code already written. I am on [DESCRIBE YOUR CURRENT STEP].
Please continue guiding me step by step.
```

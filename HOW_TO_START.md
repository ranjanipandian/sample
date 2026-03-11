# 🚀 HOW TO START THE APPLICATION

## The "Backend server not running" message means you need to start the backend!

Your application has TWO parts that must BOTH be running:
1. **Backend (Python)** - Port 8000 - Handles searches
2. **Frontend (Next.js)** - Port 3000 - Shows UI

## ⚡ EASIEST METHOD - Use the Startup Scripts

I've created simple scripts for you:

### Step 1: Start Backend (First!)
Double-click: **`START_BACKEND.bat`**

Wait until you see:
```
🚀 Starting Research Intelligence Search API Server
Host: 0.0.0.0
Port: 8000
```

**KEEP THIS WINDOW OPEN!**

### Step 2: Start Frontend (Second!)
Double-click: **`START_FRONTEND.bat`**

Wait until you see:
```
▲ Next.js 16.0.7
- Local: http://localhost:3000
```

**KEEP THIS WINDOW OPEN TOO!**

### Step 3: Open Browser
Go to: http://localhost:3000

### Step 4: Test Search
Enter "brain tumor" - should work now!

---

## 🔧 Alternative: Use QUICK_FIX.bat

Double-click: **`QUICK_FIX.bat`**

This starts both services automatically and opens the browser.

---

## ⚠️ IMPORTANT: Before First Run

Make sure `backend/.env` has your credentials:

```env
DB_PASSWORD=your_actual_postgres_password
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_actual_key
```

---

## ✅ How to Verify Backend is Running

Open in browser: http://localhost:8000/docs

You should see the FastAPI Swagger documentation page.

If you see this, the backend is running correctly!

---

## 🐛 Troubleshooting

### "Backend server not running" error?
- The backend is not started
- Solution: Run `START_BACKEND.bat` first

### Port 8000 already in use?
- Another program is using port 8000
- Solution: Close other programs or restart computer

### "Module not found" error?
- Dependencies not installed
- Solution: 
  ```bash
  cd unified-research-nextjs/backend
  venv\Scripts\activate
  pip install -r requirements.txt
  ```

### Database connection error?
- PostgreSQL not running or wrong password
- Solution: Check `backend/.env` credentials

---

## 📝 Quick Reference

**Backend:** http://localhost:8000/docs
**Frontend:** http://localhost:3000
**Health Check:** http://localhost:8000/api/health

**Start Backend:** Double-click `START_BACKEND.bat`
**Start Frontend:** Double-click `START_FRONTEND.bat`
**Start Both:** Double-click `QUICK_FIX.bat`

---

## 🎯 Summary

1. ✅ Start backend first (`START_BACKEND.bat`)
2. ✅ Start frontend second (`START_FRONTEND.bat`)
3. ✅ Keep both windows open
4. ✅ Open http://localhost:3000
5. ✅ Search should work!

**Remember: BOTH services must be running at the same time!**

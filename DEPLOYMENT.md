# 🚀 Deploying Scribble AI

Follow these steps to make your Scribble Gesture Game live for everyone to play!

## 1. Choose a Hosting Platform
We recommend **Render** or **Railway** for easy Flask deployment.

### Option A: Render (Recommended)
1. **Create a GitHub Repository**: Push your code to a new repository on GitHub.
2. **Sign in to Render**: Go to [render.com](https://render.com) and connect your GitHub account.
3. **New Web Service**: Select "New" > "Web Service".
4. **Connect Repo**: Select your `scribble-game` repository.
5. **Configure**:
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --worker-class eventlet -w 1 server:app`
6. **Deploy**: Click "Create Web Service".

## 2. Share the Link
Once deployed, Render will give you a URL like `https://scribble-ai.onrender.com`. 
- Share this link with your friend!
- One person joins as **Drawer**, the other as **Guesser**.

## 3. How to Play (Live)
- **Drawer**: You can now draw **directly in the browser**! Click the **"Enable Web Drawing 📷"** button in the top right corner. No Python script required!
- **Guesser**: Just type your guesses in the text box.

---

## (Advanced) Using the Python Client Live
If you still want to use the high-performance Python script (`hand_gesture_draw.py`) with your live server:
1. Find your live URL (e.g., `https://scribble-ai.onrender.com`).
2. Run the script pointing to that URL:
   ```bash
   python hand_gesture_draw.py --server https://scribble-ai.onrender.com --name YourName
   ```

## Troubleshooting
- **Webcam not working?** Ensure you've granted camera permissions in your browser.
- **Latency?** Real-time games depend on your internet speed. Close background tabs for better performance.

# Local Whisper AI Setup Guide

## Overview
This setup allows you to run Whisper AI locally on your computer, eliminating API costs and rate limits. Your React Native app will connect to a local server running on your machine.

## Prerequisites
- Python 3.8 or higher
- At least 4GB RAM
- Internet connection for initial model download

## Setup Instructions

### Step 1: Install Python Dependencies
Open a terminal in your project folder and run:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install fastapi uvicorn openai-whisper python-multipart
```

### Step 2: Start the Whisper Server
Run the server script:

```bash
python whisper_server.py
```

You should see:
```
🎤 Starting Local Whisper Server...
📝 Install requirements: pip install fastapi uvicorn openai-whisper python-multipart
🌐 Server will be available at: http://localhost:8000
📱 Use this URL in your React Native app
⚡ Press Ctrl+C to stop the server
```

### Step 3: Configure Your React Native App
1. Start your React Native app
2. In the "Whisper Server URL" field, enter: `http://localhost:8000`
3. Tap "Test Connection" - you should see "✅ Connected"
4. Select your language and start recording!

## Model Options
The server uses the "base" model by default (good balance of speed and accuracy). You can modify `whisper_server.py` to use different models:

- `tiny` - Fastest, least accurate (~39 MB)
- `base` - Good balance (~74 MB) **[Default]**
- `small` - Better accuracy (~244 MB)
- `medium` - High accuracy (~769 MB)
- `large` - Best accuracy (~1550 MB)

## Supported Languages
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)
- Arabic (ar)
- Hindi (hi)
- Tamil (ta)
- Auto-detect (auto) **[Recommended]**

## Network Setup
- **Same Device**: Use `http://localhost:8000`
- **Different Device on Same Network**: Use `http://YOUR_COMPUTER_IP:8000`
  - Find your IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
  - Example: `http://192.168.1.100:8000`

## Troubleshooting

### Server Won't Start
- Check if Python 3.8+ is installed: `python --version`
- Install missing packages: `pip install -r requirements.txt`
- Port 8000 in use? Change port in `whisper_server.py`: `uvicorn.run(app, port=8001)`

### React Native Can't Connect
- Make sure server is running (you should see the startup messages)
- Check the URL format: `http://localhost:8000` (not `https://`)
- If testing on device, use your computer's IP address instead of `localhost`
- Disable firewall temporarily to test connection

### Poor Audio Quality
- Ensure your device has microphone permissions
- Try recording in a quiet environment
- Use a higher quality model (small, medium, or large)

### Slow Transcription
- Use a smaller model (tiny or base)
- Ensure your computer has sufficient RAM
- Close other CPU-intensive applications

## Performance Tips
1. **First Run**: Model downloads automatically (~74MB for base model)
2. **Subsequent Runs**: Models are cached, startup is much faster
3. **RAM Usage**: ~1-2GB for base model, more for larger models
4. **CPU Usage**: Higher during transcription, minimal when idle

## Security Notes
- Server runs on local network only
- No data is sent to external services
- Audio files are processed locally and deleted immediately
- For production use, configure proper CORS origins

## Benefits of Local Setup
✅ **No API costs** - Completely free after setup  
✅ **No rate limits** - Process unlimited audio  
✅ **Privacy** - Audio never leaves your device  
✅ **Offline capable** - Works without internet (after model download)  
✅ **Fast** - No network latency for requests  
✅ **Reliable** - No external service dependencies  

Happy transcribing! 🎤➡️📝
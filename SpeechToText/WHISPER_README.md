# Multi-Language Speech-to-Text with Whisper AI

A React Native app that uses OpenAI's Whisper API for accurate speech recognition in multiple languages.

## Features

- **High-accuracy transcription** using OpenAI's Whisper AI
- **Multi-language support** (English, Hindi, Tamil, French, German, Spanish, Auto-detect)
- **Real-time recording** with visual feedback
- **Secure API key management**
- **Cross-platform** (Android/iOS)

## Setup Instructions

### 1. Install Dependencies
```bash
npm install
```

### 2. Get OpenAI API Key
1. Go to [OpenAI API Keys](https://platform.openai.com/account/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the generated key

### 3. Configure API Key
1. Open `config.js` in the project root
2. Replace `'your-openai-api-key-here'` with your actual API key:
```javascript
export const config = {
  OPENAI_API_KEY: 'sk-your-actual-api-key-here',
};
```

### 4. Build and Run
```bash
# For Android
npx react-native run-android

# For iOS
npx react-native run-ios
```

## How It Works

1. **Record Audio**: Tap "Start Recording" to begin capturing speech
2. **Stop & Process**: Tap "Stop Recording" to end capture and send to Whisper API
3. **Get Results**: The transcribed text appears in the results box

## Permissions

The app requires microphone permission to record audio. This will be requested automatically on first use.

## API Costs

- Whisper API pricing: $0.006 per minute of audio
- First $5 in usage is free for new OpenAI accounts
- See [OpenAI Pricing](https://openai.com/pricing) for current rates

## Supported Languages

- English
- Hindi
- Tamil
- French
- German
- Spanish
- Auto-detect (recommended for best accuracy)

## Technical Details

- **Audio Format**: WAV, 16kHz, mono
- **API**: OpenAI Whisper v1
- **Recording**: React Native Audio Record
- **File System**: React Native FS

## Troubleshooting

### "API Key Required" Error
- Make sure you've set your OpenAI API key correctly in `config.js`
- Verify the key is valid and has sufficient credits

### "Permission Denied" Error
- Grant microphone permission when prompted
- Check app permissions in device settings if needed

### Network Errors
- Ensure stable internet connection
- Check if OpenAI API is accessible from your network

## Security Note

- `config.js` is added to `.gitignore` to keep your API key secure
- Never commit API keys to version control
- Consider using environment variables for production apps

## Dependencies

- `@react-native-picker/picker` - Language selection
- `react-native-audio-record` - Audio recording
- `react-native-fs` - File system access
- `react-native-safe-area-context` - Safe area handling

## License

This project is for educational purposes. Please respect OpenAI's usage policies and terms of service.
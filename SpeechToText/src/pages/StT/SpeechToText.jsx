import React, { useState, useEffect } from 'react';
import { View, Text, Button, ScrollView, StyleSheet, Alert, PermissionsAndroid, Platform, TextInput } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import AudioRecord from 'react-native-audio-record';
import RNFS from 'react-native-fs';
import styles from './StTsty';
import {API_URL} from '../../util/env';

const SpeechToText = () => {
    const [recognizedText, setRecognizedText] = useState('');
    const [isListening, setIsListening] = useState(false);
    const [selectedLang, setSelectedLang] = useState('en');
    const [isProcessing, setIsProcessing] = useState(false);
    const [serverUrl, setServerUrl] = useState(API_URL);
    const [serverReady, setServerReady] = useState(false);
    const [checkingServer, setCheckingServer] = useState(false);

    const languages = [
        { label: 'English', code: 'en' },
        // { label: 'Hindi', code: 'hi' },
        { label: 'Tamil', code: 'ta' },
        // { label: 'French', code: 'fr' },
        // { label: 'German', code: 'de' },
        // { label: 'Spanish', code: 'es' },
        // { label: 'Auto-detect', code: 'auto' },
    ];

    useEffect(() => {
        const options = {
            sampleRate: 16000,
            channels: 1,
            bitsPerSample: 16,
            audioSource: 6, // VOICE_RECOGNITION
            wavFile: 'audio.wav',
        };
        AudioRecord.init(options);
        requestMicrophonePermission();
        checkWhisperServer();
        return () => {
            AudioRecord.stop();
        };
    }, []);

    const checkWhisperServer = async () => {
        try {
            setCheckingServer(true);
            setRecognizedText('Checking local Whisper server...');

            // Use AbortController instead of unsupported fetch timeout
            const controller = new AbortController();
            const id = setTimeout(() => controller.abort(), 80000); // 25s for health
            const response = await fetch(`${serverUrl}/health`, {
                method: 'GET',
                signal: controller.signal,
            });
            clearTimeout(id);

            if (response.ok) {
                setServerReady(true);
                setRecognizedText('Local Whisper server is ready! You can start recording.');
            } else {
                throw new Error('Server not healthy');
            }
        } catch (error) {
            console.error('Whisper server check failed:', error);
            setServerReady(false);
            setRecognizedText('Local Whisper server not found. Please start the server or check the URL.');
        } finally {
            setCheckingServer(false);
        }
    };

    const requestMicrophonePermission = async () => {
        if (Platform.OS === 'android') {
            try {
                const granted = await PermissionsAndroid.request(
                    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
                    {
                        title: 'Microphone Permission',
                        message: 'This app needs access to your microphone to record speech for transcription.',
                        buttonNeutral: 'Ask Me Later',
                        buttonNegative: 'Cancel',
                        buttonPositive: 'OK',
                    }
                );
                if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
                    Alert.alert('Permission Denied', 'Microphone permission is required for speech recognition.');
                }
            } catch (err) {
                console.warn(err);
            }
        }
    };

    const transcribeWithLocalWhisper = async (audioFilePath) => {
        if (!serverReady) {
            Alert.alert('Server Not Ready', 'Local Whisper server is not running. Please start the server.');
            return;
        }

        try {
            setIsProcessing(true);
            setRecognizedText('Processing with local Whisper AI...');

            // DO NOT read as base64 for large files (memory heavy) — remove the RNFS.readFile call

            // Build multipart form data with file URI
            const formData = new FormData();
            formData.append('file', {
                uri: `file://${audioFilePath}`,
                type: 'audio/wav',
                name: 'audio.wav',
            });
            formData.append('language', selectedLang || 'auto');

            // Don’t set Content-Type: multipart/form-data manually (boundary issues).
            // Use AbortController for a real (long) timeout.
            const controller = new AbortController();
            const id = setTimeout(() => controller.abort(), 300000); // 5 minutes
            const response = await fetch(`${serverUrl}/transcribe`, {
                method: 'POST',
                body: formData,
                signal: controller.signal,
            });
            clearTimeout(id);

            if (!response.ok) {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            if (result && result.text) {
                setRecognizedText(result.text.trim() || 'No speech detected in the audio.');
            } else {
                setRecognizedText('No speech detected in the audio.');
            }
        } catch (error) {
            console.error('Local Whisper error:', error);
            if (error.name === 'AbortError') {
                setRecognizedText('Request timed out. Try shorter segments or better Wi‑Fi.');
            } else if ((error.message || '').includes('Network')) {
                setRecognizedText('Network error. Check if the Whisper server is running and reachable.');
            } else {
                setRecognizedText(`Error: ${error.message || 'Failed to transcribe audio'}`);
            }
        } finally {
            setIsProcessing(false);
        }
    };

    const startListening = async () => {
        try {
            setIsListening(true);
            setRecognizedText('Recording...');
            AudioRecord.start();

            // Optional: auto-stop long recordings to avoid giant uploads (e.g., 25s chunks)
            // setTimeout(() => { if (isListening) stopListening(); }, 25000);

        } catch (error) {
            console.error('Error starting recording:', error);
            setIsListening(false);
            setRecognizedText('Error starting recording');
        }
    };

    const stopListening = async () => {
        try {
            const audioFile = await AudioRecord.stop();
            setIsListening(false);

            if (audioFile && serverReady) {
                await transcribeWithLocalWhisper(audioFile);
            } else if (!serverReady) {
                setRecognizedText('Whisper server is not ready. Please check the server URL and status.');
            }
        } catch (error) {
            console.error('Error stopping recording:', error);
            setIsListening(false);
            setRecognizedText('Error stopping recording');
        }
    };
    return (
        <ScrollView contentContainerStyle={styles.container}>
            <Text style={styles.heading}>Multi-Language Speech-to-Text</Text>

            <Text style={styles.label}>Server URL:</Text>
            <View style={styles.inputContainer}>
                <Text>{serverUrl}</Text>
                <Button
                    title="Test"
                    onPress={checkWhisperServer}
                    disabled={checkingServer}
                />
            </View>

            <Text style={styles.label}>Select Language:</Text>
            <View style={styles.pickerContainer}>
                <Picker
                    selectedValue={selectedLang}
                    onValueChange={(itemValue) => setSelectedLang(itemValue)}
                    style={styles.picker}>
                    {languages.map((lang) => (
                        <Picker.Item key={lang.code} label={lang.label} value={lang.code} />
                    ))}
                </Picker>
            </View>

            <View style={styles.buttonContainer}>
                <Button
                    title={isListening ? 'Stop Recording' : 'Start Recording'}
                    onPress={isListening ? stopListening : startListening}
                    disabled={isProcessing || checkingServer || !serverReady}
                />
            </View>

            {checkingServer && (
                <View style={styles.loadingContainer}>
                    <Text style={styles.loadingText}>
                        Checking Whisper server connection...
                    </Text>
                </View>
            )}

            {recognizedText && !isListening && !isProcessing && (
                <View style={styles.buttonRow}>
                    <View style={styles.buttonHalf}>
                        <Button
                            title="Clear Text"
                            onPress={() => setRecognizedText('')}
                            color="#ff6b6b"
                        />
                    </View>
                    <View style={styles.buttonHalf}>
                        <Button
                            title="Check Server"
                            onPress={checkWhisperServer}
                            color="#28a745"
                        />
                    </View>
                </View>
            )}

            {isProcessing && (
                <View style={styles.processingContainer}>
                    <Text style={styles.processingText}>
                        {recognizedText.includes('Retrying') ? recognizedText : 'Processing with Whisper AI...'}
                    </Text>
                </View>
            )}

            <Text style={styles.resultLabel}>Transcribed Text:</Text>
            <View style={styles.resultBox}>
                <Text style={styles.resultText}>
                    {recognizedText || 'Your speech will appear here...'}
                </Text>
            </View>

            {!serverReady && !checkingServer && (
                <View style={styles.warningContainer}>
                    <Text style={styles.warningText}>
                        ⚠️ Server Not Connected{'\n'}
                        • Local Whisper server not running{'\n'}
                        • Check if server is started{'\n'}
                        • Verify the server URL is correct{'\n'}
                        • Tap "Check Server" to test connection
                    </Text>
                </View>
            )}

            <View style={styles.infoContainer}>
                <Text style={styles.infoText}>
                    🎯 Local Whisper Server: {'\n'}
                    • No API costs or rate limits{'\n'}
                    • Runs on your local machine{'\n'}
                    • Use Auto-detect for best accuracy{'\n'}
                    • Speak clearly for 5-30 seconds{'\n'}
                    • Server status: {serverReady ? '✅ Connected' : '❌ Disconnected'}
                </Text>
            </View>
        </ScrollView>
    )
}

// const SpeechToText = () =>{
//     return(
//         <View>
//             <Text>Hi</Text>
//         </View>
//     )
// }

export default SpeechToText;
import React, { useState, useEffect } from 'react';
import { View, Text, Button, ScrollView, StyleSheet, Alert, PermissionsAndroid, Platform, TextInput, Touchable, TouchableOpacity } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import AudioRecord from 'react-native-audio-record';
import RNFS from 'react-native-fs';
import styles from './StTsty';
import {API_URL} from '../../util/env';
import { useNavigation } from '@react-navigation/native';

const SpeechToText = () => {
    const navigation = useNavigation();
    const [recognizedText, setRecognizedText] = useState('');
    const [translationMode, setTranslationMode] = useState('none'); // none | literal | high_level
    const [runNlp, setRunNlp] = useState(true);
    const [nlpResult, setNlpResult] = useState(null);
    const [commandText, setCommandText] = useState('');
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
            formData.append('translate_if_tamil', 'true'); // always get English if Tamil
            formData.append('translation_mode', translationMode);
            formData.append('run_nlp', runNlp ? 'true' : 'false');

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
            if (result) {
                // Choose display text: prefer high-level, then literal, then whisper translation, then original
                const display = result.high_level_translation || result.literal_translation || result.translation_text || result.text;
                setRecognizedText(display ? display.trim() : 'No speech detected in the audio.');
                setNlpResult(result.nlp || null);
                setCommandText(result.command_text || '');
                console.log('Transcription result:', result);
                if (result?.nlp?.intent === 'playback') {
                    const camera = result.nlp?.entities?.camera;
                    if (result.playback_url) {
                        navigation.navigate('CCTV', { playbackUrl: result.playback_url, camera, alternates: result.playback_alternates || [] });
                    } else {
                        // Fallback: if we have start/end times and maybe assumed today but no URL (older server build)
                        const ents = result?.nlp?.entities || {};
                        if (ents.start_time && ents.end_time) {
                            const assumedDate = ents.date || new Date().toISOString().substring(0,10);
                            // Build minimal local URL guess (will work if env matches)
                            const cam = camera || '1';
                            const toCompact = (t) => {
                                const parts = t.split(':');
                                const h = parts[0].padStart(2,'0');
                                const m = (parts[1]||'00').padStart(2,'0');
                                const s = (parts[2]||'00').padStart(2,'0');
                                return h+m+s;
                            };
                            const baseDay = assumedDate.replace(/-/g,'');
                            const startCompact = baseDay + 'T' + toCompact(ents.start_time);
                            const endCompact = baseDay + 'T' + toCompact(ents.end_time);
                            const camChannel = cam.toString().padStart(2,'0') + '01';
                            // Use defaults mirroring backend env assumptions
                            const fallbackHost = '192.168.1.64';
                            const fallbackUser = 'admin';
                            const fallbackPass = 'password';
                            const guessed = `rtsp://${fallbackUser}:${fallbackPass}@${fallbackHost}:554/Streaming/Channels/${camChannel}?starttime=${startCompact}Z&endtime=${endCompact}Z`;
                            navigation.navigate('CCTV', { playbackUrl: guessed, camera: cam, alternates: result.playback_alternates || [] });
                        }
                    }
                }
                
            } else {
                setRecognizedText('No speech detected in the audio.');
                setNlpResult(null);
                setCommandText('');
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
            <TouchableOpacity onPress={() => {
                navigation.navigate('CCTV');
            }}> 
                <Text style={styles.changeServerText}>CCTV Page</Text>
            </TouchableOpacity>
            <Text style={styles.label}>Translation Mode:</Text>
            <View style={styles.pickerContainer}>
                <Picker
                    selectedValue={translationMode}
                    onValueChange={(val) => setTranslationMode(val)}
                    style={styles.picker}
                >
                    <Picker.Item label="None" value="none" />
                    <Picker.Item label="Literal (MT)" value="literal" />
                    <Picker.Item label="High-Level Summary" value="high_level" />
                </Picker>
            </View>

            <View style={{ flexDirection: 'row', alignItems: 'center', marginVertical: 8 }}>
                <Button
                    title={runNlp ? 'Disable NLP' : 'Enable NLP'}
                    onPress={() => setRunNlp(!runNlp)}
                    color={runNlp ? '#ff9800' : '#2196f3'}
                />
                <Text style={{ marginLeft: 12 }}>NLP: {runNlp ? 'ON' : 'OFF'}</Text>
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

            <Text style={styles.resultLabel}>Transcribed / Translated Text:</Text>
            <View style={styles.resultBox}>
                <Text style={styles.resultText}>
                    {recognizedText || 'Your speech will appear here...'}
                </Text>
            </View>

            {nlpResult && (
                <View style={[styles.resultBox, { backgroundColor: '#eef6ff', borderColor: '#90caf9' }] }>
                    <Text style={[styles.resultLabel, { marginTop: 0 }]}>Detected Command Intent:</Text>
                    <Text style={styles.resultText}>Intent: {nlpResult.intent}</Text>
                    <Text style={styles.resultText}>Score: {nlpResult.score?.toFixed(3)}</Text>
                    {nlpResult.entities && Object.keys(nlpResult.entities).length > 0 && (() => {
                        const e = { ...nlpResult.entities };
                        // Collapse single-day range: if date_range_start==date_range_end and equals date, remove range keys
                        if (e.date_range_start && e.date_range_end && e.date_range_start === e.date_range_end) {
                            if (e.date && e.date === e.date_range_start) {
                                delete e.date_range_start; delete e.date_range_end;
                            } else {
                                // No separate date; promote to date
                                if (!e.date) e.date = e.date_range_start;
                                delete e.date_range_start; delete e.date_range_end;
                            }
                        }
                        const entityStr = Object.entries(e).map(([k,v]) => `${k}=${v}`).join(', ');
                        return <Text style={styles.resultText}>Entities: {entityStr}</Text>;
                    })()}
                    {commandText ? (
                        <Text style={[styles.resultText, { marginTop: 6, fontWeight: '600' }]}>Command: {commandText}</Text>
                    ) : null}
                </View>
            )}

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
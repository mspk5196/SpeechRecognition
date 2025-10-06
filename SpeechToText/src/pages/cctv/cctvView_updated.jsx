import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  StyleSheet, View, Text, TouchableOpacity, 
  ScrollView, TextInput, ActivityIndicator, 
  Dimensions, StatusBar, SafeAreaView 
} from 'react-native';
import { Video } from 'expo-av';
import { PanGestureHandler, PinchGestureHandler, State } from 'react-native-gesture-handler';
import Animated from 'react-native-reanimated';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { CCTV_USER, CCTV_PASSWORD } from '../../util/env';

export default function DVRScreen({ route }) {
  const playerRef = useRef(null);
  const [username] = useState(CCTV_USER);
  const [password] = useState(CCTV_PASSWORD);

  // Route params from navigation after transcription
  const playbackUrlParam = route.params?.playbackUrl || null;
  const alternatesParam = route.params?.alternates || [];
  const cameraParam = route.params?.camera || route.params?.channel;
  const expected = route.params?.expected || null; // { start_local_iso, end_local_iso, duration_ms }

  // Extract host/port/channel defaults from first playback candidate if available
  const parseAuthority = (url) => {
    if (!url) return { host: null, port: null, channel: null };
    try {
      const match = url.match(/rtsp:\/\/(?:[^:@]+:[^@]+@)?([^:\/]+):?(\d+)?.*?(?:channel=|channels\/|stream\/)(\w+)/i);
      return match ? { host: match[1], port: match[2], channel: match[3] } : { host: null, port: null, channel: null };
    } catch (e) {
      console.error('Failed parsing RTSP URL:', e);
      return { host: null, port: null, channel: null };
    }
  };
  
  const auth = parseAuthority(playbackUrlParam || alternatesParam[0]);

  const [ip, setIp] = useState(auth.host || 'cctvtest.mspkapps.in');
  const [port, setPort] = useState(auth.port || '519');
  const [channel, setChannel] = useState(cameraParam ? (parseInt(cameraParam,10).toString()) : '201');

  const [altIndex, setAltIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [lastError, setLastError] = useState(null);
  const [reloadCount, setReloadCount] = useState(0);
  const [useTcp, setUseTcp] = useState(true);
  const [networkCachingMs, setNetworkCachingMs] = useState(1500);
  const [videoLoading, setVideoLoading] = useState(true);
  const [videoDuration, setVideoDuration] = useState(0);
  const [videoPosition, setVideoPosition] = useState(0);
  const [showControls, setShowControls] = useState(true);

  // Auto-stop via media-time (stop at expected end if provided)
  const [countdownMs, setCountdownMs] = useState(null);
  const [endedByTimer, setEndedByTimer] = useState(false);
  const [autoStopEnabled, setAutoStopEnabled] = useState(true);
  const autoStopTimerRef = useRef(null);
  
  // Zoom & Pan state
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const baseScaleRef = useRef(1);
  const pinchStartDistRef = useRef(null);
  const lastPanRef = useRef({ x: 0, y: 0 });

  // Manual playback builder state
  const [manualOpen, setManualOpen] = useState(false);
  const today = new Date();
  const isoDate = today.toISOString().slice(0, 10);
  const [mpDate, setMpDate] = useState(isoDate);
  const [mpStart, setMpStart] = useState('10:00');
  const [mpEnd, setMpEnd] = useState('11:00');
  const [mpCam, setMpCam] = useState(cameraParam || '1');
  const [mpPattern, setMpPattern] = useState('/Streaming/tracks/{ch}?starttime={st}Z&endtime={et}Z');
  const [mpChannelMode, setMpChannelMode] = useState('auto'); // auto | raw
  const [manualUrl, setManualUrl] = useState('');

  const alternates = alternatesParam;
  const mode = playbackUrlParam ? 'playback' : 'live';

  const buildManual = useCallback(() => {
    let url = 'rtsp://' + username + ':' + password + '@' + ip + ':' + port;
    
    // Replace tokens in pattern
    let ch = mpCam;
    if (mpChannelMode === 'auto' && mpCam.length < 2) {
      ch = mpCam + '01'; // Common convention: cam1 becomes 101 or 201
    }
    
    // Build ISO timestamps
    let st = mpDate + 'T' + mpStart;
    let et = mpDate + 'T' + mpEnd;
    
    let result = mpPattern
      .replace('{ch}', ch)
      .replace('{st}', st)
      .replace('{et}', et);
      
    url += result;
    setManualUrl(url);
    return url;
  }, [mpDate, mpStart, mpEnd, mpCam, mpPattern, mpChannelMode, username, password, ip, port]);

  const getRtspUrl = useCallback(() => {
    if (manualUrl) return manualUrl;
    if (playbackUrlParam) return playbackUrlParam;
    
    // Default RTSP format
    const candidate = alternates[altIndex] || 'rtsp://{user}:{password}@{host}:{port}/Streaming/Channels/{channel}01?transportmode=unicast&profile=Profile_1';
    return candidate
      .replace('{user}', encodeURIComponent(username))
      .replace('{password}', encodeURIComponent(password))
      .replace('{host}', ip)
      .replace('{port}', port)
      .replace('{channel}', channel);
  }, [manualUrl, playbackUrlParam, username, password, ip, port, channel, alternates, altIndex]);

  const sanitizedUrl = (() => {
    const url = getRtspUrl();
    return url.replace(/\/\/[^:]+:[^@]+@/, '//***:***@'); // Hide credentials in UI
  })();

  const mediaOptions = [
    useTcp ? '--rtsp-tcp' : '--rtsp-mcast',
    `--network-caching=${networkCachingMs}`,
    '--no-stats',
    '--no-drop-late-frames',
    '--no-skip-frames',
  ];

  const clearAutoStopTimers = useCallback(() => {
    if (autoStopTimerRef.current) {
      clearTimeout(autoStopTimerRef.current);
      autoStopTimerRef.current = null;
    }
    setCountdownMs(null);
  }, []);

  const pausePlayback = useCallback(() => {
    if (playerRef.current) {
      playerRef.current.pauseAsync();
      setIsPlaying(false);
      setEndedByTimer(true);
      console.log('Playback paused by timer');
    }
  }, []);

  const getExpectedDurationMs = useCallback(() => {
    if (!expected || !expected.start_local_iso || !expected.end_local_iso) {
      return null;
    }
    
    try {
      const start = new Date(expected.start_local_iso);
      const end = new Date(expected.end_local_iso);
      return Math.max(1000, end - start); // At least 1 second
    } catch (e) {
      console.error('Failed to parse time range:', e);
      return null;
    }
  }, [expected]);

  // Pinch and pan gesture handlers
  const clamp = (val, min, max) => Math.max(min, Math.min(max, val));
  
  const onPinchGestureEvent = useCallback((event) => {
    if (pinchStartDistRef.current === null) {
      pinchStartDistRef.current = event.nativeEvent.scale;
      return;
    }
    
    const diff = event.nativeEvent.scale / pinchStartDistRef.current;
    const newScale = clamp(baseScaleRef.current * diff, 0.5, 5);
    setScale(newScale);
  }, []);
  
  const onPinchHandlerStateChange = useCallback((event) => {
    if (event.nativeEvent.oldState === State.ACTIVE) {
      baseScaleRef.current = scale;
      pinchStartDistRef.current = null;
    }
  }, [scale]);
  
  const onPanGestureEvent = useCallback((event) => {
    if (scale <= 1) return; // Don't pan if not zoomed
    
    const { translationX, translationY } = event.nativeEvent;
    const diffX = translationX - lastPanRef.current.x;
    const diffY = translationY - lastPanRef.current.y;
    
    lastPanRef.current = { x: translationX, y: translationY };
    
    setTranslate(prev => {
      const boundX = (scale - 1) * 160; // Limit pan to scaled bounds
      const boundY = (scale - 1) * 100;
      return {
        x: clamp(prev.x + diffX, -boundX, boundX),
        y: clamp(prev.y + diffY, -boundY, boundY),
      };
    });
  }, [scale]);
  
  const onPanHandlerStateChange = useCallback((event) => {
    if (event.nativeEvent.state === State.BEGAN) {
      lastPanRef.current = { x: 0, y: 0 };
    }
  }, []);
  
  const resetTransform = useCallback(() => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
    baseScaleRef.current = 1;
  }, []);

  // Toggle playback
  const togglePlayback = useCallback(async () => {
    if (playerRef.current) {
      if (isPlaying) {
        await playerRef.current.pauseAsync();
      } else {
        await playerRef.current.playAsync();
      }
      setIsPlaying(!isPlaying);
    }
  }, [isPlaying]);

  // Setup auto-stop timer if needed for playback
  useEffect(() => {
    clearAutoStopTimers();
    
    if (autoStopEnabled && mode === 'playback') {
      const durationMs = getExpectedDurationMs();
      if (!durationMs) return;
      
      console.log(`Setting up auto-stop timer for ${durationMs}ms`);
      setCountdownMs(durationMs);
      
      autoStopTimerRef.current = setTimeout(() => {
        pausePlayback();
      }, durationMs);
      
      // Update countdown every second
      const interval = setInterval(() => {
        setCountdownMs(prev => {
          if (prev <= 1000) {
            clearInterval(interval);
            return 0;
          }
          return prev - 1000;
        });
      }, 1000);
      
      return () => {
        clearInterval(interval);
        clearAutoStopTimers();
      };
    }
  }, [autoStopEnabled, mode, getExpectedDurationMs, clearAutoStopTimers, pausePlayback]);

  // Handle video events
  const onVideoLoad = useCallback((status) => {
    console.log('Video loaded:', status);
    setVideoLoading(false);
    setVideoDuration(status.durationMillis || 0);
  }, []);

  const onVideoError = useCallback((error) => {
    console.error('Video error:', error);
    setLastError(`Error: ${error}`);
    setVideoLoading(false);
  }, []);

  const onPlaybackStatusUpdate = useCallback((status) => {
    if (status.isLoaded) {
      setVideoPosition(status.positionMillis || 0);
      setIsPlaying(status.isPlaying);
    }
  }, []);

  const formatTime = (millis) => {
    if (!millis) return '00:00';
    const totalSeconds = Math.floor(millis / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  // Rebuild manual URL when params change
  useEffect(() => {
    if (manualOpen) buildManual();
  }, [manualOpen, buildManual]);

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />
      <ScrollView style={styles.container} contentContainerStyle={{paddingBottom: 20}}>
        <Text style={styles.header}>
          {mode === 'playback' ? 'Playback' : 'Live View'} - Camera {channel}
        </Text>

        <PanGestureHandler
          onGestureEvent={onPanGestureEvent}
          onHandlerStateChange={onPanHandlerStateChange}
        >
          <Animated.View>
            <PinchGestureHandler
              onGestureEvent={onPinchGestureEvent}
              onHandlerStateChange={onPinchHandlerStateChange}
            >
              <Animated.View style={styles.videoWrapper}>
                <Animated.View 
                  style={[
                    styles.transformer, 
                    {transform: [
                      {scale}, 
                      {translateX: translate.x}, 
                      {translateY: translate.y}
                    ]}
                  ]}
                >
                  <Video
                    ref={playerRef}
                    source={{ uri: getRtspUrl() }}
                    rate={1.0}
                    volume={1.0}
                    isMuted={false}
                    resizeMode="contain"
                    shouldPlay={true}
                    isLooping={false}
                    style={styles.video}
                    onLoad={onVideoLoad}
                    onError={onVideoError}
                    onPlaybackStatusUpdate={onPlaybackStatusUpdate}
                    useNativeControls={false}
                  />
                </Animated.View>
                
                {videoLoading && (
                  <View style={styles.loadingOverlay}>
                    <ActivityIndicator size="large" color="#2d6cdf" />
                    <Text style={styles.loadingText}>Loading video...</Text>
                  </View>
                )}
                
                {showControls && (
                  <View style={styles.videoControls}>
                    <TouchableOpacity style={styles.playButton} onPress={togglePlayback}>
                      <Icon name={isPlaying ? 'pause' : 'play-arrow'} size={28} color="#fff" />
                    </TouchableOpacity>
                    
                    {videoDuration > 0 && (
                      <View style={styles.progressContainer}>
                        <View style={styles.timeTextContainer}>
                          <Text style={styles.timeText}>{formatTime(videoPosition)}</Text>
                          <Text style={styles.timeText}>{formatTime(videoDuration)}</Text>
                        </View>
                        <View style={styles.progressBarContainer}>
                          <View style={styles.progressBarBackground} />
                          <View 
                            style={[
                              styles.progressBarForeground,
                              {width: `${(videoPosition / videoDuration) * 100}%`}
                            ]} 
                          />
                        </View>
                      </View>
                    )}
                  </View>
                )}
              </Animated.View>
            </PinchGestureHandler>
          </Animated.View>
        </PanGestureHandler>

        <View style={styles.controlPanel}>
          <View style={styles.buttonRow}>
            <TouchableOpacity 
              style={[styles.button, isPlaying ? styles.activeButton : null]} 
              onPress={togglePlayback}
            >
              <Text style={styles.buttonText}>{isPlaying ? 'Pause' : 'Play'}</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.button} onPress={() => {
              setReloadCount(prev => prev + 1);
              setVideoLoading(true);
              setLastError(null);
              setIsPlaying(true);
              setEndedByTimer(false);
            }}>
              <Text style={styles.buttonText}>Reload</Text>
            </TouchableOpacity>

            <TouchableOpacity 
              style={[styles.button, scale !== 1 ? styles.activeButton : null]} 
              onPress={resetTransform}
            >
              <Text style={styles.buttonText}>Reset Zoom</Text>
            </TouchableOpacity>

            <TouchableOpacity 
              style={[styles.button, autoStopEnabled ? styles.activeButton : null]} 
              onPress={() => setAutoStopEnabled(!autoStopEnabled)}
            >
              <Text style={styles.buttonText}>Auto-Stop: {autoStopEnabled ? 'ON' : 'OFF'}</Text>
            </TouchableOpacity>
          </View>
          
          {mode === 'playback' && countdownMs !== null && (
            <View style={styles.countdownContainer}>
              <Text style={styles.countdownLabel}>
                Auto-stop in: <Text style={styles.countdownTime}>{Math.ceil(countdownMs / 1000)}s</Text>
              </Text>
            </View>
          )}
          
          {lastError && (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>{lastError}</Text>
            </View>
          )}
        </View>

        <View style={styles.debugBox}>
          <Text style={styles.debugLabel}>Connection URL:</Text>
          <Text style={styles.debugValue} numberOfLines={2} ellipsizeMode="middle">{sanitizedUrl}</Text>
          
          {expected && (
            <>
              <Text style={styles.debugLabel}>Time Range:</Text>
              <Text style={styles.debugValue}>
                {new Date(expected.start_local_iso).toLocaleTimeString()} - {new Date(expected.end_local_iso).toLocaleTimeString()}
              </Text>
            </>
          )}
          
          {alternates.length > 1 && (
            <View style={styles.buttonRow}>
              <Text style={[styles.debugLabel, {marginRight: 8}]}>Try alternate format:</Text>
              {alternates.map((_, idx) => (
                <TouchableOpacity 
                  key={idx} 
                  style={[styles.smallBtn, altIndex === idx ? {backgroundColor: '#1a4a9a'} : null]} 
                  onPress={() => {
                    setAltIndex(idx);
                    setVideoLoading(true);
                    setLastError(null);
                  }}
                >
                  <Text style={styles.smallBtnText}>{idx + 1}</Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
          
          <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.smallBtn} onPress={() => setManualOpen(!manualOpen)}>
              <Text style={styles.smallBtnText}>{manualOpen ? 'Hide Playback Builder' : 'Show Playback Builder'}</Text>
            </TouchableOpacity>
          </View>
        </View>
        
        {manualOpen && (
          <View style={styles.manualBox}>
            <Text style={styles.manualTitle}>Manual Playback URL Builder</Text>
            
            <View style={styles.inputRow}>
              <Text style={styles.inLabel}>Date:</Text>
              <TextInput 
                style={styles.in} 
                value={mpDate} 
                onChangeText={setMpDate} 
                placeholder="YYYY-MM-DD" 
              />
            </View>
            
            <View style={styles.inputRow}>
              <Text style={styles.inLabel}>Start:</Text>
              <TextInput 
                style={styles.in} 
                value={mpStart} 
                onChangeText={setMpStart} 
                placeholder="HH:MM" 
              />
              <Text style={styles.inLabel}>End:</Text>
              <TextInput 
                style={styles.in} 
                value={mpEnd} 
                onChangeText={setMpEnd} 
                placeholder="HH:MM" 
              />
            </View>
            
            <View style={styles.inputRow}>
              <Text style={styles.inLabel}>Camera:</Text>
              <TextInput 
                style={styles.in} 
                value={mpCam} 
                onChangeText={setMpCam} 
                keyboardType="number-pad" 
              />
              <TouchableOpacity
                style={[styles.smallBtn, {marginRight: 0}]}
                onPress={() => setMpChannelMode(mpChannelMode === 'auto' ? 'raw' : 'auto')}
              >
                <Text style={styles.smallBtnText}>{mpChannelMode === 'auto' ? 'Auto' : 'Raw'}</Text>
              </TouchableOpacity>
            </View>
            
            <View style={styles.inputRow}>
              <Text style={styles.inLabel}>Pattern:</Text>
              <TextInput 
                style={styles.in} 
                value={mpPattern} 
                onChangeText={setMpPattern} 
                placeholder="/path/{ch}?start={st}&end={et}" 
              />
            </View>
            
            <Text style={styles.manualUrl} numberOfLines={2}>{manualUrl || 'Click Build to generate URL'}</Text>
            
            <View style={styles.buttonRow}>
              <TouchableOpacity 
                style={[styles.smallBtn, {backgroundColor: '#2d872d'}]} 
                onPress={buildManual}
              >
                <Text style={styles.smallBtnText}>Build URL</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.smallBtn, {backgroundColor: '#2d6cdf'}]} 
                onPress={() => {
                  const url = buildManual();
                  setManualUrl(url);
                  setVideoLoading(true);
                  setLastError(null);
                  setEndedByTimer(false);
                }}
              >
                <Text style={styles.smallBtnText}>Build & Play</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1, 
    backgroundColor: '#0c0c0c',
  },
  container: { 
    flex: 1, 
    backgroundColor: '#111' 
  },
  header: { 
    fontSize: 22, 
    fontWeight: 'bold', 
    color: '#fff', 
    textAlign: 'center', 
    marginVertical: 14,
    letterSpacing: 0.5,
  },
  videoWrapper: { 
    height: 300, 
    marginHorizontal: 10, 
    borderRadius: 12, 
    overflow: 'hidden', 
    backgroundColor: '#000',
    elevation: 5,
    position: 'relative',
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.4)',
  },
  loadingText: {
    color: '#fff',
    marginTop: 10,
    fontSize: 14,
  },
  transformer: { 
    width: '100%', 
    height: '100%' 
  },
  video: { 
    width: '100%', 
    height: '100%' 
  },
  videoControls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 10,
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  playButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.6)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 5,
  },
  progressContainer: {
    width: '100%',
  },
  timeTextContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  timeText: {
    color: '#fff',
    fontSize: 12,
  },
  progressBarContainer: {
    height: 4,
    width: '100%',
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 2,
    position: 'relative',
  },
  progressBarBackground: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderRadius: 2,
  },
  progressBarForeground: {
    position: 'absolute',
    top: 0,
    left: 0,
    bottom: 0,
    borderRadius: 2,
    backgroundColor: '#2d6cdf',
  },
  controlPanel: {
    backgroundColor: '#1a1a1a',
    marginHorizontal: 10,
    marginTop: 15,
    borderRadius: 12,
    padding: 12,
    elevation: 3,
  },
  countdownContainer: {
    marginTop: 10,
    padding: 8,
    backgroundColor: '#2a2a2a',
    borderRadius: 6,
    alignItems: 'center',
  },
  countdownLabel: {
    color: '#ddd',
    fontSize: 14,
  },
  countdownTime: {
    fontWeight: 'bold',
    color: '#2d6cdf',
  },
  errorContainer: {
    marginTop: 10,
    padding: 8,
    backgroundColor: '#3a1414',
    borderRadius: 6,
  },
  errorText: {
    color: '#ff6b6b',
    fontSize: 13,
  },
  debugBox: { 
    padding: 14, 
    backgroundColor: '#1e1e1e', 
    marginHorizontal: 10, 
    marginTop: 15, 
    borderRadius: 12,
    elevation: 2,
  },
  debugLabel: { 
    color: '#bbb', 
    fontSize: 13, 
    marginTop: 6 
  },
  debugValue: { 
    color: '#fff',
    fontSize: 13,
    marginBottom: 4,
  },
  buttonRow: { 
    flexDirection: 'row', 
    marginTop: 10, 
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  button: {
    backgroundColor: '#2d6cdf',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    marginRight: 10,
    marginBottom: 10,
    minWidth: 80,
    alignItems: 'center',
    elevation: 2,
  },
  activeButton: {
    backgroundColor: '#1a4a9a',
  },
  buttonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  smallBtn: { 
    backgroundColor: '#2d6cdf', 
    paddingVertical: 6, 
    paddingHorizontal: 12, 
    borderRadius: 6, 
    marginRight: 8, 
    marginTop: 6 
  },
  smallBtnText: { 
    color: '#fff', 
    fontSize: 12, 
    fontWeight: '600' 
  },
  manualBox: { 
    backgroundColor: '#222', 
    marginHorizontal: 10, 
    marginTop: 15,
    padding: 14, 
    borderRadius: 12, 
    marginBottom: 12,
    elevation: 2,
  },
  manualTitle: { 
    color: '#fff', 
    fontWeight: 'bold', 
    marginBottom: 12,
    fontSize: 16,
  },
  inputRow: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    marginBottom: 10 
  },
  inLabel: { 
    color: '#aaa', 
    width: 60, 
    fontSize: 13 
  },
  in: { 
    backgroundColor: '#333', 
    color: '#eee', 
    paddingHorizontal: 10, 
    paddingVertical: 6, 
    borderRadius: 6, 
    marginRight: 8, 
    flexGrow: 1, 
    minWidth: 60, 
    fontSize: 13 
  },
  manualUrl: { 
    color: '#6cf', 
    fontSize: 12, 
    marginTop: 8,
    marginBottom: 4,
  }
});
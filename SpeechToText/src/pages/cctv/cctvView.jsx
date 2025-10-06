import React, { useEffect, useRef, useState, useCallback } from 'react';
import { View, StyleSheet, Text, TouchableOpacity, TextInput, ScrollView, PanResponder } from 'react-native';
import { VLCPlayer } from 'react-native-vlc-media-player';
import { CCTV_USER, CCTV_PASSWORD, API_URL } from '../../util/env';
import { getPlayerConfig } from '../../util/apiClient';

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
    try {
      if (!url) return {}; // fallback
      const m = url.match(/^rtsp:\/\/[^@]+@([^\/]+)(\/.*)?$/i);
      if (!m) return {};
      const hostPort = m[1];
      const [host, port] = hostPort.split(':');
      return { host, port: port || '519' };
    } catch { return {}; }
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

  // Auto-stop via media-time (stop at expected end if provided)
  const [countdownMs, setCountdownMs] = useState(null);
  const [endedByTimer, setEndedByTimer] = useState(false);
  const [autoStopEnabled, setAutoStopEnabled] = useState(true);
  const [playerConfig, setPlayerConfig] = useState({
    auto_stop_playback: true,
    player_controls: true,
    progress_bar: true,
    loading_indicator: true
  });
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
    try {
      const dateBase = mpDate.replace(/-/g, '')
      const norm = (t) => { const p = t.split(':'); return p[0].padStart(2, '0') + (p[1] ? p[1].padStart(2, '0') : '00') + '00'; };
      const st = dateBase + 'T' + norm(mpStart);
      const et = dateBase + 'T' + norm(mpEnd);
      const camInt = parseInt(mpCam || '1', 10);
      // Channel variants simple heuristic
      let chVal = camInt + '01';
      if (mpChannelMode === 'raw') chVal = mpCam;
      if (mpChannelMode === 'auto' && camInt < 10) {
        chVal = camInt.toString().padStart(2, '0') + '01';
      }
      let pattern = mpPattern;
      pattern = pattern.replace('{channel}', chVal).replace('{ch}', chVal);
      if (!pattern.includes('{st}')) pattern += (pattern.includes('?') ? '&' : '?') + 'starttime={st}Z&endtime={et}Z';
      pattern = pattern.replace('{st}', st).replace('{et}', et);
      let full;
      if (pattern.startsWith('rtsp://')) full = pattern; else full = `rtsp://${encodeURIComponent(username)}:${encodeURIComponent(password)}@${ip}:${port}${pattern}`;
      setManualUrl(full);
      setAltIndex(0);
      setReloadCount(c => c + 1);
    } catch (e) {
      setManualUrl('ERROR: ' + e.message);
    }
  }, [mpDate, mpStart, mpEnd, mpCam, mpPattern, mpChannelMode, username, password, ip, port]);

  const getRtspUrl = useCallback(() => {
    if (manualUrl) return manualUrl;
    
    if (playbackUrlParam) {
      if (alternates.length > 0) {
        return alternates[Math.min(altIndex, alternates.length - 1)];
      }
      return playbackUrlParam;
    }
    console.log("hi");
    return `rtsp://${username}:${password}@${ip}:${port}/Streaming/Channels/${channel}`;
  }, [manualUrl, playbackUrlParam, username, password, ip, port, channel, alternates, altIndex]);

  const sanitizedUrl = (() => {
    const full = getRtspUrl();
    return full.replace(/:\/\/.*?:.*?@/, '://***:***@');
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
    try { playerRef.current?.pause && playerRef.current.pause(); } catch (_) {}
    setIsPlaying(false);
    setEndedByTimer(true);
  }, []);

  const getExpectedDurationMs = useCallback(() => {
    // Prefer explicit duration; else compute from start/end; else 0
    if (expected?.duration_ms && expected.duration_ms > 0) return expected.duration_ms;
    if (expected?.start_local_iso && expected?.end_local_iso) {
      const s = Date.parse(expected.start_local_iso);
      const e = Date.parse(expected.end_local_iso);
      if (!Number.isNaN(s) && !Number.isNaN(e)) return Math.max(0, e - s);
    }
    return 0;
  }, [expected]);

  // Pinch and pan gesture handlers
  const clamp = (val, min, max) => Math.max(min, Math.min(max, val));
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: (evt) => {
        const t = evt?.nativeEvent?.touches || [];
        return t.length === 2 || scale > 1; // pinch or pan when zoomed
      },
      onMoveShouldSetPanResponder: (evt, gestureState) => {
        const t = evt?.nativeEvent?.touches || [];
        if (t.length === 2) return true; // pinch
        return scale > 1 && (Math.abs(gestureState.dx) + Math.abs(gestureState.dy) > 4);
      },
      onStartShouldSetPanResponderCapture: (evt) => {
        const t = evt?.nativeEvent?.touches || [];
        return t.length === 2 || scale > 1;
      },
      onMoveShouldSetPanResponderCapture: (evt) => {
        const t = evt?.nativeEvent?.touches || [];
        return t.length === 2 || scale > 1;
      },
      onPanResponderGrant: (evt, gestureState) => {
        lastPanRef.current = { ...translate };
        baseScaleRef.current = scale;
        if (evt.nativeEvent.touches && evt.nativeEvent.touches.length === 2) {
          const [a, b] = evt.nativeEvent.touches;
          const dx = a.pageX - b.pageX;
          const dy = a.pageY - b.pageY;
          pinchStartDistRef.current = Math.hypot(dx, dy);
        } else {
          pinchStartDistRef.current = null;
        }
      },
      onPanResponderMove: (evt, gestureState) => {
        if (evt.nativeEvent.touches && evt.nativeEvent.touches.length === 2 && pinchStartDistRef.current) {
          const [a, b] = evt.nativeEvent.touches;
          const dx = a.pageX - b.pageX;
          const dy = a.pageY - b.pageY;
          const dist = Math.hypot(dx, dy);
          const nextScale = clamp((dist / pinchStartDistRef.current) * baseScaleRef.current, 1, 4);
          setScale(nextScale);
        } else if (evt.nativeEvent.touches && evt.nativeEvent.touches.length === 1) {
          const nx = lastPanRef.current.x + gestureState.dx;
          const ny = lastPanRef.current.y + gestureState.dy;
          setTranslate({ x: nx, y: ny });
        }
      },
      onPanResponderRelease: () => {
        baseScaleRef.current = scale;
        lastPanRef.current = { ...translate };
        pinchStartDistRef.current = null;
      },
      onPanResponderTerminationRequest: () => false,
    })
  ).current;

  const handleError = (e) => {
    console.error('VLC Player Error:', e);
    setLastError(e);
    if (!manualUrl && alternates.length > 0 && altIndex < alternates.length - 1) {
      setTimeout(() => {
        setAltIndex(i => i + 1);
        setReloadCount(c => c + 1);
      }, 750);
    }
  };

  const manualReload = () => {
    setLastError(null);
    setReloadCount(c => c + 1);
    if (playerRef.current) {
      try { playerRef.current.reload(); } catch (_) { }
    }
  };

  // Mark success back to backend once a playback URL has played for a bit
  useEffect(() => {
    if (!playbackUrlParam && !manualUrl) return;
    if (mode !== 'playback' && !manualUrl) return;
    const timer = setTimeout(() => {
      const activeUrl = getRtspUrl();
      // Extract pattern guess and channel from URL
      try {
        const u = activeUrl.split('?')[0];
        let patternName = 'unknown';
        if (u.includes('/Streaming/tracks/')) patternName = 'tracks';
        else if (u.includes('Playback=1')) patternName = 'playflag';
        else if (u.includes('/ISAPI/Streaming/tracks/')) patternName = 'isapi_tracks';
        else if (u.includes('/Streaming/Channels/')) patternName = 'base';
        const chMatch = u.match(/\/(tracks|Channels)\/([^?]+)$/i);
        const ch = chMatch ? chMatch[2] : cameraParam || channel;
        fetch(`${API_URL || ''}/playback/mark_success`, {
          method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `pattern=${encodeURIComponent(patternName)}&channel=${encodeURIComponent(ch)}`
        }).catch(() => {});
      } catch (_) {}
    }, 4000); // 4s of stable playback implies success
    return () => clearTimeout(timer);
  }, [altIndex, manualUrl, playbackUrlParam, mode, getRtspUrl, cameraParam, channel]);

  useEffect(() => {
    if (mode === 'playback' || manualUrl) return; // don't auto reload playback or manual
    const interval = setInterval(() => {
      if (playerRef.current && isPlaying) {
        playerRef.current.reload();
      }
    }, 60000);
    return () => clearInterval(interval);
  }, [mode, manualUrl, isPlaying]);

  // Reset auto-stop timers whenever source changes
  useEffect(() => {
    clearAutoStopTimers();
    setEndedByTimer(false);
    // Reset zoom when source changes
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }, [manualUrl, altIndex, reloadCount, useTcp, networkCachingMs, clearAutoStopTimers]);

  // Fetch player configuration from server
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const config = await getPlayerConfig();
        setPlayerConfig(config);
        setAutoStopEnabled(config.auto_stop_playback);
      } catch (error) {
        console.warn('Failed to load player configuration:', error);
      }
    };
    
    loadConfig();
  }, []);
  
  // Handle auto-stop functionality when expected playback time is available
  useEffect(() => {
    // Only apply auto-stop if enabled and we have expected duration data
    if (!autoStopEnabled || mode !== 'playback' || !expected) return;
    
    // Clear any existing auto-stop timers
    if (autoStopTimerRef.current) {
      clearTimeout(autoStopTimerRef.current);
      autoStopTimerRef.current = null;
    }
    
    // Calculate duration from expected values
    const durationMs = getExpectedDurationMs();
    if (durationMs <= 0) return;
    
    console.log(`Setting up auto-stop timer for ${durationMs}ms (${durationMs/1000}s)`);
    
    // Set countdown for UI display
    setCountdownMs(durationMs);
    
    // Set up timer for auto-stop
    autoStopTimerRef.current = setTimeout(() => {
      if (isPlaying) {
        pausePlayback();
        console.log('Auto-stop timer completed - pausing playback');
      }
      setCountdownMs(null);
    }, durationMs);
    
    // Update countdown every second for UI display
    const intervalId = setInterval(() => {
      setCountdownMs(prev => {
        if (prev === null || prev <= 0) {
          clearInterval(intervalId);
          return null;
        }
        return prev - 1000;
      });
    }, 1000);
    
    return () => {
      // Clean up on unmount or when dependencies change
      if (autoStopTimerRef.current) {
        clearTimeout(autoStopTimerRef.current);
        autoStopTimerRef.current = null;
      }
      clearInterval(intervalId);
    };
  }, [autoStopEnabled, mode, expected, isPlaying, getExpectedDurationMs, pausePlayback]);

  // Cleanup on unmount
  useEffect(() => () => clearAutoStopTimers(), [clearAutoStopTimers]);

  const formatMs = (ms) => {
    if (ms == null) return '-';
    const s = Math.max(0, Math.floor(ms / 1000));
    const mm = Math.floor(s / 60).toString().padStart(2, '0');
    const ss = (s % 60).toString().padStart(2, '0');
    return `${mm}:${ss}`;
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>
        {(manualUrl ? 'Manual Playback' : (mode === 'playback' ? 'Playback CCTV Stream' : 'Live CCTV Stream'))} {cameraParam ? `(Camera ${cameraParam})` : ''}
      </Text>
  <ScrollView style={styles.scroll} contentContainerStyle={{ paddingBottom: 20 }} scrollEnabled={scale === 1}>
        <View style={styles.debugBox}>
          <Text style={styles.debugLabel}>RTSP URL:</Text>
          <Text style={styles.debugValue}>{sanitizedUrl}</Text>
          <Text style={styles.debugLabel}>Mode: <Text style={styles.debugValue}>{manualUrl ? 'manual' : mode}</Text></Text>
          <Text style={styles.debugLabel}>Reloads: <Text style={styles.debugValue}>{reloadCount}</Text></Text>
          <Text style={styles.debugLabel}>TCP: <Text style={styles.debugValue}>{useTcp ? 'on' : 'off'}</Text></Text>
          {expected && (
            <>
              <Text style={styles.debugLabel}>Expected:</Text>
              <Text style={styles.debugLabel}>- Start: <Text style={styles.debugValue}>{expected.start_local_iso || '-'}</Text></Text>
              <Text style={styles.debugLabel}>- End: <Text style={styles.debugValue}>{expected.end_local_iso || '-'}</Text></Text>
              <Text style={styles.debugLabel}>- Duration: <Text style={styles.debugValue}>{formatMs(getExpectedDurationMs())}</Text></Text>
              <Text style={styles.debugLabel}>- Auto-stop: <Text style={styles.debugValue}>{autoStopEnabled ? 'on (media-time)' : 'off'}</Text></Text>
              {autoStopEnabled && (
                <Text style={styles.debugLabel}>- Remaining: <Text style={styles.debugValue}>{formatMs(countdownMs)}</Text></Text>
              )}
              {endedByTimer && (
                <Text style={[styles.debugLabel, { color: '#6cf' }]}>Reached end of requested window. Paused.</Text>
              )}
            </>
          )}
          <Text style={styles.debugLabel}>Player Config:</Text>
          <Text style={styles.debugLabel}>- Auto-stop: <Text style={styles.debugValue}>{playerConfig.auto_stop_playback ? 'Enabled' : 'Disabled'}</Text></Text>
          <Text style={styles.debugLabel}>- Controls: <Text style={styles.debugValue}>{playerConfig.player_controls ? 'Visible' : 'Hidden'}</Text></Text>
          <Text style={styles.debugLabel}>- Progress bar: <Text style={styles.debugValue}>{playerConfig.progress_bar ? 'Visible' : 'Hidden'}</Text></Text>
          <Text style={styles.debugLabel}>- Loading indicator: <Text style={styles.debugValue}>{playerConfig.loading_indicator ? 'Visible' : 'Hidden'}</Text></Text>
          {alternates.length > 1 && mode==='playback' && !manualUrl && (
            <Text style={styles.debugLabel}>Alternate {altIndex+1}/{alternates.length}</Text>
          )}
          {lastError && (
            <Text style={[styles.debugLabel, { color: '#ff6b6b' }]}>Last Error: <Text style={styles.debugValue}>{JSON.stringify(lastError)}</Text></Text>
          )}
          <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.smallBtn} onPress={manualReload}>
              <Text style={styles.smallBtnText}>Reload</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.smallBtn} onPress={() => setUseTcp(t => !t)}>
              <Text style={styles.smallBtnText}>{useTcp ? 'Use UDP' : 'Use TCP'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.smallBtn} onPress={() => setNetworkCachingMs(ms => ms === 1500 ? 300 : 1500)}>
              <Text style={styles.smallBtnText}>Cache {networkCachingMs}ms</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.smallBtn} onPress={() => setManualOpen(o => !o)}>
              <Text style={styles.smallBtnText}>{manualOpen ? 'Hide Manual' : 'Manual'}</Text>
            </TouchableOpacity>
            {/* Playback controls */}
            <TouchableOpacity style={styles.smallBtn} onPress={() => { 
              try { 
                if (playerRef.current?.resume) playerRef.current.resume();
                else if (playerRef.current?.play) playerRef.current.play();
              } catch (_) {}
              setIsPlaying(true); 
            }}>
              <Text style={styles.smallBtnText}>Play</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.smallBtn} onPress={() => { try { playerRef.current?.pause && playerRef.current.pause(); } catch (_) {} setIsPlaying(false); }}>
              <Text style={styles.smallBtnText}>Pause</Text>
            </TouchableOpacity>
            {/* Zoom controls */}
            <TouchableOpacity style={styles.smallBtn} onPress={() => setScale(s => Math.min(4, s + 0.25))}>
              <Text style={styles.smallBtnText}>Zoom +</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.smallBtn} onPress={() => setScale(s => { const ns = Math.max(1, s - 0.25); if (ns === 1) setTranslate({ x: 0, y: 0 }); return ns; })}>
              <Text style={styles.smallBtnText}>Zoom -</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.smallBtn} onPress={() => { setScale(1); setTranslate({ x: 0, y: 0 }); baseScaleRef.current = 1; lastPanRef.current = { x: 0, y: 0 }; }}>
              <Text style={styles.smallBtnText}>Reset Zoom</Text>
            </TouchableOpacity>
            {mode === 'playback' && expected && (
              <TouchableOpacity 
                style={[styles.smallBtn, { backgroundColor: autoStopEnabled ? '#2d6cdf' : '#555' }]} 
                onPress={() => { 
                  const newValue = !autoStopEnabled; 
                  setAutoStopEnabled(newValue); 
                  if (!newValue) clearAutoStopTimers();
                }}
              >
                <Text style={styles.smallBtnText}>Auto-Stop: {autoStopEnabled ? 'ON' : 'OFF'}</Text>
              </TouchableOpacity>
            )}
            {alternates.length > 1 && mode==='playback' && !manualUrl && (
              <TouchableOpacity style={styles.smallBtn} onPress={() => { setAltIndex(i => (i + 1) % alternates.length); setReloadCount(c=>c+1); }}>
                <Text style={styles.smallBtnText}>Next Alt</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>

        {manualOpen && (
          <View style={styles.manualBox}>
            <Text style={styles.manualTitle}>Manual Playback Builder</Text>
            <View style={styles.inputRow}>
              <Text style={styles.inLabel}>IP</Text>
              <TextInput style={styles.in} value={ip} onChangeText={setIp} />
              <Text style={styles.inLabel}>Port</Text>
              <TextInput style={styles.in} value={port} onChangeText={setPort} keyboardType='numeric' />
            </View>
            <View style={styles.inputRow}>
              <Text style={styles.inLabel}>Date</Text>
              <TextInput style={styles.in} value={mpDate} onChangeText={setMpDate} placeholder='YYYY-MM-DD' />
              <Text style={styles.inLabel}>Cam</Text>
              <TextInput style={styles.in} value={mpCam} onChangeText={setMpCam} keyboardType='numeric' />
            </View>
            <View style={styles.inputRow}>
              <Text style={styles.inLabel}>Start</Text>
              <TextInput style={styles.in} value={mpStart} onChangeText={setMpStart} placeholder='HH:MM' />
              <Text style={styles.inLabel}>End</Text>
              <TextInput style={styles.in} value={mpEnd} onChangeText={setMpEnd} placeholder='HH:MM' />
            </View>
            <View style={styles.inputRow}>
              <Text style={styles.inLabel}>ChMode</Text>
              <TextInput style={styles.in} value={mpChannelMode} onChangeText={setMpChannelMode} placeholder='auto|raw' />
              <Text style={styles.inLabel}>Channel</Text>
              <TextInput style={styles.in} value={channel} onChangeText={setChannel} />
            </View>
            <Text style={styles.inLabel}>Pattern</Text>
            <TextInput style={[styles.in, { width: '100%' }]} value={mpPattern} onChangeText={setMpPattern} multiline />
            <View style={styles.buttonRow}>
              <TouchableOpacity style={styles.smallBtn} onPress={buildManual}>
                <Text style={styles.smallBtnText}>Build & Play</Text>
              </TouchableOpacity>
              {manualUrl ? (
                <TouchableOpacity style={[styles.smallBtn, { backgroundColor: '#555' }]} onPress={() => { setManualUrl(''); setReloadCount(c => c + 1); }}>
                  <Text style={styles.smallBtnText}>Clear</Text>
                </TouchableOpacity>
              ) : null}
            </View>
            {manualUrl ? <Text style={styles.manualUrl}>{manualUrl}</Text> : null}
          </View>
        )}

        <View style={styles.videoWrapper} {...panResponder.panHandlers}>
          <View style={[styles.transformer, { transform: [{ translateX: translate.x }, { translateY: translate.y }, { scale }] }]}>
            <VLCPlayer
              key={sanitizedUrl + reloadCount + (useTcp?'tcp':'udp') + altIndex}
              ref={playerRef}
              source={{ uri: getRtspUrl(), initType: 2 }}
              autoplay={isPlaying}
              style={styles.video}
              autoAspectRatio={true}
              resizeMode="contain"
              initType={2}
              onBuffering={() => console.log('Buffering...')}
              onPlaying={() => { console.log('Playing'); setLastError(null); setEndedByTimer(false); }}
              onPaused={() => console.log('Paused')}
              onStopped={() => console.log('Stopped')}
              onProgress={(e) => {
                try {
                  const toMs = (v) => {
                    if (v == null || Number.isNaN(v)) return 0;
                    const n = Number(v);
                    return n < 10000 ? Math.floor(n * 1000) : Math.floor(n);
                  };
                  const curMs = toMs(e?.currentTime ?? e?.position ?? 0);
                  const dur = getExpectedDurationMs();
                  if (dur > 0) {
                    const rem = Math.max(0, dur - curMs);
                    setCountdownMs(rem);
                    if (rem <= 250) {
                      try { playerRef.current?.pause && playerRef.current.pause(); } catch(_) {}
                      setIsPlaying(false);
                      setEndedByTimer(true);
                    }
                  }
                } catch(_) {}
              }}
              onError={handleError}
              mediaOptions={mediaOptions}
            />
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#111' },
  scroll: { flex: 1 },
  header: { fontSize: 20, fontWeight: 'bold', color: '#fff', textAlign: 'center', marginVertical: 10 },
  videoWrapper: { height: 260, marginHorizontal: 10, borderRadius: 8, overflow: 'hidden', backgroundColor: '#000' },
  transformer: { width: '100%', height: '100%' },
  video: { width: '100%', height: '100%' },
  debugBox: { padding: 12, backgroundColor: '#1e1e1e', marginHorizontal: 10, marginBottom: 8, borderRadius: 8 },
  debugLabel: { color: '#bbb', fontSize: 12, marginTop: 2 },
  debugValue: { color: '#fff' },
  buttonRow: { flexDirection: 'row', marginTop: 8, flexWrap: 'wrap' },
  smallBtn: { backgroundColor: '#2d6cdf', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 6, marginRight: 8, marginTop: 6 },
  smallBtnText: { color: '#fff', fontSize: 12, fontWeight: '600' },
  manualBox: { backgroundColor: '#222', marginHorizontal: 10, padding: 12, borderRadius: 8, marginBottom: 12 },
  manualTitle: { color: '#fff', fontWeight: 'bold', marginBottom: 8 },
  inputRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 6 },
  inLabel: { color: '#aaa', width: 60, fontSize: 12 },
  in: { backgroundColor: '#333', color: '#eee', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4, marginRight: 8, flexGrow: 1, minWidth: 60, fontSize: 12 },
  manualUrl: { color: '#6cf', fontSize: 11, marginTop: 8 }
});

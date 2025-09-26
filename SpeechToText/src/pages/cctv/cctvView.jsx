import React, { useEffect, useRef, useState, useCallback } from 'react';
import { View, StyleSheet, Text, TouchableOpacity, TextInput, ScrollView } from 'react-native';
import { VLCPlayer } from 'react-native-vlc-media-player';
import { CCTV_USER, CCTV_PASSWORD } from '../../util/env';

export default function DVRScreen({ route }) {
  const playerRef = useRef(null);
  const [username] = useState(CCTV_USER);
  const [password] = useState(CCTV_PASSWORD);
  const params = route?.params || {};
  const playbackUrlParam = params.playbackUrl;
  const alternates = params.alternates || [];
  const [altIndex, setAltIndex] = useState(0);
  const cameraParam = '101';
  const [ip, setIp] = useState(params.host || 'cctvtest.mspkapps.in');
  const [port, setPort] = useState(params.port || '519');
  const [channel, setChannel] = useState(params.channel || '101');
  const mode = playbackUrlParam ? 'playback' : 'live';
  const [isPlaying, setIsPlaying] = useState(true);
  const [lastError, setLastError] = useState(null);
  const [reloadCount, setReloadCount] = useState(0);
  const [useTcp, setUseTcp] = useState(true);
  const [networkCachingMs, setNetworkCachingMs] = useState(1500);

  // Manual playback builder state
  const [manualOpen, setManualOpen] = useState(false);
  const today = new Date();
  const isoDate = today.toISOString().slice(0, 10);
  const [mpDate, setMpDate] = useState(isoDate);
  const [mpStart, setMpStart] = useState('10:00');
  const [mpEnd, setMpEnd] = useState('11:00');
  const [mpCam, setMpCam] = useState(cameraParam || '1');
  const [mpPattern, setMpPattern] = useState('/Streaming/Channels/{ch}?Playback=1&starttime={st}Z&endtime={et}Z');
  const [mpChannelMode, setMpChannelMode] = useState('auto'); // auto | raw
  const [manualUrl, setManualUrl] = useState('');

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
        // prefer zero padded +01 form first
        chVal = camInt.toString().padStart(2, '0') + '01';
      }
      let pattern = mpPattern;
      // inject channel token
      pattern = pattern.replace('{channel}', chVal).replace('{ch}', chVal);
      // ensure time placeholders
      if (!pattern.includes('{st}')) pattern += (pattern.includes('?') ? '&' : '?') + 'starttime={st}Z&endtime={et}Z';
      pattern = pattern.replace('{st}', st).replace('{et}', et);
      // Prepend authority if pattern not full URL
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
    // return `rtsp://${username}:${password}@${ip}:${port}/Streaming/Channels/${1}?Playback=1&starttime=${'10:00'}Z&endtime=${'11:00'}Z`;
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

  useEffect(() => {
    if (mode === 'playback' || manualUrl) return; // don't auto reload playback or manual
    const interval = setInterval(() => {
      if (playerRef.current && isPlaying) {
        playerRef.current.reload();
      }
    }, 60000);
    return () => clearInterval(interval);
  }, [mode, manualUrl, isPlaying]);

  return (
    <View style={styles.container}>
      <Text style={styles.header}>
        {(manualUrl ? 'Manual Playback' : (mode === 'playback' ? 'Playback CCTV Stream' : 'Live CCTV Stream'))} {cameraParam ? `(Camera ${cameraParam})` : ''}
      </Text>
      <ScrollView style={styles.scroll} contentContainerStyle={{ paddingBottom: 20 }}>
        <View style={styles.debugBox}>
          <Text style={styles.debugLabel}>RTSP URL:</Text>
          <Text style={styles.debugValue}>{sanitizedUrl}</Text>
          <Text style={styles.debugLabel}>Mode: <Text style={styles.debugValue}>{manualUrl ? 'manual' : mode}</Text></Text>
          <Text style={styles.debugLabel}>Reloads: <Text style={styles.debugValue}>{reloadCount}</Text></Text>
          <Text style={styles.debugLabel}>TCP: <Text style={styles.debugValue}>{useTcp ? 'on' : 'off'}</Text></Text>
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
            <TouchableOpacity style={styles.smallBtn} onPress={() => { setManualOpen(o => !o); }}>
              <Text style={styles.smallBtnText}>{manualOpen ? 'Hide Manual' : 'Manual'}</Text>
            </TouchableOpacity>
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

        <VLCPlayer
          key={sanitizedUrl + reloadCount + (useTcp?'tcp':'udp') + altIndex}
          ref={playerRef}
          // source={{ uri: `rtsp://${CCTV_USER}:${CCTV_PASSWORD}@cctvtest.mspkapps.in:519/Streaming/Channels/${1}?Playback=1&starttime='10:00'Z&endtime='11:00'Z` }}
          source={{ uri: getRtspUrl(), initType: 2 }}
          autoplay={isPlaying}
          style={styles.video}
          autoAspectRatio={true}
          resizeMode="contain"
          initType={2}
          onBuffering={() => console.log('Buffering...')}
          onPlaying={() => { console.log('Playing'); setLastError(null); }}
          onPaused={() => console.log('Paused')}
          onStopped={() => console.log('Stopped')}
          onError={handleError}
          mediaOptions={mediaOptions}
        />

        {/* <VLCPlayer
          source={{
            uri: `rtsp://${CCTV_USER}:${CCTV_PASSWORD}@cctvtest.mspkapps.in:519/Streaming/tracks/101?starttime=20250926T073000Z&endtime=20250926T110000Z`
          }}
          autoplay={true}
          style={styles.video}
          autoAspectRatio={true}
          resizeMode="contain"
        /> */}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#111' },
  scroll: { flex: 1 },
  header: { fontSize: 20, fontWeight: 'bold', color: '#fff', textAlign: 'center', marginVertical: 10 },
  video: { height: 260,  marginHorizontal: 10, borderRadius: 8 },
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

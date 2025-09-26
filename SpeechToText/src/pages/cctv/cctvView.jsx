import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { VLCPlayer } from 'react-native-vlc-media-player';
import { CCTV_USER, CCTV_PASSWORD } from '../../util/env';

export default function DVRScreen({ route }) {
  const playerRef = useRef(null);
  const [username] = useState(CCTV_USER);
  const [password] = useState(CCTV_PASSWORD);
  const [ip] = useState('cctvtest.mspkapps.in');
  const [port] = useState('519');
  const params = route?.params || {};
  const playbackUrlParam = params.playbackUrl;
  const cameraParam = params.camera;
  const mode = playbackUrlParam ? 'playback' : 'live';
  const [channel] = useState(params.channel || '101');
 
  const getRtspUrl = () => {
    if (playbackUrlParam) return playbackUrlParam; // fully formed URL from backend
    return `rtsp://${username}:${password}@${ip}:${port}/Streaming/Channels/${channel}`;
  };

  const [isPlaying, setIsPlaying] = useState(true); // Start automatically

  // Optional: auto-reconnect if stream drops
  useEffect(() => {
    const interval = setInterval(() => {
      if (playerRef.current) { 
        playerRef.current.reload(); // Reload stream every 60s
      }
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <View style={styles.container}>
  <Text style={styles.header}>{mode === 'playback' ? 'Playback CCTV Stream' : 'Live CCTV Stream'} {cameraParam ? `(Camera ${cameraParam})` : ''}</Text>

      <VLCPlayer
        ref={playerRef}
        source={{ uri: getRtspUrl() }}
        autoplay={isPlaying}   // Auto-play enabled
        style={styles.video}
        onBuffering={() => console.log('Buffering...')}
        onPlaying={() => console.log('Playing')}
        onPaused={() => console.log('Paused')}
        onStopped={() => console.log('Stopped')}
        onError={(e) => console.error('VLC Player Error:', e)}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#111' },
  header: { fontSize: 20, fontWeight: 'bold', color: '#fff', textAlign: 'center', marginVertical: 10 },
  video: { flex: 1 },
});

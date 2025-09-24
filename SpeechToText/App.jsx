import React, { useState, useEffect } from 'react';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import Routes from './src/components/Routes/Routes';
import { StatusBar } from 'react-native';

const App = () => {

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="dark-content" backgroundColor="#f5f5f5" />
      <SafeAreaView style={{ flex: 1, backgroundColor: '#f5f5f5', padding: 5 }}>
        <Routes />
      </SafeAreaView>
    </SafeAreaProvider>
  );
};



export default App;

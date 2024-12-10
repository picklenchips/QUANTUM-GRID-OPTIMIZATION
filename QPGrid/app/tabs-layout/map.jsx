import React from 'react';
import { StyleSheet, View, Dimensions } from 'react-native';
import { WebView } from 'react-native-webview';

const MapScreen = () => {
    const webviewHeight = Dimensions.get('window').height;

    return (
        <View style={styles.container}>
            <WebView
                originWhitelist={['*']}
                source={require('../../../web/opengridmap_source.html')} 
                style={{ height: webviewHeight }}
            />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
});

export default MapScreen;

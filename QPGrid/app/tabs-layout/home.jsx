import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';

// Only 'home' / 'map' / 'info' are registered tabs (see _layout.jsx) --
// 'Dashboard'/'Live Data' map onto the map tab (the only data view that
// exists), 'About' onto the info tab (the only content page that exists).
// No new screens invented, just wired to what's actually there.
const HomeScreen = () => {
    const router = useRouter();
    return (
        <View style={styles.container}>
            <Text style={styles.appName}>QPGrid</Text>
            <Text style={styles.slogan}>Empowering the Future of Electrical Grids with Quantum Intelligence</Text>
            <View style={styles.featureContainer}>
                <Text style={styles.featureTitle}>Main Features</Text>
                <Text style={styles.featureText}>⚡ Grid Optimization: Optimize electrical grids using cutting-edge quantum algorithms.</Text>
                <Text style={styles.featureText}>🔍 Real-Time Monitoring: Monitor grid performance and renewable energy supply.</Text>
                <Text style={styles.featureText}>📊 Interactive Visualizations: Visualize grid data and optimization processes.</Text>
            </View>
            <View style={styles.buttonContainer}>
                <TouchableOpacity style={styles.button} onPress={() => router.push('/tabs-layout/map')}>
                <Text style={styles.buttonText}>Dashboard</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.button} onPress={() => router.push('/tabs-layout/map')}>
                <Text style={styles.buttonText}>Live Data</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.button} onPress={() => router.push('/tabs-layout/info')}>
                <Text style={styles.buttonText}>Info</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.button} onPress={() => router.push('/tabs-layout/info')}>
                <Text style={styles.buttonText}>About</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#1c1e2b', // Deep blue background
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
    },
    logo: {
        width: 100,
        height: 100,
        marginBottom: 20,
    },
    appName: {
        fontSize: 32,
        fontWeight: 'bold',
        color: '#00ffcc', // Electric green
        marginBottom: 10,
    },
    slogan: {
        fontSize: 16,
        color: '#ffffff', // White text
        textAlign: 'center',
        marginBottom: 30,
    },
    featureContainer: {
        width: '100%',
        alignItems: 'center',
        marginBottom: 30,
    },
    featureTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#00ffcc',
        marginBottom: 10,
    },
    featureText: {
        fontSize: 14,
        color: '#ffffff',
        textAlign: 'center',
        marginVertical: 5,
    },
    buttonContainer: {
        width: '100%',
        alignItems: 'center',
    },
    button: {
        backgroundColor: '#00ffcc',
        paddingVertical: 15,
        paddingHorizontal: 30,
        borderRadius: 5,
        marginVertical: 10,
        width: '80%',
        alignItems: 'center',
    },
    buttonText: {
        fontSize: 18,
        color: '#1c1e2b',
        fontWeight: 'bold',
    },
});

export default HomeScreen;

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';

const WelcomeScreen = () => {
    const router = useRouter();

    const handleStart = () => {
        router.push('/tabs-layout');
    };

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Welcome to QPGrid</Text>
            <Text style={styles.subtitle}>Let's get started with optimizing the electrical grid!</Text>
            <TouchableOpacity style={styles.button} onPress={handleStart}>
                <Text style={styles.buttonText}>Get Started</Text>
            </TouchableOpacity>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
        backgroundColor: '#1c1e2b', 
    },
    title: {
        fontSize: 32,
        fontWeight: 'bold',
        color: '#00ffcc', 
        marginBottom: 10,
    },
    subtitle: {
        fontSize: 18,
        color: '#ffffff', 
        textAlign: 'center',
        marginBottom: 30,
    },
    button: {
        backgroundColor: '#00ffcc', 
        paddingVertical: 15,
        paddingHorizontal: 30,
        borderRadius: 5,
    },
    buttonText: {
        fontSize: 18,
        color: '#1c1e2b', 
        fontWeight: 'bold',
    },
});

export default WelcomeScreen;

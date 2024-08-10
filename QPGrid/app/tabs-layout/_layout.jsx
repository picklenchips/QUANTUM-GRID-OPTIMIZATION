import React from 'react';
import { Text } from 'react-native';
import { Tabs } from 'expo-router';

const TabsLayout = () => {
    return (
        <Tabs>
            <Tabs.Screen
                name="home"
                options={{
                    title: 'Home',
                    tabBarIcon: ({ color, size }) => (
                        <Text style={{ fontSize: size, color }}>🏠</Text>
                    ),
                }}
            />
            <Tabs.Screen
                name="map"
                options={{
                    title: 'Map',
                    tabBarIcon: ({ color, size }) => (
                        <Text style={{ fontSize: size, color }}>🗺️</Text>
                    ),
                }}
            />
            <Tabs.Screen
                name="info"
                options={{
                    title: 'Info',
                    tabBarIcon: ({ color, size }) => (
                        <Text style={{ fontSize: size, color }}>ℹ️</Text>
                    ),
                }}
            />
        </Tabs>
    );
};

export default TabsLayout;

import React from 'react';
import { View, ScrollView, Text, Image, StyleSheet, TouchableOpacity } from 'react-native';

// Component for rendering each section
const Section = ({ title, content, imageUrl }) => (
    <View style={styles.section}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.content}>{content}</Text>
        {imageUrl && <Image source={{ uri: imageUrl }} style={styles.image} />}
    </View>
);

const Info = ({ navigation }) => {
    return (
        <View style={styles.container}>
            <ScrollView contentContainerStyle={styles.scrollContainer}>
                <Section
                    title="Introduction to Climate Change and Power Grid"
                    content="Climate change refers to long-term changes in temperature and weather patterns. The power grid is a complex network of electrical components that supply electricity from producers to consumers."
                />
                <Section
                    title="Impact of Climate Change on the Power Grid"
                    content="Rising temperatures increase electricity demand, especially for cooling. Extreme weather events can damage infrastructure, causing power outages and disruptions."
                    imageUrl="./image/impact.jpg"
                />
                <Section
                    title="Data and Insights"
                    content="Power usage increases during heatwaves, straining the grid. Renewable energy sources like solar and wind are essential for reducing carbon emissions."
                />
                <Section
                    title="Optimization and Technology Solutions"
                    content="Optimization involves making the best use of available resources. Quantum algorithms and machine learning can improve grid management, reduce energy waste, and enhance reliability."
                />
                <Section
                    title="Future Trends and Innovations"
                    content="Smart grids, decentralized energy systems, and emerging technologies offer promising solutions for a more sustainable energy future."
                />
                <Section
                    title="Resources and Further Reading"
                    content="Explore additional resources and join organizations advocating for sustainable energy practices."
                />
            </ScrollView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#1c1e2b', 
    },
    scrollContainer: {
        padding: 20,
    },
    section: {
        marginBottom: 20,
        backgroundColor: '#2a2d3e', 
        padding: 15,
        borderRadius: 10,
        elevation: 3, 
        shadowColor: '#000', 
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 5,
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 5,
        color: '#00ffcc', 
    },
    content: {
        fontSize: 16,
        lineHeight: 24,
        color: '#ffffff',
    },
    image: {
        width: '100%',
        height: 200,
        resizeMode: 'cover',
        marginVertical: 10,
    },
    buttonContainer: {
        padding: 20,
        alignItems: 'center',
    },
    button: {
        backgroundColor: '#00ffcc', 
        paddingVertical: 15,
        paddingHorizontal: 30,
        borderRadius: 5,
        width: '80%',
        alignItems: 'center',
    },
    buttonText: {
        fontSize: 18,
        color: '#1c1e2b', 
        fontWeight: 'bold',
    },
});

export default Info;

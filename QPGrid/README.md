# QPGrid

Work-in-progress app that hosts a map interface + educational notebooks for this project. 

# Prerequisites

Before you begin, ensure you have the following installed:

1. Node.js and npm: Download and install Node.js from [here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm). npm (Node Package Manager) will be installed automatically with Node.js.

2. Expo CLI: Install Expo CLI globally by running the following command in your terminal:

```sh
npm install expo
```

## 🚀 Getting Started

1. Clone the repository (this maybe done already):

```sh
git clone <repository-url> 
cd QPGrid
```

2. Install dependencies:
Within the project directory, run:

```sh
npm install
```

3. Start the development server:
To launch the app, use the following command:

```sh
npx expo start
```

4. Running on your device (this information will appear in the terminal after you run npx expo start):

[On_iOS_or_Android]:

-> Download the Expo Go app from the App Store (for iOS) or the Google Play Store (for Android).

-> Scan the QR code displayed in the terminal or browser when running `npx expo start`. This will load the app on your device using Expo Go.

[On_a_simulator_or_emulator]:

-> If you prefer to run the app on a simulator or emulator, you can choose the appropriate option in the Expo Developer Tools that opens in your browser when you run `npx expo start`.

## Expo Router Example
We use expo-router to build native navigation using files in the app/ directory.

Example
To create a new project with Expo Router, you can use the following command:

```sh
npx create-expo-app -e with-router
```

## 📝 Notes

- [Expo Router: Docs](https://docs.expo.dev/router/introduction/)

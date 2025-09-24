import { NavigationContainer } from "@react-navigation/native";
import SpeechToText from "../../pages/StT/SpeechToText";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

const Routes = () => {
    const Stack = createNativeStackNavigator();
    return (
        <NavigationContainer>
            
            <Stack.Navigator initialRouteName="SpeechToText">
                <Stack.Screen name="SpeechToText" component={SpeechToText} options={{ headerShown: false }} />
                {/* <Stack.Screen name="Recording" component={RecordingScreen} />
                <Stack.Screen name="Results" component={ResultsScreen} /> */}
            </Stack.Navigator>
        </NavigationContainer>

    );
};
export default Routes;
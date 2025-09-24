import { StyleSheet } from "react-native";

const styles = StyleSheet.create({
    container: { flexGrow: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
    heading: { fontSize: 22, fontWeight: 'bold', marginBottom: 20, color: '#333' },
    label: { fontSize: 16, marginBottom: 8, color: '#555' },
    pickerContainer: {
        borderWidth: 1,
        borderColor: '#ccc',
        borderRadius: 5,
        width: '80%',
        marginBottom: 20,
        backgroundColor: 'white'
    },
    picker: { width: '100%' },
    buttonContainer: { marginBottom: 20 },
    resultLabel: { fontSize: 16, fontWeight: 'bold', marginBottom: 8, color: '#333' },
    resultBox: {
        borderWidth: 1,
        borderColor: '#ddd',
        padding: 10,
        borderRadius: 5,
        width: '100%',
        minHeight: 80,
        backgroundColor: 'white'
    },
    resultText: { fontSize: 16, color: '#333' },
    processingContainer: {
        backgroundColor: '#e3f2fd',
        padding: 10,
        borderRadius: 5,
        marginBottom: 20,
    },
    processingText: {
        color: '#1976d2',
        textAlign: 'center',
        fontWeight: 'bold',
    },
    infoContainer: {
        backgroundColor: '#fff3cd',
        padding: 10,
        borderRadius: 5,
        marginTop: 20,
        borderLeftWidth: 4,
        borderLeftColor: '#ffc107',
    },
    infoText: {
        color: '#856404',
        fontSize: 12,
        textAlign: 'center',
    },
    warningContainer: {
        backgroundColor: '#f8d7da',
        padding: 10,
        borderRadius: 5,
        marginBottom: 20,
        borderLeftWidth: 4,
        borderLeftColor: '#dc3545',
    },
    warningText: {
        color: '#721c24',
        fontSize: 12,
        textAlign: 'center',
        fontWeight: 'bold',
    },
    buttonRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '100%',
        marginBottom: 20,
    },
    buttonHalf: {
        flex: 1,
        marginHorizontal: 5,
    },
    loadingContainer: {
        backgroundColor: '#e3f2fd',
        padding: 15,
        borderRadius: 5,
        marginBottom: 20,
        borderLeftWidth: 4,
        borderLeftColor: '#2196f3',
    },
    loadingText: {
        color: '#1976d2',
        textAlign: 'center',
        fontWeight: 'bold',
        fontSize: 14,
    },
    inputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        width: '80%',
        marginBottom: 20,
        gap: 10,
    },
    textInput: {
        flex: 1,
        borderWidth: 1,
        borderColor: '#ccc',
        borderRadius: 5,
        padding: 10,
        backgroundColor: 'white',
        fontSize: 14,
        color: '#333',
    },
});

export default styles;
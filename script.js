let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
context.strokeStyle = '#ffffff';
context.lineWidth = 12;
context.lineCap = 'round';
context.lineJoin = 'round';
let isDrawing = false;
drawingCanvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    context.beginPath();
    context.moveTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
});
drawingCanvas.addEventListener('pointermove', (e) => {
    if (isDrawing) {
        context.lineTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
        context.stroke();
    }
});
drawingCanvas.addEventListener('pointerup', () => {
    isDrawing = false;
    recognizeDraw();
});
async function loadModel() {
    try {
        loading = true;
        model = await tf.loadLayersModel('./tfjs_model/model.json');
        loading = false;
    } catch (error) {
        console.error('Error loading model:', error);
    }
}
loadModel();
function clearAll(event) {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    document.getElementById('predictedDigit').textContent = '—';
    document.getElementById('confidenceScore').textContent = '—';
}
function recognizeDraw(event) {
    if (loading) return;
    if (!model) {
        console.error('Model not loaded');
        return;
    }
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(drawingCanvas, 0, 0, 28, 28);
    const imageData = tempContext.getImageData(0, 0, 28, 28);
    const grayscaleData = new Uint8Array(28 * 28);
    for (let i = 0; i < imageData.data.length; i += 4) {
        const gray = Math.floor((imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3);
        grayscaleData[i / 4] = gray;
    }
    try {
        const tensor = tf.tensor4d(grayscaleData, [1, 28, 28, 1]).toFloat().div(255);
        tf.tidy(() => {
            const predictions = model.predict(tensor);
            const confidence = predictions.dataSync()[predictions.argMax().dataSync()[0]];
            const predictedDigit = predictions.argMax().dataSync()[0];
            document.getElementById('predictedDigit').textContent = predictedDigit.toString();
            document.getElementById('confidenceScore').textContent = (confidence * 100).toFixed(1) + '%';
        });
    } catch (error) {
        console.error('Error making prediction:', error);
    }
}
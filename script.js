let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');

context.strokeStyle = '#ffffff';
context.lineWidth = 12;
context.lineCap = 'round';
context.lineJoin = 'round';

function clearAll() {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

function recognizeDraw() {
    if (loading) return;
    const tensor = preprocessImage(drawingCanvas);
    loading = true;
    model.predict(tensor).then((predictions) => {
        const predicted = predictions.argMax(1);
        const confidence = predictions.max(1);
        predictedDigit.textContent = predicted.toString();
        confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
        loading = false;
    }).catch((error) => {
        console.error('Error during prediction:', error);
        loading = false;
    });
}

function preprocessImage(canvas) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempContext.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        const grayscale = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = grayscale;
    }
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]);
    return tensor.toFloat().div(255);
}

loadModel().then((loadedModel) => {
    model = loadedModel;
}).catch((error) => {
    console.error('Error during model loading:', error);
});
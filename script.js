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

function clearAll(event) {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

async function recognizeDraw(event) {
    if (loading) return;
    if (!model) {
        console.error('Model is not loaded yet.');
        return;
    }
    loading = true;
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
    const tensor = tf.tensor3d(grayscaleData, [28, 28, 1]).reshape([1, 28, 28, 1]).toFloat().div(255);
    try {
        const predictions = await model.predict(tensor);
        const confidence = predictions.max().dataSync()[0];
        const digit = predictions.argMax().dataSync()[0];
        predictedDigit.textContent = digit.toString();
        confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
    } catch (error) {
        console.error('Error in prediction:', error);
    } finally {
        loading = false;
    }
}

drawingCanvas.addEventListener('pointerdown', (event) => {
    context.beginPath();
    context.moveTo(event.clientX - drawingCanvas.getBoundingClientRect().left, event.clientY - drawingCanvas.getBoundingClientRect().top);
});

drawingCanvas.addEventListener('pointermove', (event) => {
    if (event.buttons === 1) {
        context.lineTo(event.clientX - drawingCanvas.getBoundingClientRect().left, event.clientY - drawingCanvas.getBoundingClientRect().top);
        context.stroke();
    }
});

drawingCanvas.addEventListener('pointerup', recognizeDraw);

tf.loadLayersModel('/tfjs_model/model.json').then((loadedModel) => {
    model = loadedModel;
}).catch((error) => {
    console.error('Error loading model:', error);
});
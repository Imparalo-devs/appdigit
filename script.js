let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');

context.strokeStyle = '#ffffff';
context.lineWidth = 12;
context.lineCap = 'round';
context.lineJoin = 'round';

// Load TensorFlow.js model
async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('./tfjs_model/model.json');
    loading = false;
}

// Clear canvas
function clearAll() {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = 'N/A';
    confidenceScore.textContent = 'Confidence score -> 0';
}

// Recognize digit
async function recognizeDraw() {
    if (loading) return;
    const tensor = preprocessImage(drawingCanvas);
    const prediction = await model.predict(tensor);
    const confidence = prediction.dataSync()[0];
    const digit = prediction.argMax().dataSync()[0];
    predictedDigit.textContent = digit;
    confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
}

// Preprocess image
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
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
    return tensor.reshape([1, 28, 28, 1]);
}

// Handle drawing on canvas
drawingCanvas.addEventListener('pointerdown', (e) => {
    context.beginPath();
    context.moveTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
});

drawingCanvas.addEventListener('pointermove', (e) => {
    if (e.buttons === 1) {
        context.lineTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
        context.stroke();
    }
});

drawingCanvas.addEventListener('pointerup', recognizeDraw);

loadModel();
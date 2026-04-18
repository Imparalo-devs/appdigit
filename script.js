let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');

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
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

// Recognize drawn digit
async function recognizeDraw() {
    if (loading) return;
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(drawingCanvas, 0, 0, 28, 28);
    const imageData = ctx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const grayscale = [];
    for (let i = 0; i < pixels.length; i += 4) {
        const gray = pixels[i] * 0.2126 + pixels[i + 1] * 0.7152 + pixels[i + 2] * 0.0722;
        grayscale.push(gray / 255);
    }
    const tensor = tf.tensor2d(grayscale, [28, 28], 'float32');
    const prediction = await model.predict(tensor);
    const probabilities = prediction.dataSync();
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    const confidence = probabilities[maxIndex];
    predictedDigit.textContent = maxIndex.toString();
    confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
}

// Handle pointer events
drawingCanvas.addEventListener('pointerdown', (e) => {
    context.beginPath();
    context.moveTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
});
drawingCanvas.addEventListener('pointermove', (e) => {
    if (e.buttons === 1) {
        context.lineTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
        context.stroke();
        context.closePath();
    }
});

drawingCanvas.addEventListener('pointerup', () => {
    // Do nothing on pointerup event
});

// Load model on page load
loadModel();
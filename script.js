let model;
let loading = false;
let canvas = document.getElementById('drawingCanvas');
let ctx = canvas.getContext('2d');
let predictedDigitLabel = document.getElementById('predictedDigit');
let confidenceScoreLabel = document.getElementById('confidenceScore');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

let drawing = false;
let lastX, lastY;

function clearAll(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigitLabel.textContent = 'Your number is N/A';
    confidenceScoreLabel.textContent = 'Confidence score -> 0';
}

function recognizeDraw(event) {
    if (loading) return;
    loading = true;
    const tensor = preprocessImage(canvas).catch((err) => console.error('Error preprocessing image:', err));
    model.predict(tensor).then(predictions => {
        const predictedDigit = predictions.argMax(-1).dataSync()[0];
        const confidence = predictions.max(-1).dataSync()[0];
        predictedDigitLabel.textContent = `Your number is ${predictedDigit}`;
        confidenceScoreLabel.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
        loading = false;
    });
}

function preprocessImage(canvas) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        const grayscale = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = grayscale;
    }
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]);
    return tensor.toFloat().div(255);
}

async function loadModel() {
    model = await tf.loadLayersModel('./tfjs_model/model.json');
}

loadModel().catch((err) => console.error('Error loading model:', err));

canvas.addEventListener('pointerdown', (e) => {
    drawing = true;
    lastX = e.clientX;
    lastY = e.clientY;
});

canvas.addEventListener('pointermove', (e) => {
    if (drawing && e.buttons === 1) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        lastX = x;
        lastY = y;
    }
});

canvas.addEventListener('pointerup', () => {
    drawing = false;
    recognizeDraw(event);
});
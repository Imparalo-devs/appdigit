import { recognize } from './recognition.js';
let model;
let loading = true; // Add this flag
let canvas = document.getElementById('drawHere');
let ctx = canvas.getContext('2d');
let predictedDigit = document.getElementById('Your number is N/A');
let confidenceScore = document.getElementById('Confidence score -> 0');
let isDrawing = false;

ctx.strokeStyle = '#e5a50a';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function clearAll(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigit.textContent = 'N/A';
    confidenceScore.textContent = '0';
}

function recognizeDraw(event) {
    if (loading) return;
    loading = true;
    const tensor = preprocessImage(canvas);
    recognize(tensor).then((predictions) => {
        const predictedDigitIndex = predictions.argMax(-1);
        const confidence = predictions.max(-1);
        predictedDigit.textContent = predictedDigitIndex.toString();
        confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
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
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
    return tensor;
}

import { loadModel } from './model_loader.js';
loadModel().then(async (loadedModel) => {
    model = loadedModel;
    loading = false;
    document.getElementById('Recognize').disabled = false; // Enable button
}).catch((err) => {
    alert('Failed to load model. Check console for details.');
    console.error('Model load failed:', err);
    document.getElementById('Recognize').disabled = true; // Disable button if model load fails
});
document.getElementById('Recognize').addEventListener('click', recognizeDraw);
document.getElementById('Clear').addEventListener('click', clearAll);
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});
canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseleave', () => isDrawing = false);
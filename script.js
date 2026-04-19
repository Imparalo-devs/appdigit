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

// Load model
async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('./tfjs_model/model.json');
    loading = false;
}

loadModel();

// Clear canvas
function clearAll(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigitLabel.textContent = 'Your number is N/A';
    confidenceScoreLabel.textContent = 'Confidence score -> 0';
}

// Recognize digit
function recognizeDraw(event) {
    if (loading) return;
    const tensor = preprocessCanvasImage(canvas);
    const prediction = await model.predict(tensor);
    const probabilities = await prediction.data();
    const predictedDigit = probabilities.indexOf(Math.max(...probabilities));
    const confidence = Math.max(...probabilities);
    predictedDigitLabel.textContent = `Your number is ${predictedDigit}`;
    confidenceScoreLabel.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
}

// Preprocess canvas image
function preprocessCanvasImage(canvas) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const grayscaleData = new Uint8ClampedArray(28 * 28);
    for (let i = 0; i < imageData.data.length; i += 4) {
        const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
        grayscaleData[i / 4] = gray;
    }
    const tensor = tf.tensor3d(grayscaleData, [28, 28, 1]);
    return tensor.div(255);
}

// Handle pointer events
canvas.addEventListener('pointerdown', (e) => {
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
});

canvas.addEventListener('pointermove', (e) => {
    if (e.buttons === 1) {
        ctx.lineTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
        ctx.stroke();
    }
});
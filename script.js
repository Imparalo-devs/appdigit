let model;
let loading = true;
let drawingCanvas = document.getElementById('drawingCanvas');
let ctx = drawingCanvas.getContext('2d');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Load TensorFlow.js model
async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    loading = false;
}

loadModel();

drawingCanvas.addEventListener('pointerdown', (e) => {
    ctx.beginPath();
    ctx.moveTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
});

drawingCanvas.addEventListener('pointermove', (e) => {
    if (e.buttons === 1) {
        ctx.lineTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
        ctx.stroke();
    }
});

drawingCanvas.addEventListener('pointerup', async () => {
    await recognizeDraw();
});

function clearAll() {
    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

async function recognizeDraw() {
    if (loading) return;
    const tensor = preprocessImage(drawingCanvas);
    tf.tidy(() => {
        const prediction = model.predict(tensor);
        const probabilities = prediction.dataSync();
        const predictedDigitIndex = probabilities.indexOf(Math.max(...probabilities));
        const confidence = probabilities[predictedDigitIndex];
        predictedDigit.textContent = predictedDigitIndex.toString();
        confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
    });
}

function preprocessImage(canvas) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const grayscaleData = new Uint8Array(28 * 28);
    for (let i = 0; i < imageData.data.length; i += 4) {
        const grayscale = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
        grayscaleData[i / 4] = grayscale;
    }
    const tensor = tf.tensor3d(grayscaleData, [28, 28, 1]);
    return tensor.div(255);
}
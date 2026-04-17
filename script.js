let model;
let loading = false;
let canvas = document.getElementById('drawingCanvas');
let ctx = canvas.getContext('2d');
let recognizeBtn = document.getElementById('recognizeBtn');
let clearBtn = document.getElementById('clearBtn');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Load TF.js Model
async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('/tfjs_model/model.json');
    loading = false;
}

// Preprocess Image
function preprocessImage() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = gray;
    }
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).div(255);
    return tensor;
}

// Run Inference
async function runInference(tensor) {
    const predictions = await tf.tidy(() => model.predict(tensor));
    const probabilities = predictions.dataSync();
    const predictedDigitIndex = probabilities.indexOf(Math.max(...probabilities));
    const confidence = Math.max(...probabilities);
    return { predictedDigit: predictedDigitIndex, confidence };
}

// Display Result
function displayResult(predictedDigit, confidence) {
    predictedDigit.textContent = predictedDigit;
    confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
}

// Event Handlers
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

canvas.addEventListener('pointerup', async (e) => {
    if (!loading) {
        const tensor = preprocessImage();
        const { predictedDigit, confidence } = await runInference(tensor);
        displayResult(predictedDigit, confidence);
    }
});

recognizeBtn.addEventListener('click', async () => {
    if (!loading) {
        const tensor = preprocessImage();
        const { predictedDigit, confidence } = await runInference(tensor);
        displayResult(predictedDigit, confidence);
    }
});

clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
});

loadModel();
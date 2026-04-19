let model;
let loading = false;
let canvas = document.getElementById('drawingCanvas');
let ctx = canvas.getContext('2d');
let predictedDigitLabel = document.getElementById('predictedDigit');
let confidenceScoreLabel = document.getElementById('confidenceScore');

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

async function recognizeDraw(event) {
    if (loading) return;
    loading = true;
    const tensor = preprocessImage(canvas);
    model.predict(tensor).then(predictions => { try {
        const predictedDigit = predictions.argMax(-1);
        const confidence = predictions.max(-1);
        predictedDigitLabel.textContent = `Your number is ${predictedDigit.dataSync()[0]}`;
        confidenceScoreLabel.textContent = `Confidence score -> ${(confidence.dataSync()[0] * 100).toFixed(1)}%`;
        loading = false;
    });
}

canvas.addEventListener('pointerdown', (e) => {
    drawing = true;
    lastX = e.clientX - canvas.getBoundingClientRect().left;
    lastY = e.clientY - canvas.getBoundingClientRect().top;
});

canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const x = e.clientX - canvas.getBoundingClientRect().left;
    const y = e.clientY - canvas.getBoundingClientRect().top;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastX = x;
    lastY = y;
});

canvas.addEventListener('pointerup', () => {
    drawing = false;
    recognizeDraw().catch(error => console.error('Error recognizing draw:', error));
});

async function loadModel() {
    model = await tf.loadLayersModel('./tfjs_model/model.json');
}

loadModel().catch(error => console.error('Error loading model:', error));

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
        const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = gray;
    }
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
    return tensor.reshape([1, 28, 28, 1]);
}
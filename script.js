let model;
let loading = true;
let canvas = document.getElementById('drawHere');
let ctx = canvas.getContext('2d');
let predictedDigit = document.getElementById('predicted-digit');
let confidenceScore = document.getElementById('confidence-score');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

let drawing = false;
let lastX, lastY;

function clearAll(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

function recognizeDraw(event) {
    if (loading) return;
    loading = true;
    const tensor = preprocessImage(canvas);
    model.predict(tensor).then(predictions => {
        const predicted = predictions.argMax(-1);
        const confidence = predictions.max(-1);
        predictedDigit.textContent = predicted.toString();
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
    const grayscale = [];
    for (let i = 0; i < pixels.length; i += 4) {
        const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscale.push(gray / 255);
    }
    return tf.tensor4d(grayscale, [1, 28, 28, 1]);
}

async function loadModel() {
    model = await tf.loadLayersModel('./tfjs_model/model.json');
    loading = false;
}

loadModel().catch((error) => console.error('Error loading model:', error));

canvas.addEventListener('pointerdown', (e) => {
    drawing = true;
    lastX = e.clientX;
    lastY = e.clientY;
});

canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastX = x;
    lastY = y;
});

canvas.addEventListener('pointerup', () => {
    drawing = false;
    recognizeDraw(event);
});
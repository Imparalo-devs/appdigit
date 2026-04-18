let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let ctx = drawingCanvas.getContext('2d');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');
let confidenceScore = document.getElementById('confidenceScore');
let predictedDigit = document.getElementById('predictedDigit');
let drawing = false;
let lastX, lastY;

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Load TensorFlow.js model
async async function loadModel() { try {
    loading = true;
    model = await loadModelKeras3('./tfjs_model/model.json');
    loading = false;
}

loadModel().catch((error) => console.error('Error loading model:', error));

drawingCanvas.addEventListener('pointerdown', (e) => {
    drawing = true;
    lastX = e.clientX - drawingCanvas.getBoundingClientRect().left;
    lastY = e.clientY - drawingCanvas.getBoundingClientRect().top;
});

drawingCanvas.addEventListener('pointermove', (e) => {
    if (drawing) {
        let x = e.clientX - drawingCanvas.getBoundingClientRect().left;
        let y = e.clientY - drawingCanvas.getBoundingClientRect().top;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        lastX = x;
        lastY = y;
    }
});

drawingCanvas.addEventListener('pointerup', () => {
    drawing = false;
    recognizeDraw();
});

function clearAll() {
    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

async function recognizeDraw() {
    if (loading) return;
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(drawingCanvas, 0, 0, 28, 28);
    let imageData = tempCtx.getImageData(0, 0, 28, 28);
    let grayscale = new Uint8Array(28 * 28);
    for (let i = 0; i < imageData.data.length; i += 4) {
        let gray = Math.floor((imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3);
        grayscale[i / 4] = gray;
    }
    let tensor = tf.tensor4d(grayscale, [1, 28, 28, 1]);
    tensor = tensor.div(255);
    tf.tidy(() => {
        let predictions = model.predict(tensor);
        let confidence = predictions.dataSync()[0];
        let digit = predictions.argMax().dataSync()[0];
        confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
        predictedDigit.textContent = `Your number is ${digit}`;
    });
}
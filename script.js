let model;
let loading = false;
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

let drawing = false;
let lastX, lastY;

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

async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('/tfjs_model/model.json');
    loading = false;
}

loadModel();

function clearAll() {
    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = 'N/A';
    confidenceScore.textContent = 'Confidence score -> 0';
}

function recognizeDraw() {
    if (loading) return;
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(drawingCanvas, 0, 0, 28, 28);
    let imageData = tempCtx.getImageData(0, 0, 28, 28);
    let pixels = imageData.data;
    let grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        let gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = gray;
    }
    let tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
    tf.tidy(() => {
        let predictions = model.predict(tensor);
        let confidence = predictions.max().dataSync()[0];
        let digit = predictions.argMax().dataSync()[0];
        predictedDigit.textContent = digit.toString();
        confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
    });
}
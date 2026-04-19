let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');
let confidenceScore = document.getElementById('confidenceScore');
let predictedDigit = document.getElementById('predictedDigit');

context.strokeStyle = '#ffffff';
context.lineWidth = 12;
context.lineCap = 'round';
context.lineJoin = 'round';

// Load TF.js Model
async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('/tfjs_model/model.json');
    loading = false;
}

loadModel();

drawingCanvas.addEventListener('pointerdown', (e) => {
    context.beginPath();
    context.moveTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
});

drawingCanvas.addEventListener('pointermove', (e) => {
    if (e.buttons === 1) {
        context.lineTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
        context.stroke();
    }
});

drawingCanvas.addEventListener('pointerup', (e) => recognizeDraw(e));

clearBtn.addEventListener('click', clearAll);

function clearAll(event) {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

function recognizeDraw(event) {
    if (loading) return;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(drawingCanvas, 0, 0, 28, 28);
    const imageData = tempContext.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = gray;
    }
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
    tf.tidy(() => {
        try { const output = model.predict(tensor); } catch (error) { console.error('Error predicting output:', error); }
        if (output) { const probabilities = output.dataSync(); } else { console.error('No output from model prediction'); }
        const predicted = probabilities.indexOf(Math.max(...probabilities));
        const confidence = Math.max(...probabilities);
        predictedDigit.textContent = predicted.toString();
        confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
    });
}
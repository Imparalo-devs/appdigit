let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let ctx = drawingCanvas.getContext('2d');
let predictedDigitLabel = document.getElementById('predictedDigitLabel');
let confidenceScoreLabel = document.getElementById('confidenceScoreLabel');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

let isDrawing = false;
drawingCanvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
});
drawingCanvas.addEventListener('pointermove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
        ctx.stroke();
    }
});
drawingCanvas.addEventListener('pointerup', () => {
    isDrawing = false;
    recognizeDraw();
});

async function loadModel() {
    try {
        loading = true;
        model = await tf.loadLayersModel('tfjs_model/model.json');
        loading = false;
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

loadModel();

function clearAll(event) {
    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigitLabel.textContent = 'Your number is N/A';
    confidenceScoreLabel.textContent = 'Confidence score -> 0';
}

async function recognizeDraw(event) {
    if (loading) return;
    if (!model) {
        console.error('Model not loaded');
        return;
    }
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(drawingCanvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const grayscaleData = new Uint8Array(28 * 28);
    for (let i = 0; i < imageData.data.length; i += 4) {
        const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
        grayscaleData[i / 4] = gray;
    }
    const tensor = tf.tensor3d(grayscaleData, [28, 28, 1]).toFloat().div(255);
    try {
        tf.tidy(() => {
            const predictions = model.predict(tensor);
            const confidence = predictions.dataSync()[0];
            const predictedDigit = predictions.argMax(1).dataSync()[0];
            predictedDigitLabel.textContent = `Your number is ${predictedDigit}`;
            confidenceScoreLabel.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
        });
    } catch (error) {
        console.error('Error making prediction:', error);
    }
}
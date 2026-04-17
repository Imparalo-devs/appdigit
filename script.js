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

function clearAll() {
    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = 'N/A';
    confidenceScore.textContent = 'Confidence score -> 0';
}

function recognizeDraw() {
    if (loading) return;
    loading = true;
    const tensor = preprocessImage(drawingCanvas);
    model.predict(tensor).then((predictions) => {
        const predictedDigitValue = predictions.argMax(1).dataSync()[0];
        const confidence = predictions.max(1).dataSync()[0];
        predictedDigit.textContent = predictedDigitValue.toString();
        confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
        loading = false;
    });
}

function loadModel() {
    return tf.loadLayersModel('/tfjs_model/model.json');
}

loadModel().then((loadedModel) => {
    model = loadedModel;
});

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

drawingCanvas.addEventListener('pointerup', recognizeDraw);
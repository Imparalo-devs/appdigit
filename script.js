let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');

context.strokeStyle = '#ffffff';
context.lineWidth = 12;
context.lineCap = 'round';
context.lineJoin = 'round';

let isDrawing = false;
let lastX, lastY;

drawingCanvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    lastX = e.clientX - drawingCanvas.getBoundingClientRect().left;
    lastY = e.clientY - drawingCanvas.getBoundingClientRect().top;
});

drawingCanvas.addEventListener('pointermove', (e) => {
    if (isDrawing && e.buttons === 1) {
        let x = e.clientX - drawingCanvas.getBoundingClientRect().left;
        let y = e.clientY - drawingCanvas.getBoundingClientRect().top;
        context.beginPath();
        context.moveTo(lastX, lastY);
        context.lineTo(x, y);
        context.stroke();
        lastX = x;
        lastY = y;
    }
});

drawingCanvas.addEventListener('pointerup', () => {
    isDrawing = false;
    recognizeDraw();
});

async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('/tfjs_model/model.json');
    loading = false;
}

loadModel().catch((err) => console.error('Error loading model:', err));

function clearAll() {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = 'Your number is N/A';
    confidenceScore.textContent = 'Confidence score -> 0';
}

function recognizeDraw() {
    if (loading) return;
    let tensor = tf.browser.fromPixels(drawingCanvas, 1);
    tensor = tf.image.resizeBilinear(tensor, [28, 28]);
    tensor = tf.cast(tensor, 'float32');
    tensor = tf.div(tensor, 255);
    tensor = tf.expandDims(tensor, 0);
    tensor = tf.expandDims(tensor, -1);
    tf.tidy(() => {
        let predictions = model.predict(tensor).catch((err) => console.error('Error making prediction:', err));
        let scores = predictions.dataSync() || [];
        let maxScore = Math.max(...scores);
        let digit = scores.indexOf(maxScore);
        predictedDigit.textContent = `Your number is ${digit}`;
        confidenceScore.textContent = `Confidence score -> ${(maxScore * 100).toFixed(1)}%`;
    });
}
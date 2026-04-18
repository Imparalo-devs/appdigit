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

// Load TensorFlow.js model
async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('../tfjs_model/model.json');
    loading = false;
}

// Clear canvas and reset predicted digit and confidence score
function clearAll(event) {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

// Recognize drawn digit
async function recognizeDraw(event) {
    if (loading) return;
    try {
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(drawingCanvas, 0, 0, 28, 28);
        const imageData = ctx.getImageData(0, 0, 28, 28);
        const pixels = imageData.data;
        const grayscalePixels = new Uint8Array(28 * 28);
        for (let i = 0; i < pixels.length; i += 4) {
            const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
            grayscalePixels[i / 4] = gray;
        }
        const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
        tf.tidy(() => {
            const output = model.predict(tensor);
            const probabilities = output.dataSync();
            const predicted = probabilities.indexOf(Math.max(...probabilities));
            const confidence = Math.max(...probabilities);
            predictedDigit.textContent = predicted.toString();
            confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
        });
    } catch (error) {
        console.error('Error recognizing drawn digit:', error);
    }
}

// Handle pointer events
drawingCanvas.addEventListener('pointerdown', (event) => {
    context.beginPath();
    context.moveTo(event.clientX - drawingCanvas.getBoundingClientRect().left, event.clientY - drawingCanvas.getBoundingClientRect().top);
});

drawingCanvas.addEventListener('pointermove', (event) => {
    if (event.buttons === 1) {
        context.lineTo(event.clientX - drawingCanvas.getBoundingClientRect().left, event.clientY - drawingCanvas.getBoundingClientRect().top);
        context.stroke();
    }
});

drawingCanvas.addEventListener('pointerup', async (event) => {
    await recognizeDraw(event);
});

loadModel();
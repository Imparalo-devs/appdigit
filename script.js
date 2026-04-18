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
    try {
        loading = true;
        model = await tf.loadLayersModel('/tfjs_model/model.json');
        loading = false;
    } catch (error) {
        console.error('Error loading model:', error);
    }
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

drawingCanvas.addEventListener('pointerup', () => {
    recognizeDraw();
});

function clearAll(event) {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

function recognizeDraw(event) {
    if (loading) return;
    try {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempContext = tempCanvas.getContext('2d');
        tempContext.drawImage(drawingCanvas, 0, 0, 28, 28);
        const imageData = tempContext.getImageData(0, 0, 28, 28);
        const grayscaleData = new Uint8Array(28 * 28);
        for (let i = 0; i < imageData.data.length; i += 4) {
            const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
            grayscaleData[i / 4] = gray;
        }
        const tensor = tf.tensor4d(grayscaleData, [1, 28, 28, 1]).toFloat().div(255);
        tf.tidy(() => {
            const predictions = model.predict(tensor);
            const confidence = predictions.dataSync()[predictions.argMax().dataSync()[0]];
            const digit = predictions.argMax().dataSync()[0];
            predictedDigit.textContent = digit.toString();
            confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
        });
    } catch (error) {
        console.error('Error recognizing draw:', error);
    }
}
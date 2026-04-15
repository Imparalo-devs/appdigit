let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');

// Load TensorFlow.js model
async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('/tfjs_model/model.json');
    loading = false;
}

// Clear canvas and reset predicted digit and confidence score
function clearAll() {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = 'N/A';
    confidenceScore.textContent = 'Confidence score -> 0';
}

// Recognize drawn digit
async function recognizeDraw() {
    if (loading) return;
    // Get pixel data from canvas
    const pixels = context.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height).data;
    // Resize to 28x28 and convert to grayscale
    const resizedPixels = tf.tidy(() => {
        const tensor = tf.tensor3d(pixels, [drawingCanvas.height, drawingCanvas.width, 4], 'float32');
        const resizedTensor = tf.image.resizeBilinear(tensor, [28, 28]);
        const grayscaleTensor = tf.mean(resizedTensor, 2);
        return grayscaleTensor;
    });
    // Normalize to [0,1]
    const normalizedTensor = tf.div(resizedPixels, 255);
    // Reshape to tensor [1,28,28,1]
    const inputTensor = normalizedTensor.reshape([1, 28, 28, 1]);
    // Run inference
    const output = await model.predict(inputTensor);
    const probabilities = await output.data();
    const predictedDigitIndex = probabilities.indexOf(Math.max(...probabilities));
    const confidence = Math.max(...probabilities);
    // Update predicted digit and confidence score
    predictedDigit.textContent = predictedDigitIndex.toString();
    confidenceScore.textContent = `Confidence score -> ${confidence.toFixed(2) * 100}%`;
}

// Load model on page load
loadModel();

// Handle pointer events on canvas
drawingCanvas.addEventListener('pointerdown', (e) => {
    context.beginPath();
    context.moveTo(e.clientX - drawingCanvas.offsetLeft, e.clientY - drawingCanvas.offsetTop);
});

drawingCanvas.addEventListener('pointermove', (e) => {
    context.lineTo(e.clientX - drawingCanvas.offsetLeft, e.clientY - drawingCanvas.offsetTop);
    context.stroke();
});

drawingCanvas.addEventListener('pointerup', recognizeDraw);
let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let ctx = drawingCanvas.getContext('2d');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Load TensorFlow.js model
async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    loading = false;
}

// Clear canvas and reset predicted digit and confidence score
function clearAll(event) {
    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = 'N/A';
    confidenceScore.textContent = 'Confidence score -> 0';
}

// Recognize drawn digit
async function recognizeDraw(event) {
    if (loading) return;
    loading = true;
    // Preprocess image
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(drawingCanvas, 0, 0, 28, 28);
    let imageData = tempCtx.getImageData(0, 0, 28, 28);
    let grayscaleData = new Uint8Array(28 * 28);
    for (let i = 0; i < imageData.data.length; i += 4) {
        let gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
        grayscaleData[i / 4] = gray;
    }
    let tensor = tf.tensor4d(grayscaleData, [1, 28, 28, 1]).toFloat().div(255);
    // Run inference
    let predictions = await tf.tidy(() => model.predict(tensor));
    let confidence = predictions.max().dataSync()[0];
    let digit = predictions.argMax().dataSync()[0];
    // Display result
    predictedDigit.textContent = digit.toString();
    confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
    loading = false;
}

// Handle user input
drawingCanvas.addEventListener('pointerdown', (event) => {
    ctx.beginPath();
    ctx.moveTo(event.clientX - drawingCanvas.getBoundingClientRect().left, event.clientY - drawingCanvas.getBoundingClientRect().top);
});
drawingCanvas.addEventListener('pointermove', (event) => {
    if (event.buttons === 1) {
        ctx.lineTo(event.clientX - drawingCanvas.getBoundingClientRect().left, event.clientY - drawingCanvas.getBoundingClientRect().top);
        ctx.stroke();
    }
});
drawingCanvas.addEventListener('pointerup', recognizeDraw);

loadModel();
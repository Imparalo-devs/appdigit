let model;
let loading = false;
let canvas = document.getElementById('drawHere');
let ctx = canvas.getContext('2d');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');
let clearButton = document.getElementById('clear');
let recognizeButton = document.getElementById('recognize');

// Initialize canvas context
ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Load TensorFlow.js model
async function loadModel() {
    model = await tf.loadLayersModel('./tfjs_model/model.json');
    loading = false;
}

// Clear canvas
function clearAll(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

// Recognize drawn digit
async function recognizeDraw() {
    if (loading) return;
    loading = true;
    // Preprocess image
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    let imageData = tempCtx.getImageData(0, 0, 28, 28);
    let grayscaleData = new Uint8Array(28 * 28);
    for (let i = 0; i < imageData.data.length; i += 4) {
        let gray = Math.floor((imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3);
        grayscaleData[i / 4] = gray;
    }
    let tensor = tf.tensor4d(grayscaleData, [1, 28, 28, 1]);
    tensor = tensor.div(255);
    // Run inference
    let predictions = await tf.tidy(() => model.predict(tensor));
    let probabilities = await predictions.data();
    let maxProbability = Math.max(...probabilities);
    let predictedDigitIndex = probabilities.indexOf(maxProbability);
    predictedDigit.textContent = predictedDigitIndex.toString();
    confidenceScore.textContent = (maxProbability * 100).toFixed(1) + '%';
    loading = false;
}

// Load model on page load
await try { await loadModel(); } catch (error) { console.error('Error loading model:', error); }

// Handle user drawing on canvas
canvas.addEventListener('pointerdown', (event) => {
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.getBoundingClientRect().left, event.clientY - canvas.getBoundingClientRect().top);
});

canvas.addEventListener('pointermove', (event) => {
    if (event.buttons === 1) {
        ctx.lineTo(event.clientX - canvas.getBoundingClientRect().left, event.clientY - canvas.getBoundingClientRect().top);
        ctx.stroke();
    }
});

canvas.addEventListener('pointerup', recognizeDraw);
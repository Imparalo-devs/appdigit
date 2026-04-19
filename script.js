let model;
let loading = false;
let canvas = document.getElementById('drawHere');
let ctx = canvas.getContext('2d');
let predictedDigitLabel = document.getElementById('predictedDigitLabel');
let confidenceScoreLabel = document.getElementById('confidenceScoreLabel');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Load TensorFlow.js model
async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('./tfjs_model/model.json');
    loading = false;
}

loadModel();

// Clear canvas
function clearAll(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigitLabel.textContent = 'Your number is N/A';
    confidenceScoreLabel.textContent = 'Confidence score -> 0';
}

// Recognize drawn digit
function recognizeDraw(event) {
    if (loading) return;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = gray;
    }
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).reshape([1, 28, 28, 1]).toFloat().div(255);
    tf.tidy(() => {
        try { const output = await model.predict(tensor); } catch (error) { console.error(error); }
        const probabilities = (await output.data());
        const predictedDigit = probabilities.indexOf(Math.max(...(probabilities || [])));
        const confidence = Math.max(...probabilities);
        predictedDigitLabel.textContent = `Your number is ${predictedDigit}`;
        confidenceScoreLabel.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
    });
}

// Draw on canvas
canvas.addEventListener('pointerdown', (e) => {
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
});

canvas.addEventListener('pointermove', (e) => {
    if (e.buttons === 1) {
        ctx.lineTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
        ctx.stroke();
    }
});

canvas.addEventListener('pointerup', async (e) => { await recognizeDraw(e); });
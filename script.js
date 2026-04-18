let model;
let loading = false;
let canvas = document.getElementById('drawingCanvas');
let ctx = canvas.getContext('2d');
let drawing = false;
let lastX, lastY;

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Load TensorFlow.js model
async function loadModel() {
    try {
        loading = true;
        model = await tf.loadLayersModel('./tfjs_model/model.json');
        loading = false;
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Clear canvas
function clearAll(event) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictedDigit').textContent = '—';
    document.getElementById('confidenceScore').textContent = '—';
}

// Recognize drawn digit
function recognizeDraw(event) {
    if (loading) return;
    const tensor = preprocessImage(canvas);
    model.predict(tensor).then(predictions => {
        try {
            const confidence = predictions.max().dataSync()[0];
            const digit = predictions.argMax().dataSync()[0];
            document.getElementById('predictedDigit').textContent = `Your number is ${digit}`;
            document.getElementById('confidenceScore').textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
        } catch (error) {
            console.error('Error processing predictions:', error);
        }
    }).catch(error => {
        console.error('Error making prediction:', error);
    });
}

// Preprocess image
function preprocessImage(canvas) {
    try {
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
        const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
        return tensor.reshape([1, 28, 28, 1]);
    } catch (error) {
        console.error('Error preprocessing image:', error);
        return null;
    }
}

// Handle pointer events
canvas.addEventListener('pointerdown', (event) => {
    drawing = true;
    lastX = event.clientX;
    lastY = event.clientY;
});

canvas.addEventListener('pointermove', (event) => {
    if (drawing) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        lastX = x;
        lastY = y;
    }
});

canvas.addEventListener('pointerup', () => {
    drawing = false;
    recognizeDraw();
});

loadModel().then(() => {
    console.log('Model loaded');
});
let model;
let context;
let canvas;
let isDrawing = false;
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');

// Load TensorFlow.js model
async function loadModel() {
    model = await loadModelKeras3('./tfjs_model/model.json');
}

// Initialize canvas context
function initCanvas() {
    canvas = document.getElementById('drawingCanvas');
    context = canvas.getContext('2d');
    context.strokeStyle = '#ffffff';
    context.lineWidth = 12;
    context.lineCap = 'round';
    context.lineJoin = 'round';
}

// Handle pointer down event
function handlePointerDown(event) {
    isDrawing = true;
    context.beginPath();
    context.moveTo(event.clientX - canvas.getBoundingClientRect().left, event.clientY - canvas.getBoundingClientRect().top);
}

// Handle pointer move event
function handlePointerMove(event) {
    if (isDrawing) {
        context.lineTo(event.clientX - canvas.getBoundingClientRect().left, event.clientY - canvas.getBoundingClientRect().top);
        context.stroke();
    }
}

// Handle pointer up event
function handlePointerUp(event) {
    isDrawing = false;
    recognizeDraw(event);
}

// Clear canvas
function clearAll(event) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

// Recognize drawn digit
async function recognizeDraw(event) {
    try {
        // Preprocess image
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempContext = tempCanvas.getContext('2d');
        tempContext.drawImage(canvas, 0, 0, 28, 28);
        const imageData = tempContext.getImageData(0, 0, 28, 28);
        const grayscaleData = new Uint8Array(28 * 28);
        for (let i = 0; i < imageData.data.length; i += 4) {
            const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
            grayscaleData[i / 4] = gray;
        }
        const tensor = tf.tensor3d(grayscaleData, [28, 28, 1]).toFloat().div(255);
        
        // Run inference
        const predictions = await model.predict(tensor);
        const confidence = predictions.dataSync()[0];
        const digit = predictions.argMax().dataSync()[0];
        
        // Display result
        predictedDigit.textContent = `Your number is ${digit}`;
        confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
    } catch (error) {
        console.error('Error recognizing drawn digit:', error);
        predictedDigit.textContent = 'Error';
        confidenceScore.textContent = '—';
    }
}

// Initialize
loadModel();
initCanvas();

// Add event listeners
canvas.addEventListener('pointerdown', handlePointerDown);
canvas.addEventListener('pointermove', handlePointerMove);
canvas.addEventListener('pointerup', handlePointerUp);
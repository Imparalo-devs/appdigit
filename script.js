let model;
let loading = false;
let drawingCanvas = document.getElementById('drawHere');
let context = drawingCanvas.getContext('2d');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');

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
    if (isDrawing) {
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

loadModel();

function clearAll() {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = 'Your number is N/A';
    confidenceScore.textContent = 'Confidence score -> 0';
}

function recognizeDraw() {
    if (loading) return;
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    let tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(drawingCanvas, 0, 0, 28, 28);
    let imageData = tempContext.getImageData(0, 0, 28, 28);
    let pixels = imageData.data;
    let grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        let gray = Math.floor((pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3);
        grayscalePixels[i / 4] = gray;
    }
    let tensor = tf.tensor(grayscalePixels, [28, 28, 1], 'float32').div(255);
    let predictions = model.predict(tensor);
    let confidence = tf.softmax(predictions).dataSync()[0];
    let digit = predictions.argMax().dataSync()[0];
    predictedDigit.textContent = `Your number is ${digit}`;
    confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
}

// Add a debouncing mechanism to prevent excessive calls to the recognizeDraw function
let timeoutId;
function recognizeDrawDebounced() {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(recognizeDraw, 500);
}
drawingCanvas.addEventListener('pointerup', recognizeDrawDebounced);
```

Note that this is just one possible way to improve the code. There are many other ways to optimize the code, and the best approach will depend on the specific requirements of the project.
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
let isDrawing = false;
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');

context.strokeStyle = '#ffffff';
context.lineWidth = 12;
context.lineCap = 'round';
context.lineJoin = 'round';

function clearAll() {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = 'N/A';
    confidenceScore.textContent = 'Confidence score -> 0';
}

function recognizeDraw() {
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    let tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(drawingCanvas, 0, 0, 28, 28);
    let imageData = tempContext.getImageData(0, 0, 28, 28);
    let grayscaleData = new Uint8Array(28 * 28);
    for (let i = 0; i < imageData.data.length; i += 4) {
        let gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
        grayscaleData[i / 4] = gray;
    }
    let tensor = tf.tensor3d(grayscaleData, [28, 28, 1], 'uint8').toFloat().div(255);
    loadModel().then(model => {
        tf.tidy(() => {
            let predictions = model.predict(tensor);
            let confidence = predictions.max().dataSync()[0];
            let digit = predictions.argMax().dataSync()[0];
            predictedDigit.textContent = digit.toString();
            confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
        });
    });
}

drawingCanvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    let rect = drawingCanvas.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    context.beginPath();
    context.moveTo(x, y);
});

drawingCanvas.addEventListener('pointermove', (e) => {
    if (isDrawing) {
        let rect = drawingCanvas.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        context.lineTo(x, y);
        context.stroke();
    }
});

drawingCanvas.addEventListener('pointerup', () => {
    isDrawing = false;
    recognizeDraw();
});
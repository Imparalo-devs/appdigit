let model;
let loading = true;
let drawingCanvas = document.getElementById('drawingCanvas');
let ctx = drawingCanvas.getContext('2d');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

let isDrawing = false;
drawingCanvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
});
drawingCanvas.addEventListener('pointermove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
        ctx.stroke();
    }
});
drawingCanvas.addEventListener('pointerup', () => {
    isDrawing = false;
    recognizeDraw();
});

loadModelKeras3('./tfjs_model/model.json').then((loadedModel) => {
    model = loadedModel;
    loading = false;
});

function clearAll(event) {
    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

function recognizeDraw(event) {
    if (loading) return;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(drawingCanvas, 0, 0, tempCanvas.width, tempCanvas.height);
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const pixels = imageData.data;
    const grayscalePixels = new Uint8Array(pixels.length / 4);
    for (let i = 0; i < pixels.length; i += 4) {
        const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = gray;
    }
    const tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).reshape([1, 28, 28, 1]).toFloat().div(255);
    await model.predict(tensor).then((predictions) => {
        const confidence = predictions.dataSync()[0];
        const digit = predictions.argMax().dataSync()[0];
        predictedDigit.textContent = digit.toString();
        confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
    });
}
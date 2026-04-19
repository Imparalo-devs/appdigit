let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
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
drawingCanvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    context.beginPath();
    context.moveTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
});
drawingCanvas.addEventListener('pointermove', (e) => {
    if (isDrawing) {
        context.lineTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
        context.stroke();
    }
});
drawingCanvas.addEventListener('pointerup', () => {
    isDrawing = false;
    recognizeDraw();
});

async function loadModel() {
    loading = true;
    model = await loadModelKeras3('./tfjs_model/model.json');
    loading = false;
}

loadModel();

function clearAll() {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = 'N/A';
    confidenceScore.textContent = '0';
}

function recognizeDraw() {
    if (loading) return;
    let tensor = tf.tidy(() => {
        let canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        let ctx = canvas.getContext('2d');
        ctx.drawImage(drawingCanvas, 0, 0, 28, 28);
        let imageData = ctx.getImageData(0, 0, 28, 28);
        let grayscale = new Uint8Array(28 * 28);
        for (let i = 0; i < 28 * 28; i++) {
            let index = i * 4;
            let gray = (imageData.data[index] + imageData.data[index + 1] + imageData.data[index + 2]) / 3;
            grayscale[i] = gray;
        }
        let tensor = tf.tensor3d(grayscale, [28, 28, 1]);
        tensor = tensor.toFloat().div(255);
        return tensor;
    });
    await model.predict(tensor).then((predictions) => {
        let confidence = predictions.dataSync()[0];
        let digit = predictions.argMax().dataSync()[0];
        predictedDigit.textContent = digit;
        confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
    });
}
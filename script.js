async function loadModelKeras3(url) {
  const handler = tf.io.fileSystem(url);
  return tf.loadLayersModel(handler);
}

let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let ctx = drawingCanvas.getContext('2d');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');
let confidenceScore = document.getElementById('confidenceScore');
let predictedDigit = document.getElementById('predictedDigit');

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
    let tensor = tf.tensor4d(grayscaleData, [1, 28, 28, 1]);
    tensor = tensor.div(255);
    model.predict(tensor).then(async (predictions) => {
        let confidence = predictions.dataSync()[0];
        let digit = predictions.argMax().dataSync()[0];
        predictedDigit.textContent = digit.toString();
        confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
    });
}
let model;
let loading = false;
let canvas = document.getElementById('draw-here');
let ctx = canvas.getContext('2d');
let predictedDigit = document.getElementById('predicted-digit');
let confidenceScore = document.getElementById('confidence-score');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

let drawing = false;
let lastX, lastY;

canvas.addEventListener('pointerdown', (e) => {
    drawing = true;
    lastX = e.clientX;
    lastY = e.clientY;
});

canvas.addEventListener('pointerup', (e) => {
    drawing = false;
    recognizeDraw();
});

canvas.addEventListener('pointermove', (e) => {
    if (drawing) {
        let rect = canvas.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        lastX = x;
        lastY = y;
    }
});

async function loadModel() {
    loading = true;
    model = await tf.loadLayersModel('./tfjs_model/model.json');
    loading = false;
}

loadModel();

function clearAll() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

function recognizeDraw() {
    if (loading) return;
    let tensor = tf.browser.fromPixels(canvas);
    tensor = tf.image.resizeBilinear(tensor, [28, 28]);
    tensor = tf.cast(tensor, 'float32');
    tensor = tf.div(tensor, 255);
    tensor = tf.expandDims(tensor, 0);
    tensor = tf.expandDims(tensor, -1);
    let predictions = await model.predict(tensor);
    let confidence = (await predictions.data())[0];
    let digit = (await predictions.argMax(1).data())[0];
    predictedDigit.textContent = digit.toString();
    confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
}
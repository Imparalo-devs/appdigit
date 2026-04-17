let canvas = document.getElementById('drawHere');
let ctx = canvas.getContext('2d');
let drawing = false;
let lastX, lastY;

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function clearAll() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictedDigit').innerText = '—';
    document.getElementById('confidenceScore').innerText = '—';
}

function recognizeDraw() {
    if (loading) return;
    let tensor = preprocessImage(ctx);
    model.predict(tensor).then(prediction => {
        let predictedDigit = prediction.argMax(1).dataSync()[0];
        let confidence = prediction.max(1).dataSync()[0] * 100;
        document.getElementById('predictedDigit').innerText = predictedDigit.toString();
        document.getElementById('confidenceScore').innerText = `Confidence score -> ${confidence.toFixed(1)}%`;
    });
}

function preprocessImage(ctx) {
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    let imageData = tempCtx.getImageData(0, 0, 28, 28);
    let pixels = imageData.data;
    let grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        let grayscale = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscalePixels[i / 4] = grayscale;
    }
    let tensor = tf.tensor3d(grayscalePixels, [28, 28, 1]).toFloat().div(255);
    return tensor;
}

canvas.addEventListener('pointerdown', (e) => {
    drawing = true;
    lastX = e.clientX - canvas.getBoundingClientRect().left;
    lastY = e.clientY - canvas.getBoundingClientRect().top;
});

canvas.addEventListener('pointermove', (e) => {
    if (drawing && e.buttons === 1) {
        let x = e.clientX - canvas.getBoundingClientRect().left;
        let y = e.clientY - canvas.getBoundingClientRect().top;
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
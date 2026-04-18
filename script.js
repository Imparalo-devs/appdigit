let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
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
    document.getElementById('predictedDigit').textContent = '—';
    document.getElementById('confidenceScore').textContent = '—';
}
function recognizeDraw() {
    if (loading) return;
    const tensor = tf.tidy(() => {
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(drawingCanvas, 0, 0, 28, 28);
        const imageData = ctx.getImageData(0, 0, 28, 28);
        const data = new Uint8Array(28 * 28);
        for (let i = 0; i < 28 * 28; i++) {
            data[i] = imageData.data[i * 4];
        }
        return tf.tensor4d(data, [1, 28, 28, 1]).toFloat().div(255);
    });
    model.predict(tensor).then((predictions) => {
        try {
        const confidence = predictions.dataSync()[0];
        const digit = predictions.argMax().dataSync()[0];
        document.getElementById('predictedDigit').textContent = digit.toString();
        document.getElementById('confidenceScore').textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
        } catch (error) {
          console.error('Error in recognizeDraw:', error);
        }
    });
}
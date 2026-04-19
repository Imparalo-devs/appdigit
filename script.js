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

function clearAll(event) {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

function recognizeDraw(event) {
    if (loading) return;
    loading = true;
    const tensor = preprocessImage(drawingCanvas);
    model.predict(tensor).then((predictions) => {
        try {
          const predicted = predictions.argMax(1);
          const confidence = predictions.max(1);
          predictedDigit.textContent = predicted.toString();
          confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
          loading = false;
        } catch (error) {
          console.error(error);
        }
      });
        try {
        const predicted = predictions.argMax(1);
        const confidence = predictions.max(1);
        predictedDigit.textContent = predicted.toString();
        confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
        loading = false;
    });
}

drawingCanvas.addEventListener('pointerdown', (event) => {
    context.beginPath();
    context.moveTo(event.clientX - drawingCanvas.getBoundingClientRect().left, event.clientY - drawingCanvas.getBoundingClientRect().top);
});

drawingCanvas.addEventListener('pointermove', (event) => {
    if (event.buttons === 1) {
        context.lineTo(event.clientX - drawingCanvas.getBoundingClientRect().left, event.clientY - drawingCanvas.getBoundingClientRect().top);
        context.stroke();
    }
});

drawingCanvas.addEventListener('pointerup', () => recognizeDraw(event));

loadModelKeras3('./tfjs_model/model.json').then((loadedModel) => {
    model = loadedModel;
});

function preprocessImage(canvas) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempContext.getImageData(0, 0, 28, 28);
    const pixels = imageData.data;
    const grayscale = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
        const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
        grayscale[i / 4] = gray;
    }
    const tensor = tf.tensor3d(grayscale, [28, 28, 1]);
    return tensor.toFloat().div(255);
}
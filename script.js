let loading = false;
let model;
let canvas = document.getElementById('drawingCanvas');
let ctx = canvas.getContext('2d');
let drawing = false;
let lastX, lastY;

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

loadModel().then(() => {
  loading = false;
});

// Event listeners
document.getElementById('clearBtn').addEventListener('click', clearAll);
document.getElementById('recognizeBtn').addEventListener('click', recognizeDraw);

canvas.addEventListener('pointerdown', (e) => {
  drawing = true;
  lastX = e.clientX - canvas.getBoundingClientRect().left;
  lastY = e.clientY - canvas.getBoundingClientRect().top;
});

canvas.addEventListener('pointermove', (e) => {
  if (drawing) {
    let x = e.clientX - canvas.getBoundingClientRect().left;
    let y = e.clientY - canvas.getBoundingClientRect().top;
    drawOnCanvas(lastX, lastY, x, y);
    lastX = x;
    lastY = y;
  }
});

canvas.addEventListener('pointerup', () => {
  drawing = false;
  recognizeDraw();
});

function clearAll() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('predictedDigit').textContent = '—';
  document.getElementById('confidenceScore').textContent = '—';
}

function recognizeDraw() {
  if (loading || !model) return;
  let tensor = preprocessImage(canvas);
  runInference(tensor).then((result) => {
    let predictedDigit = result.predictedDigit;
    let confidence = result.confidence;
    document.getElementById('predictedDigit').textContent = predictedDigit;
    document.getElementById('confidenceScore').textContent = `Confidence score -> ${confidence.toFixed(1)}%`;
  }).catch((error) => {
    console.error('Error recognizing draw:', error);
  });
}
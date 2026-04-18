let model;
let loading = false;
let canvas = document.getElementById('drawingCanvas');
let ctx = canvas.getContext('2d');
let drawing = false;
let lastX, lastY;

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Load TensorFlow.js model
async function loadModel() {
  try {
    loading = true;
    model = await tf.loadLayersModel('./tfjs_model/model.json');
    loading = false;
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

loadModel();

// Handle drawing on canvas
canvas.addEventListener('pointerdown', (e) => {
  drawing = true;
  lastX = e.clientX - canvas.getBoundingClientRect().left;
  lastY = e.clientY - canvas.getBoundingClientRect().top;
});

canvas.addEventListener('pointermove', (e) => {
  if (drawing && e.buttons === 1) {
    const x = e.clientX - canvas.getBoundingClientRect().left;
    const y = e.clientY - canvas.getBoundingClientRect().top;
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

// Clear canvas
function clearAll(event) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('predictedDigit').textContent = '—';
  document.getElementById('confidenceScore').textContent = '—';
}

// Recognize drawn digit
async function recognizeDraw(event) {
  if (loading) return;
  try {
    const tensor = tf.tidy(() => {
      const canvasData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = 28;
      tempCanvas.height = 28;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);
      const tempData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
      const grayscaleData = new Uint8ClampedArray(tempData.data.length);
      for (let i = 0; i < tempData.data.length; i += 4) {
        const gray = (tempData.data[i] + tempData.data[i + 1] + tempData.data[i + 2]) / 3;
        grayscaleData[i] = gray;
        grayscaleData[i + 1] = gray;
        grayscaleData[i + 2] = gray;
        grayscaleData[i + 3] = 255;
      }
      const tensor = tf.tensor3d(grayscaleData, [28, 28, 1], 'float32');
      return tensor.div(255);
    });
    const predictions = await model.predict(tensor);
    const confidence = predictions.max().dataSync()[0];
    const predictedDigit = predictions.argMax().dataSync()[0];
    document.getElementById('predictedDigit').textContent = predictedDigit.toString();
    document.getElementById('confidenceScore').textContent = (confidence * 100).toFixed(1) + '%';
  } catch (error) {
    console.error('Error recognizing digit:', error);
  }
}
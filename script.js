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
  const rect = drawingCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  context.beginPath();
  context.moveTo(x, y);
});
drawingCanvas.addEventListener('pointermove', (e) => {
  if (isDrawing) {
    const rect = drawingCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    context.lineTo(x, y);
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
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempContext = tempCanvas.getContext('2d');
  tempContext.drawImage(drawingCanvas, 0, 0, 28, 28);
  const imageData = tempContext.getImageData(0, 0, 28, 28);
  const grayscaleData = new Uint8Array(28 * 28);
  for (let i = 0; i < imageData.data.length; i += 4) {
    const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
    grayscaleData[i / 4] = gray;
  }
  const tensor = tf.tensor4d(grayscaleData, [1, 28, 28, 1]);
  tensor = tensor.div(255);
  tf.tidy(() => {
    const output = model.predict(tensor);
    const probabilities = output.dataSync();
    const predictedDigit = probabilities.indexOf(Math.max(...probabilities));
    const confidence = Math.max(...probabilities);
    document.getElementById('predictedDigit').textContent = predictedDigit.toString();
    document.getElementById('confidenceScore').textContent = (confidence * 100).toFixed(1) + '%';
  });
}
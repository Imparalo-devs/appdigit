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

// Load model
async function loadModel() {
  try {
    loading = true;
    model = await tf.loadLayersModel('tfjs_model/model.json');
    loading = false;
  } catch (error) {
    console.error('Error loading model:', error);
    loading = false;
  }
}

loadModel();

drawingCanvas.addEventListener('pointerdown', (e) => {
  ctx.beginPath();
  ctx.moveTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
});

drawingCanvas.addEventListener('pointermove', (e) => {
  if (e.buttons === 1) {
    ctx.lineTo(e.clientX - drawingCanvas.getBoundingClientRect().left, e.clientY - drawingCanvas.getBoundingClientRect().top);
    ctx.stroke();
  }
});

drawingCanvas.addEventListener('pointerup', recognizeDraw);

clearBtn.addEventListener('click', clearAll);
recognizeBtn.addEventListener('click', recognizeDraw);

function clearAll() {
  ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
  predictedDigit.textContent = '—';
  confidenceScore.textContent = '—';
}

function recognizeDraw() {
  if (loading) return;
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(drawingCanvas, 0, 0, 28, 28);
  const imageData = tempCtx.getImageData(0, 0, 28, 28);
  const pixels = imageData.data;
  const grayscalePixels = new Uint8Array(28 * 28);
  for (let i = 0; i < pixels.length; i += 4) {
    const gray = Math.floor((pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3);
    grayscalePixels[i / 4] = gray;
  }
  try {
    const tensor = tf.tensor4d(grayscalePixels, [1, 28, 28, 1]).toFloat().div(255);
    tf.tidy(() => {
      try {
        const prediction = model.predict(tensor);
        const confidence = prediction.dataSync()[0];
        const digit = prediction.argMax().dataSync()[0];
        predictedDigit.textContent = digit.toString();
        confidenceScore.textContent = `Confidence score -> ${(confidence * 100).toFixed(1)}%`;
      } catch (error) {
        console.error('Error making prediction:', error);
      }
    });
  } catch (error) {
    console.error('Error creating tensor:', error);
  }
}
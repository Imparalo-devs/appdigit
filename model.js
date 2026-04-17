async function loadModel() {
  if (model) return;
  loading = true;
  try {
    model = await tf.loadLayersModel('/tfjs_model/model.json');
  } catch (error) {
    console.error('Error loading model:', error);
    model = null;
  } finally {
    loading = false;
  }
}

async function runInference(tensor) {
  try {
    let result = await tf.tidy(() => model.predict(tensor));
    let probabilities = result.dataSync();
    let predictedDigit = probabilities.indexOf(Math.max(...probabilities));
    let confidence = Math.max(...probabilities);
    return { predictedDigit, confidence };
  } catch (error) {
    console.error('Error running inference:', error);
    throw error;
  }
}

function preprocessImage(canvas) {
  try {
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    let tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    let imageData = tempCtx.getImageData(0, 0, 28, 28);
    let pixels = imageData.data;
    let grayscalePixels = new Uint8Array(28 * 28);
    for (let i = 0; i < pixels.length; i += 4) {
      let gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
      grayscalePixels[i / 4] = gray;
    }
    let tensor = tf.tensor3d(grayscalePixels, [28, 28, 1], 'uint8');
    return tensor.toFloat().div(255);
  } catch (error) {
    console.error('Error preprocessing image:', error);
    throw error;
  }
}
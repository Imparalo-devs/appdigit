let model;
let loading = false;

async function loadModel() {
    if (loading) return;
    loading = true;
    model = await tf.loadLayersModel('/tfjs_model/model.json');
    loading = false;
}

async function runInference() {
    if (!model) await loadModel();
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const tensor = tf.tidy(() => {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, 28, 28);
        const tempImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const grayscaleImageData = new Uint8ClampedArray(tempImageData.data.length);
        for (let i = 0; i < tempImageData.data.length; i += 4) {
            const gray = (tempImageData.data[i] + tempImageData.data[i + 1] + tempImageData.data[i + 2]) / 3;
            grayscaleImageData[i] = gray;
            grayscaleImageData[i + 1] = gray;
            grayscaleImageData[i + 2] = gray;
            grayscaleImageData[i + 3] = 255;
        }
        const tensor = tf.tensor3d(grayscaleImageData, [28, 28, 1], 'uint8');
        return tensor.toFloat().div(255);
    });
    const predictions = model.predict(tensor);
    const confidence = await predictions.data();
    const predictedDigit = confidence.indexOf(Math.max(...confidence));
    document.getElementById('predictedDigit').textContent = predictedDigit.toString();
    document.getElementById('confidenceScore').textContent = `Confidence score -> ${(confidence[predictedDigit] * 100).toFixed(1)}%`;
}
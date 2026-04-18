let model;
let loading = false;
let drawingCanvas = document.getElementById('drawingCanvas');
let context = drawingCanvas.getContext('2d');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');
let confidenceScore = document.getElementById('confidenceScore');
let predictedDigit = document.getElementById('predictedDigit');

context.strokeStyle = '#ffffff';
context.lineWidth = 12;
context.lineCap = 'round';
context.lineJoin = 'round';

// Load model
async function loadModel() {
    try {
        loading = true;
        model = await tf.loadLayersModel('/tfjs_model/model.json');
        loading = false;
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Clear canvas
function clearAll() {
    context.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

clearBtn.addEventListener('click', clearAll);

// Recognize digit
async function recognizeDraw() {
    if (loading) return;
    try {
        const tensor = tf.tidy(() => {
            const pixels = context.getImageData(0, 0, drawingCanvas.width, drawingCanvas.height).data;
            const grayscale = new Uint8Array(pixels.length / 4);
            for (let i = 0; i < pixels.length; i += 4) {
                const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
                grayscale[i / 4] = gray;
            }
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempContext = tempCanvas.getContext('2d');
            const tempImageData = tempContext.createImageData(28, 28);
            for (let i = 0; i < grayscale.length; i++) {
                tempImageData.data[i * 4] = grayscale[i];
                tempImageData.data[i * 4 + 1] = grayscale[i];
                tempImageData.data[i * 4 + 2] = grayscale[i];
                tempImageData.data[i * 4 + 3] = 255;
            }
            tempContext.putImageData(tempImageData, 0, 0);
            const tensor = tf.browser.fromPixels(tempCanvas);
            return tensor.toFloat().div(255);
        });
        const prediction = await model.predict(tensor);
        if (prediction) {
            const probabilities = prediction.dataSync();
            if (probabilities) {
                const maxIndex = probabilities.indexOf(Math.max(...probabilities));
                const confidence = probabilities[maxIndex];
                predictedDigit.textContent = maxIndex.toString();
                confidenceScore.textContent = (confidence * 100).toFixed(1) + '%';
            } else {
                console.error('Error getting prediction data');
            }
        } else {
            console.error('Error making prediction');
        }
    } catch (error) {
        console.error('Error recognizing digit:', error);
    }
}

recognizeBtn.addEventListener('click', recognizeDraw);
drawingCanvas.addEventListener('pointerup', recognizeDraw);
drawingCanvas.addEventListener('pointermove', (e) => {
    if (e.buttons === 1) {
        const rect = drawingCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        context.lineTo(x, y);
        context.stroke();
    }
});
drawingCanvas.addEventListener('pointerdown', (e) => {
    const rect = drawingCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    context.moveTo(x, y);
});

loadModel();
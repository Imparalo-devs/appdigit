let drawingCanvas = document.getElementById('drawingCanvas');
let ctx = drawingCanvas.getContext('2d');
let isDrawing = false;
let predictedDigit = document.getElementById('predictedDigit');
let confidenceScore = document.getElementById('confidenceScore');
let clearBtn = document.getElementById('clearBtn');
let recognizeBtn = document.getElementById('recognizeBtn');

ctx.strokeStyle = '#ffffff';
ctx.lineWidth = 12;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function clearAll(event) {
    ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    predictedDigit.textContent = '—';
    confidenceScore.textContent = '—';
}

function recognizeDraw(event) {
    let tensor = preprocessImage(drawingCanvas);
    runInference(tensor).then(result => {
        displayResult(result);
    });
}

drawingCanvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    let rect = drawingCanvas.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    ctx.beginPath();
    ctx.moveTo(x, y);
});

drawingCanvas.addEventListener('pointermove', (e) => {
    if (isDrawing) {
        let rect = drawingCanvas.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;
        ctx.lineTo(x, y);
        ctx.stroke();
    }
});

drawingCanvas.addEventListener('pointerup', (e) => {
    isDrawing = false;
    recognizeDraw(e);
});
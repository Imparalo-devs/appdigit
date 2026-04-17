let isDrawing = false;
let lastX, lastY;
const drawingCanvas = document.getElementById('drawingCanvas');
const context = drawingCanvas.getContext('2d');

drawingCanvas.addEventListener('pointerdown', (e) => {
    isDrawing = true;
    lastX = e.clientX - drawingCanvas.getBoundingClientRect().left;
    lastY = e.clientY - drawingCanvas.getBoundingClientRect().top;
});

drawingCanvas.addEventListener('pointermove', (e) => {
    if (!isDrawing) return;
    const x = e.clientX - drawingCanvas.getBoundingClientRect().left;
    const y = e.clientY - drawingCanvas.getBoundingClientRect().top;
    context.beginPath();
    context.moveTo(lastX, lastY);
    context.lineTo(x, y);
    context.stroke();
    lastX = x;
    lastY = y;
});

drawingCanvas.addEventListener('pointerup', () => {
    isDrawing = false;
});
export function initCanvas() {
    // No initialization needed
}

export function drawOnCanvas(ctx, lastX, lastY, x, y) {
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
}
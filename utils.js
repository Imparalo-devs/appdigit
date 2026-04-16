function preprocessImage(canvas) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tempContext.getImageData(0, 0, 28, 28);
    const grayscale = [];
    for (let i = 0; i < imageData.data.length; i += 4) {
        const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
        grayscale.push(gray / 255);
    }
    return tf.tensor3d(grayscale, [28, 28, 1]);
}

export { preprocessImage };
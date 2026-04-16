async function loadModel() {
    return tf.loadLayersModel('/tfjs_model/model.json');
}

function runInference(tensor) {
    return tf.tidy(() => {
        const prediction = model.predict(tensor);
        return prediction.dataSync();
    });
}

export { loadModel, runInference };
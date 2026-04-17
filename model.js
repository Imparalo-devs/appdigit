function loadModel() {
    return tf.loadLayersModel('/tfjs_model/model.json');
}

function runInference(model, tensor) {
    return model.predict(tensor);
}
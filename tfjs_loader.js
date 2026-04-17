function loadModel() {
    return tf.loadLayersModel('/tfjs_model/model.json');
}
export { loadModel };
function loadModel() {
    return tf.loadGraphModel('./tfjs_model/model.json');
}
export { loadModel };
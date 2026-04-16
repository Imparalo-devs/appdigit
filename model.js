import * as tf from '@tensorflow/tfjs';

export function loadModel() {
    return tf.loadLayersModel('/tfjs_model/model.json');
}
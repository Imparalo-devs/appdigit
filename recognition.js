export async function recognize(tensor) {
    if (!window.model) throw new Error('Model not loaded');
    
    const predictions = window.model.execute(tensor); // Raw logits
    const data = await predictions.data(); // [logit_0, ..., logit_9]
    
    // Applica softmax manualmente
    const sumExp = data.reduce((acc, x) => acc + Math.exp(x), 0);
    const softmaxData = data.map(x => Math.exp(x) / sumExp);
    
    return softmaxData; // [prob_0, ..., prob_9]
}
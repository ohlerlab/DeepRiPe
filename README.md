### DeepRiPe
#### Deep neural networks for interpreting RNA binding protein target preferences

Deep learning has become a powerful paradigm to analyze the binding sites of regulatory factors including RNA-binding proteins (RBPs), owing to its strength to learn complex features from possibly multiple sources of raw data. However, the interpretability of these models, which is crucial to improve our understanding of RBP binding preferences and functions, has not yet been investigated in significant detail.
We have designed a multitask and multimodal deep neural network for characterizing in vivo RBP binding preferences. The model incorporates not only the sequence but also the region type of the binding sites as input, which helps the model to boost the prediction performance. To interpret the model, we quantified the contribution of the input features to the predictive score of each RBP. Learning across multiple RBPs at once, we are able to avoid experimental biases and to identify the RNA sequence motifs and transcript context patterns that are the most important for the predictions of each individual RBP. Our findings are consistent with known motifs and binding behaviors of RBPs and can provide new insights about the regulatory functions of RBPs.


###

Input data for the PAR_CLIP models can be found at: https://ohlerlab.mdc-berlin.de/software/DeepRiPe_140/

# Questions
1. how would you prepare the data or get the data labeled for the training? 
2. What model would you choose given that I have limited GPU resources? say a single nVidia 4090 GPU
3. how to fine tune the model?
4. how to let the model have a memory so that it can remember all my previous conversations and respond to me more intelligently.

# Notes on Implementation
In my implementation, I've decided to use an encoder-decoder network with a memory module, where the pre-trained model serves as the encoder. To simulate memory, the network stores past input encodings, which could be adjusted to incorporate outputs as well. These stored encodings are fed into early attention layers in the decoder. This is a deliberate decision I've made with the internt of learning information from memory before attending to the current input's encoding. For optimal results, the model's forward operation is designed for teacher forcing, while the actual deployment of the model is handled by methods in the main.py folder, such as translate() and greedy_decode().
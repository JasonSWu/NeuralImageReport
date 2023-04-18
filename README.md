# Questions
1. how would you prepare the data or get the data labeled for the training? 
2. What model would you choose given that I have limited GPU resources? say a single nVidia 4090 GPU
3. how to fine tune the model?
4. how to let the model have a memory so that it can remember all my previous conversations and respond to me more intelligently.

# Notes on Implementation
In model.py, I've chosen to implement an encoder-decoder network with a memory module. The pre-trained model functions as the encoder. When our model is not training, the network stores encodings of past inputs (which could be modified to include outputs) and feeds these into early attention layers in the decoder. This is a notable design choice, as attending to memory in the bottom layers of the decoder theoretically helps the network learn information before attending to our current input's encoding. The forward of the model is intended for teacher forcing, and actual deployment of the model is handled by methods in the main.py folder, such as translate() and greedy_decode().
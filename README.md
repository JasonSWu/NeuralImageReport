# NeuralImage
1. how would you prepare the data or get the data labeled for the training? 
Ensuring data diversity/balance, we do topic modeling or clustering via unsupervised learning. Use K-means, LDA, LSA, BerTopic with tf-idf (or c-tf-idf).
2. What model would you choose given that I have limited GPU resources? say a single nVidia 4090 GPU
3. how to fine tune the model?
Take LLM, freeze or partially freeze weights, apply layer on top, execute supervised training. Huggingface seems to have a good implementation
across HuggingFace, TensorFlow, and PyTorch using the Trainer class. TensorFlow only has BERT models to load.
4. how to let the model have a memory so that it can remember all my previous conversations and respond to me more intelligently.
For memory, if want to query from previous conversations, can do key-value store or memory network. If the language model can accept large input, could just
concatenate new input with old inputs. Seems computationally costly.

# If we want to be able to choose which conversations to do, we could execute a latent dirichelet analysis
# on the input to analysis its topics, then compare this to the topics of the memory. 

Reinforcement learning from human feedback
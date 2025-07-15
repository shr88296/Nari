I would like to implement a streaming mechanism for dia inside the model file @dia/model.py. Note that dia uses a llama backend optimized with triton.

For this implementation we will look at @parler_streaming.py and @chatterbox_streaming.py as examples.

### Steps to Implement Streaming TTS

1.  **Create a Token-Level Streamer:**
    *   Implement a generator function that performs autoregressive inference to produce acoustic tokens.
    *   Instead of generating all tokens at once, this function should yield chunks of tokens of a specified `chunk_size`. It accumulates tokens in a buffer and yields the buffer when it's full.

2.  **Manage Context with an Overlap:**
    *   To avoid audio artifacts and maintain vocal consistency between chunks, a context window (the `overlap`) is used.
    *   For each new chunk of tokens to be converted to audio, a small number of tokens from the end of the *previous* chunk are prepended to it. This provides the model with context of what was just "spoken."

3.  **Convert Tokens to Audio and Crop:**
    *   The combined tokens (context overlap + new chunk) are passed to the vocoder or audio decoder to generate a waveform.
    *   Crucially, the resulting audio waveform is then cropped to remove the portion corresponding to the overlapping context tokens. Only the audio for the *new* tokens is kept. This prevents hearing the overlap.

4.  **Yield Audio Chunks:**
    *   The final, cropped audio chunk is what gets yielded to the consumer of the streaming function.

5.  **Smooth Chunk Boundaries (Optional but Recommended):**
    *   To prevent audible clicks or pops between chunks, a smoothing technique can be applied, such as a short linear fade-in to the beginning of each new audio chunk.
# Emotion Recognition in Conversations (ERC) and Emotion Flip Reasoning (EFR)

This repository contains the implementation and experiments for the **Emotion Recognition in Conversations (ERC)** and **Emotion Flip Reasoning (EFR)** tasks as part of the EDiReF shared task competition. The aim is to classify emotions in each utterance of a dialogue and identify trigger utterances that cause an emotion shift within the conversation. The project leverages transformer-based models, specifically BERT and ELECTRA, to achieve these goals.

## Models

We implemented and compared three different approaches to process the outputs from BERT and ELECTRA:

- **Concatenation (concat)**: Concatenates the encoding of the dialogue with the target utterance.
- **No Pooling (nopool)**: Uses the raw first token from the last hidden layer (the [CLS] token) as a representation of the whole dialogue.
- **Extraction (extraction)**: Processes the entire dialogue in a single forward pass, extracting the target utterance's tokens from the last hidden state and concatenating them with the [CLS] token embedding.

Additional information regarding the results obtained and the specifications on the techniques used can be found in the file `emotion_discovery_report.pdf`.

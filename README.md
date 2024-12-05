# MultimodalRAG

![Architecture](https://github.com/user-attachments/assets/aca2ff35-4358-4758-98e7-1f5c4a2063b3)

 Multimodal RAG System: End-to-End Workflow
 Our Multimodal Retrieval-Augmented Generation (RAG) system enables efficient retrieval
 and generation of insights from diverse data modalities such as video, audio, text, and
 images. Below, we provide a comprehensive breakdownof the workflow:
 1. Data Processing
 The systembegins by converting raw, heterogeneous input data into normalized textual
 representations suitable for downstream processing.
 Input Sources:
 • Videos: YouTube videos are downloadedand converted to audio files in .mp3 format.
 • PDFDocuments: Textandimagesareextracted directly from PDFs.
 Processing Steps:
 • Video Transcription:- Audio from.mp3files is transcribed into text using the Whisper model, which is robust
 for multilingual and noisy audio data.
 • Text Extraction from PDFs:- Rawtext is extracted, while images within the PDF are identified and isolated for further
 processing.
 • Image Captioning:- Captions summarizing extracted images are generated using the Salesforce/blip-image
captioning-base model, which provides a concise textual description of image content.
 • Summarization:- Lengthy textual content, whether transcriptions or extracted text, is summarized to retain
 key insights and improve query retrieval efficiency.
 Output: The processed data is stored in a unified textual format, creating a consistent
 foundation for embedding and retrieval.
2. Embedding
 The normalized textual data is transformed into a vector representation using a pre-trained
 embeddingmodel. This step ensures data from different modalities (audio, text, image) is
 mappedinto asharedsemantic space.
 • Model: Weusesentence-transformers/all-MiniLM-L6-v2, a lightweight and efficient model
 for sentence-level embeddings.
 • EmbeddingSpace: This model generates fixed-length vector representations, enabling the
 comparison andretrieval of semantically similar content across various data types.

3.Retriever
 The retriever component identifies the most relevant pieces of information from the pre
embeddeddataset in response to auser query.
 • QueryEmbedding:- User queries are transformed into vector representations using the same embedding
 model(sentence-transformers/all-MiniLM-L6-v2) to maintain consistency.
 • Nearest Neighbor Search:- Thesystem performsa nearestneighbor search in the vector database (e.g., FAISS,
 Pinecone) to identify the top-N relevant chunks based on vector similarity.
 4. Augmented Generation
 This step combines the retrieved chunks with the user query to create an augmented input
 for the generative model.
 • Augmentation:- Retrieved chunks (e.g., text passages, image captions) are concatenated with the user’s
 original query to provide context and improve the relevance of generated answers.
 • Generative Model:- Theaugmentedinput isfedinto thefacebook/bart-base generative model, which
 produces a coherent and contextually rich response.
 • Output:- Thegenerated response is designed to address the query comprehensively, incorporating
 insights from all relevant data modalities.
 5. User Interface
 The final step ensures the system is accessible and intuitive for end users.
 • Implementation: The interface is built using Streamlit, a Python-based framework for
 rapid development of interactive web applications.


 

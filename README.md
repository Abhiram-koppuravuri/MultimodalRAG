# MultimodalRAG

![Architecture](https://github.com/user-attachments/assets/aca2ff35-4358-4758-98e7-1f5c4a2063b3)

 # Multimodal Retrieval-Augmented Generation (RAG) System

Our **Multimodal RAG System** enables efficient retrieval and generation of insights from diverse data modalities, including video, audio, text, and images. Below is an end-to-end workflow of the system:

---

## Workflow Overview

### 1. Data Processing
The system begins by normalizing raw, heterogeneous input data into textual representations for downstream processing.

#### Input Sources:
- **Videos**: YouTube videos are downloaded and converted to audio files in `.mp3` format.
- **PDF Documents**: Text and images are extracted directly from PDFs.

#### Processing Steps:
- **Video Transcription**:
  - Audio from `.mp3` files is transcribed into text using the **Whisper model**, robust for multilingual and noisy audio data.
- **Text Extraction from PDFs**:
  - Raw text is extracted, while images within the PDFs are identified and isolated for further processing.
- **Image Captioning**:
  - Captions summarizing extracted images are generated using the **Salesforce/blip-image-captioning-base** model.
- **Summarization**:
  - Lengthy text (from transcriptions or extracted content) is summarized to retain key insights and improve query efficiency.

#### Output:
Processed data is stored in a unified textual format, creating a consistent foundation for embedding and retrieval.

---

### 2. Embedding
Normalized textual data is transformed into vector representations for semantic mapping.

#### Key Components:
- **Model**: 
  - We use **sentence-transformers/all-MiniLM-L6-v2**, a lightweight and efficient model for sentence-level embeddings.
- **Embedding Space**: 
  - Generates fixed-length vector representations to enable semantic comparison across data types.

---

### 3. Retriever
Identifies the most relevant information from the pre-embedded dataset in response to user queries.

#### Steps:
- **Query Embedding**: 
  - User queries are transformed into vector representations using the same embedding model (**sentence-transformers/all-MiniLM-L6-v2**) for consistency.
- **Nearest Neighbor Search**: 
  - The system performs a nearest neighbor search in the vector database (**FAISS**) to identify the top-N relevant chunks based on vector similarity.

---

### 4. Augmented Generation
Generates a coherent and contextually rich response by combining retrieved data with the user query.

#### Process:
- **Augmentation**:
  - Retrieved chunks (e.g., text passages, image captions) are concatenated with the user query to provide context.
- **Generative Model**:
  - The augmented input is fed into the **facebook/bart-base** generative model to produce a response.
- **Output**:
  - The response addresses the user query comprehensively, incorporating insights from all relevant modalities.

---

### 5. User Interface
Provides an intuitive and interactive front-end for the system.

#### Implementation:
- Built using **Streamlit**, a Python framework for rapid development of web applications.

---

## System Components and Models
| Component              | Model/Technology                                  |
|------------------------|---------------------------------------------------|
| Video Transcription    | Whisper                                           |
| Image Captioning       | Salesforce/blip-image-captioning-base            |
| Embedding              | sentence-transformers/all-MiniLM-L6-v2           |
| Vector Database        | FAISS, Pinecone                                  |
| Generative Model       | facebook/bart-base                               |
| Front-End Framework    | Streamlit                                        |

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Abhiram-koppuravuri/MultimodalRAG.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Application:
   ```bash
   streamlit run app.py



 

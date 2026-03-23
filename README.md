Multimodal Topic Segmentation using Heterogeneous Graph Transformers (HGT)
Overview

With the rapid expansion of online education platforms, lecture videos have become a dominant medium for knowledge delivery. However, these videos are typically long and lack clearly defined topic boundaries, making it difficult for learners to efficiently navigate, search, and retrieve relevant information. Traditional topic segmentation approaches rely primarily on textual transcripts and assume that topic transitions are reflected through changes in vocabulary or linguistic patterns. In real instructional settings, this assumption does not hold, as concepts are often introduced and explained using a combination of textual explanations, mathematical expressions, tables, and diagrams. This multimodal nature of lecture content necessitates a more comprehensive approach to topic segmentation.

This work presents a multimodal topic segmentation framework that models lecture content as a structured and heterogeneous representation. By integrating multiple instructional modalities and capturing their relationships, the proposed approach enables more accurate identification of topic boundaries in lecture videos and structured educational documents.

Proposed Methodology

The proposed system is organized as a multi-stage pipeline that progressively transforms raw lecture data into structured topic segments. The first stage focuses on visual processing, where lecture videos are decomposed into a sequence of frames. Each frame is analyzed to detect instructional elements such as text regions, mathematical expressions, tables, and diagrams. These elements are extracted and converted into instructional units that are aligned with the lecture timeline. Each instructional unit represents a minimal semantic component of the lecture and contains information about its content, temporal position, and modality type. This stage effectively transforms unstructured video data into a sequence of meaningful and temporally ordered instructional representations.

In the second stage, multimodal representation learning is performed. Each instructional unit undergoes modality-aware preprocessing to preserve its structural and semantic characteristics. Different types of instructional content are processed using specialized encoders tailored to their modality. Textual content is encoded using language models capable of capturing contextual semantics, while mathematical expressions are handled using models designed for symbolic representation. Tables are processed using structure-aware encoders that capture relationships between rows and columns, and diagrams are encoded using vision-based models that extract spatial and visual features. Since these modality-specific representations exist in different feature spaces, they are projected into a shared semantic space, enabling meaningful comparison and interaction across modalities.

The third stage involves constructing a structured graph representation of the lecture content. In this representation, each instructional unit is treated as a node, and relationships between units are modeled as edges. These relationships capture temporal continuity, cross-modal interactions, and semantic similarity between instructional components. This graph-based structure allows the system to represent both local transitions and long-range dependencies within the lecture. A Heterogeneous Graph Transformer is then applied to this graph to learn context-aware representations of instructional units. By leveraging relation-specific attention mechanisms, the model captures the sequential flow of the lecture, interactions between different modalities, and deeper semantic relationships that extend across distant parts of the content.

Following graph-based fusion, the system performs topic boundary detection by analyzing changes in the semantic continuity of instructional units over time. Instead of relying on simple threshold-based methods, a change-point detection mechanism is used to identify significant transitions in the content. This approach enables the detection of both abrupt and gradual topic shifts, which are common in real lecture scenarios. Once potential boundaries are identified, the lecture is segmented into contiguous topic units.

To further refine segmentation quality, a similarity-based grouping step is applied. This step ensures that segments with similar semantic content are grouped together, improving coherence and reducing fragmentation. The final output is a set of structured topic segments that accurately reflect the conceptual organization of the lecture.

Dataset Representation

To ensure consistency across different data sources, all datasets are converted into a unified representation. Each instructional unit is stored with a document identifier, a unit identifier, a temporal index indicating its position in the sequence, the content itself, and a modality label specifying whether the content is text, equation, table, or diagram. This standardized format allows the pipeline to process heterogeneous datasets in a consistent and scalable manner.

Datasets Used

The proposed framework is evaluated using a diverse collection of datasets that represent different types of instructional content. These include lecture transcripts, scientific articles, structured documents, and multimodal academic corpora. In addition to real-world datasets, a synthetic dataset is created to simulate realistic lecture scenarios with controlled modality distributions and clearly defined topic boundaries. This combination of datasets enables comprehensive evaluation across varying levels of multimodal complexity.

Evaluation and Results

The performance of the framework is evaluated using standard topic segmentation metrics that measure segmentation accuracy, boundary consistency, and overall detection quality. Experimental results demonstrate that incorporating multimodal information significantly improves segmentation performance compared to traditional text-based approaches. The proposed method achieves consistent improvements across multiple datasets, particularly in scenarios where non-textual instructional elements play a key role in topic transitions.

Limitations and Future Work

While the proposed framework demonstrates strong performance, it has certain limitations. The use of graph-based models introduces computational complexity, especially for long lecture sequences. The performance of the system also depends on the accuracy of the visual instructional element detection stage. Additionally, the current framework does not incorporate audio or speech signals, which can provide valuable cues for topic transitions.

Future work will focus on integrating additional modalities such as speech and audio features, optimizing the model for real-time applications, and improving scalability through more efficient graph-based architectures.

Applications

The proposed framework has broad applications in educational technology, including lecture video navigation, automated content summarization, intelligent tutoring systems, and semantic indexing of educational materials. By enabling accurate topic segmentation, the system enhances the accessibility and usability of complex instructional content.






 







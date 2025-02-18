### **PDF Block Classifier with Machine Learning Assistance**

This project is a GUI-based tool for classifying text blocks in PDF documents into predefined categories such as headers, body text (`<p>`), footers, quotes, and excluded content. It combines manual annotation with machine learning to streamline the classification process.
The tool is designed to assist users in organizing and extracting the text from PDF documents, for instance for conversion into a reflowable format.
This was mostly a learning project, and the code is relatively crude and messy.

#### **Key Features**
1. **Interactive GUI**: A Tkinter interface allows users to visually inspect and classify text blocks on PDF pages.
2. **Manual Annotation**: Users can click on text blocks to assign labels (e.g., header, body, footer) manually.
3. **Machine Learning Assistance**: A neural network model (MLP with residual blocks) assists in predicting block labels based on geometric and textual features, reducing the need for manual annotation.
4. **Feature Extraction**: The tool extracts a variety of features from text blocks, including geometric properties (position, size), textual properties (font size, word count), and linguistic features (capitalization, punctuation).
5. **Incremental Training**: The model is trained incrementally on each page as users annotate blocks, improving its predictions over time.
6. **Multi-Page Support**: The tool processes PDF documents page by page, allowing classification of entire files.
7. **Export Results**: Classified text blocks are exported to structured output files for further integration into other workflows (like https://github.com/Taylor-eOS/calibre-epub).

#### **How It Works**
1. **PDF Processing**: The tool uses `PyMuPDF` (Fitz) to extract text blocks and their geometric properties from PDF pages.
2. **Feature Extraction**: For each block, features such as position, font size, word count, and linguistic properties are computed.
3. **Manual Classification**: Users classify blocks by clicking on them and selecting a label from the GUI.
4. **Model Training**: The annotated blocks are used to train a neural network model, which then predicts labels for the next page.
5. **Prediction and Feedback**: The model's predictions are displayed in the GUI, and users can correct misclassifications, which further improves the model.

#### **Technical Details**
- **Backend**: The machine learning model is implemented using PyTorch, with a custom neural network architecture that includes residual blocks for quick learning while still getting better long-term.
- **Feature Engineering**: The tool computes a wide range of features for each text block, including geometric, textual, and linguistic properties.
- **Incremental Learning**: The model is trained incrementally as users annotate blocks, allowing it to adapt to the specific characteristics of the document being processed.

#### **Getting Started**
1. Clone the repository or download the files `main_script`, `model_util`, `utils`, and `gui_core`.
2. Install the required dependencies via the `requirements.txt` file.
3. Place your input PDF file in the same folder.
4. Run `main_script` and provide the basename the PDF file.

#### **Testing**
The script `testing.py` is used to run all data in the file `ground_truth.json` and compute an accuracy value, for quick model evaluation.

#### **Future Enhancements (wishlist)**
- Adding an embedding of text meaning as a feature (requires extracting entire document up-front)
- Relational features like other block counts on the same page

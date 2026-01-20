### **PDF Block Classifier with Machine Learning Assistance**

This project is a GUI-based tool for classifying text blocks in PDF documents into predefined categories such as headers, body text (`<p>`), footers, quotes, and excluded content. The user manually selects classifications and the model trains on each page to improve the prediction of subsequent pages. Classification works smoothly now, I ran through a 600 page book in a few minutes.<br>
The tool is designed to assist users in organizing and extracting the text from PDF documents, for instance for conversion into a reflowable format.<br><br>
This was a learning project that gradually extended on itself, and the code is very messy, to the point of being difficult to handle. The approach of guessing blocks in isolation without taking recent mistakes into account is imperfect and throws away a lot of information. So the script rattles through quite a complex machinery for fairly inaccurate results.<br>

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

#### **Usage**
1. Clone the repository or download the `py` files.
2. Create an environment and install the dependencies via the `requirements.txt` file.
3. Place your input PDF file in the same folder.
4. Run `main_script.py` and provide the basename the PDF file.

#### **Testing**
The script `testing.py` is used to run all data in the file `ground_truth.json` and compute an accuracy value, for quick model evaluation.

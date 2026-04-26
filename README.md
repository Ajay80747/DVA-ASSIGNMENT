# 📊 Instagram Multilingual Social Media Analysis

This project performs a comprehensive analysis of Instagram social media data (500k simulated records) using advanced NLP, Network Analysis, and Dimensionality Reduction techniques.

## 🚀 Features
- **Multilingual NLP**: Support for 9+ Indian languages (Hindi, Telugu, Marathi, etc.) using TF-IDF and Multilingual BERT.
- **Dimensionality Reduction**: PCA and t-SNE (2D & 3D) visualizations for linguistic cluster identification.
- **Network Analysis**: User interaction graphs and community detection using NetworkX.
- **Data Storytelling**: Narrative-driven insights into regional engagement trends.
- **Streamlit Dashboard**: A modern, interactive UI with dark theme, sidebar filters, and KPI metrics.

## 🛠️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ajay80747/DVA-ASSIGNMENT.git
   cd DVA-ASSIGNMENT
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn networkx transformers torch
   ```

3. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   *Note: If you encounter a "PermissionError" or "Access is denied" on Windows, try:*
   ```powershell
   $env:USERPROFILE=$pwd; streamlit run app.py
   ```

4. **Open the Notebook**:
   Use Google Colab or Jupyter to run `DVA_A2.ipynb` for the full analysis pipeline.

## 🛠️ Tech Stack
- **Python**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **NLP**: Transformers (BERT), Indic-NLP
- **Graph**: NetworkX
- **Dashboard**: Streamlit

## 📂 Project Structure
- `DVA_A2.ipynb`: Main analysis notebook with PCA and Storytelling.
- `app.py`: Advanced Streamlit dashboard implementation.
- `instagram_data.jsonl`: Simulated dataset (500k records).

## 🤖 Master Prompt for AI Generation
> "Act as a Senior Data Scientist. Create a complete Python pipeline for Instagram Social Media Analysis. The pipeline must include: 1) Multilingual data simulation (500k records) for languages like Hindi, Telugu, Marathi. 2) TF-IDF vectorization. 3) PCA and t-SNE dimensionality reduction with 2D/3D Plotly visualizations. 4) Network Analysis using NetworkX to find user clusters. 5) Sentiment analysis using Multilingual BERT. 6) A Data Storytelling section explaining insights. 7) A modern Streamlit dashboard with dark theme, sidebar filters, KPI cards, and interactive tabs. DO NOT use Tableau; use only Python libraries like Plotly and Streamlit for all visualizations."

---
*Developed for DVA Assignment - 2026*
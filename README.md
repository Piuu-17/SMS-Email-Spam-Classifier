# SMS / Email Spam Classifier 

An end-to-end Machine Learning web application that classifies messages (SMS or Email) as **Spam** or **Not Spam (Ham)** in real-time. Built using Natural Language Processing (NLP) techniques, scikit-learn, Streamlit, and deployed for interactive testing.


##  Executive Summary
* **Problem**: Unwanted promotional and phishing messages degrade user experience and pose security risks.
* **Solution**: A binary classification pipeline trained on real-world text data using NLP text pre-processing and machine learning algorithms (such as Naive Bayes / Random Forest) to accurately filter out spam.
* **Deployment**: Served through an interactive [Streamlit](https://streamlit.io/) web dashboard.


##  Tech Stack & Key Libraries
* **Language**: Python 3.x
* **Data Processing & Analysis**: Pandas, NumPy
* **NLP & Text Processing**: NLTK (Tokenization, Stopwords Removal, Stemming/Lemmatization), Bag of Words / TF-IDF Vectorization
* **Machine Learning**: Scikit-Learn (Multinomial Naive Bayes, Random Forest, Model Evaluation metrics)
* **Web Framework**: Streamlit
* **Version Control & Deployment**: Git, GitHub, Streamlit Community Cloud / Render / Heroku

##  Machine Learning Pipeline

flowchart TD
    A[Data Acquisition] --> B[Data Cleaning & Prep]
    B --> C[Exploratory Data Analysis]
    C --> D[Text Preprocessing]
    D --> E[Feature Extraction / Vectorization]
    E --> F[Model Training & Comparison]
    F --> G[Hyperparameter Tuning & Evaluation]
    G --> H[Model Persistence]
    H --> I[Web Interface / API Integration]
    I --> J[Deployment & Monitoring]

    subgraph Data Cleaning & Prep
        B1[Remove Duplicates]
        B2[Handle Missing Values]
        B3[Label Encoding]
    end

    subgraph Text Preprocessing
        D1[Lowercasing]
        D2[Tokenization]
        D3[Remove Special Characters]
        D4[Remove Stopwords & Punctuation]
        D5[Stemming / Lemmatization]
    end

    subgraph Feature Extraction / Vectorization
        E1[TF-IDF Vectorizer / Bag of Words]
    end

    subgraph Model Training & Comparison
        F1[Multinomial Naive Bayes]
        F2[Logistic Regression]
        F3[Random Forest / SVC]
    end

    subgraph Model Persistence
        H1[Export Model & Vectorizer via Pickle/Joblib]
    end

    subgraph Web Interface / API Integration
        I1[Streamlit App / Flask API]
    end

  ### Key Insights

1. **Top Performing Models:**
   * **Naive Bayes (NB):** Achieved **100% Precision** (`1.0`) with a solid accuracy of **94.00%**. In spam detection, precision is critical because false positives (classifying a legitimate email/message as spam) must be minimized.
   * **Extra Trees Classifier (ETC) & Random Forest (RF):** Delivered the best overall balance of high accuracy and high precision:
     * **ETC:** Accuracy = **97.68%**, Precision = **97.50%**
     * **RF:** Accuracy = **97.49%**, Precision = **98.28%**

2. **Underperforming Models:**
   * **Support Vector Classifier (SVC):** Exhibited a `Precision of 0.0` (with `UndefinedMetricWarning`), indicating that the model predicted zero positive samples (spam) in the test set.
   * **K-Nearest Neighbors (KN) & Decision Tree (DT):** Displayed lower precision compared to ensemble models (**77.12%** and **87.74%**, respectively).


### Future Improvements

1. **Hyperparameter Tuning & Threshold Optimization:**
   * Tune decision thresholds (especially for Naive Bayes and Support Vector Machines) to optimize the precision-recall trade-off rather than using the default 0.5 probability cutoff.

2. **Advanced Feature Engineering:**
   * **TF-IDF Enhancements:** Experiment with `ngram_range=(1, 2)` (unigrams + bigrams) and adjust `max_features` in `TfidfVectorizer` to capture contextual word sequences.
   * **Custom Features:** Engineer domain-specific features like character count, word count, presence of URL/phone patterns, or punctuation density (e.g., exclamation marks).

3. **Handling Class Imbalance:**
   * Implement techniques such as **SMOTE** (Synthetic Minority Over-sampling) or adjusting class weights (`class_weight='balanced'`) to prevent models like SVC from defaulting to the majority class.

4. **Deep Learning & Transformer Models:**
   * Evaluate pre-trained Transformer models like **BERT** or **RoBERTa** (`distilbert-base-uncased`) for superior contextual representation and classification accuracy.

5. **Voting / Stacking Classifier:**
   * Combine the top-performing models (e.g., Naive Bayes, Extra Trees, Random Forest, and XGBoost) using a **Soft Voting Classifier** or **Stacking Classifier** to boost generalization.

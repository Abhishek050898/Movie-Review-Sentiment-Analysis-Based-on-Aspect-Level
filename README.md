
# Movie Reviews Sentiment Analysis Based On Aspect-Level

This repository contains the code and dataset used in the research paper titled **"Movie Reviews Sentiment Analysis Based On Aspect-Level"** published in the *International Journal of All Research Education and Scientific Methods (IJARESM)*, November 2022.

## Abstract

The paper focuses on aspect-level sentiment analysis of movie reviews. Aspect-based sentiment analysis (ABSA) helps to detect emotional polarity on specific elements of a film, such as acting, direction, and cinematography. The study utilizes machine learning models to improve classification accuracy by focusing on specific aspects of movie reviews, offering more precise sentiment analysis than traditional methods.

## Features

- **Aspect-Based Sentiment Analysis**: Analyze sentiment polarity on specific movie aspects.
- **Machine Learning Models**: Various classifiers like Logistic Regression, Support Vector Machines, and Naive Bayes were used.
- **Data Preparation**: IMDb dataset from Kaggle, processed to suit ABSA models.
- **Hyperparameter Tuning**: Grid Search and cross-validation were used to optimize model performance.

## Methodology

The research employed:
- Aspect Extraction and Classification using a machine learning approach.
- A series of models such as Logistic Regression, Naive Bayes, and SVM were implemented to classify the polarity of reviews.

## Results

- **Accuracy**: Logistic Regression provided the best accuracy after hyperparameter tuning, reaching up to 89.81%.
- The system effectively handles slang, acronyms, emoticons, and misspellings in movie reviews.

## Dataset

The dataset used is from IMDb and contains movie reviews with labeled sentiment.

## Technologies

- **Python**: Used for dataset preprocessing, model training, and evaluation.
- **Scikit-learn**: Machine learning framework.
- **Pandas & NumPy**: Data manipulation and processing.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aspect-level-sentiment-analysis.git
   cd aspect-level-sentiment-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   Ensure the dataset from IMDb is placed in the correct directory (`/data` folder).

4. Run the model:
   ```bash
   python sentiment_analysis.py
   ```

## License

This project is licensed under the MIT License.

## Citation

If you use this code or dataset, please cite the following paper:

Kumar, Abhishek, and Kharyal, Kashish. "Movie Reviews Sentiment Analysis Based On Aspect-Level." International Journal of All Research Education and Scientific Methods (IJARESM), vol. 10, issue 11, 2022.

---

For more details, read the full paper [here](https://www.ijaresm.com/movie-reviews-sentiment-analysis-based-on-aspect-leve).

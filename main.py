import sys
import string
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score


class TextLengthTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to compute the character length of text documents."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not hasattr(X, '__iter__'):
            raise TypeError("Input X must be an iterable of text documents.")
        return np.array([[len(str(text))] for text in X], dtype=np.float64)


class PunctuationDensityTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to calculate the density of punctuation marks in text."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not hasattr(X, '__iter__'):
            raise TypeError("Input X must be an iterable of text documents.")
        
        features = []
        for text in X:
            text_str = str(text)
            length = len(text_str)
            if length == 0:
                features.append([0.0])
            else:
                punct_count = sum(1 for char in text_str if char in string.punctuation)
                features.append([punct_count / length])
        return np.array(features, dtype=np.float64)


def build_classifier_pipeline():
    """Constructs the FeatureUnion pipeline combined with OneVsRest classification."""
    return Pipeline([
        ('features', FeatureUnion([
            # Scalable tokenization and vectorization using hashing trick
            ('hashing_text', HashingVectorizer(
                n_features=2**16, 
                alternate_sign=False, 
                norm='l2',
                ngram_range=(1, 2)
            )),
            # Custom structural feature: Text Length
            ('text_length', Pipeline([
                ('extractor', TextLengthTransformer()),
                ('scaler', StandardScaler(with_mean=False))
            ])),
            # Custom structural feature: Punctuation Density
            ('punct_density', Pipeline([
                ('extractor', PunctuationDensityTransformer()),
                ('scaler', StandardScaler(with_mean=False))
            ]))
        ])),
        # Robust multiclass linear model wrapper
        ('classifier', OneVsRestClassifier(
            SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42, max_iter=1000)
        ))
    ])


def main():
    try:
        # Representative multi-class dataset simulation
        X_train = [
            "The stock market rallied today as tech shares surged on strong earnings reports.",
            "Quarterly financial results beat analyst estimates, driving massive market growth.",
            "The central bank announced a new interest rate cut to boost economic liquidity.",
            "Python and machine learning libraries are essential for modern software engineering.",
            "Deep neural networks and cloud architecture scale enterprise microservices.",
            "The striker scored a stunning goal in the final minutes of the football championship.",
            "The cricket match was thrilling as the team chased down a massive total.",
            "An intense political debate broke out in parliament regarding the new tax reform bill.",
            "The election campaign concluded with massive rallies across the capital city."
        ]
        y_train = ["Finance", "Finance", "Finance", "Tech", "Tech", "Sports", "Sports", "Politics", "Politics"]

        X_test = [
            "Investors are watching stock prices soar after positive economic indicators.",
            "Developers love deploying containerized applications using modern cloud pipelines.",
            "The basketball team secured a playoff spot with a dominant defensive performance.",
            "Lawmakers voted on the constitutional amendment during the legislative session."
        ]
        y_test = ["Finance", "Tech", "Sports", "Politics"]

        print("Initializing advanced multiclass classification pipeline...")
        pipeline = build_classifier_pipeline()

        print("Training model on dataset...")
        pipeline.fit(X_train, y_train)

        print("Evaluating model performance on test samples...")
        y_pred = pipeline.predict(X_test)

        print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.2f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
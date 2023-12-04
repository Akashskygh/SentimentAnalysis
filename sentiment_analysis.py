import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

df = pd.read_csv("Tweets.csv")
df.info()
df.describe()

sns.countplot(x='airline_sentiment', data=df)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='airline', hue='airline_sentiment', data=df)
plt.title('Sentiment Distribution for Each Airline')
plt.legend(title='Sentiment')
plt.show()

df['tweet_length'] = df['text'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(df['tweet_length'], bins=30, kde=True)
plt.title('Distribution of Tweet Length')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Drop non-numeric columns for PCA
numeric_cols = ['airline_sentiment_confidence', 'negativereason_confidence', 'retweet_count', 'tweet_length']
X_numeric = df[numeric_cols]

# Handling missing values
X_numeric = df[numeric_cols].copy()  # Create a copy of the numeric columns
X_numeric.fillna(X_numeric.mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Find number of components to explain 96% of the variance
n_components_96 = sum(cumulative_variance_ratio < 0.96) + 1

print("Number of components to explain 96% of the variance:", n_components_96)
print("Explained variance ratios:", explained_variance_ratio)

features = df['text'].values
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

y = df['airline_sentiment']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(processed_features, y, test_size=0.2)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(max_depth=5)

# Fit the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(10, 8))
plot_tree(dt_classifier, class_names=dt_classifier.classes_, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
ada_classifier = AdaBoostClassifier()
ada_classifier.fit(X_train, y_train)
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train)
mlp_classifier = MLPClassifier(max_iter=500)
mlp_classifier.fit(X_train, y_train)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_rf_classifier = grid_search.best_estimator_

mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['relu', 'tanh', 'identity']
}
mlp_grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=1000), param_grid=mlp_param_grid, cv=5)
mlp_grid_search.fit(X_train, y_train)
best_mlp_classifier = mlp_grid_search.best_estimator_

models = {
    'Random Forest': rf_classifier,
    'Random Forest (With Grid Search)': best_rf_classifier,
    'AdaBoost': ada_classifier,
    'Gradient Boosting': gb_classifier,
    'MLP': mlp_classifier,
    'MLP (With Grid Search)': best_mlp_classifier
}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.3f}")

for name, model in models.items():
    scores = cross_val_score(model, processed_features, y, cv=5, scoring='accuracy')
    print(f"{name} Cross-Validation Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

y_pred = best_rf_classifier.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

y_pred = best_mlp_classifier.predict(X_test)
print("MLP Classification Report:")
print(classification_report(y_test, y_pred))
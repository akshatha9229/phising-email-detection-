from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Example emails (you can replace with a bigger dataset)
emails = [
    "Your account is blocked, click here to login",   # phishing
    "Win a free prize, visit this link",              # phishing
    "Meeting tomorrow at 10 AM",                      # safe
    "Project report attached, please check"           # safe
]
labels = [1, 1, 0, 0]  # 1 = phishing, 0 = safe

# Step 1: Convert words into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Step 2: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

# Step 3: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Predict on test data
y_pred = model.predict(X_test)

# Step 5: Show results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

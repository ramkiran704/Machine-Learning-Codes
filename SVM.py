
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('iris.data')

# Display the original dataset
print(df.head())
print('Rows and columns:', df.shape)
print('Column names:', df.columns)
print('Target names:', df['Iris-setosa'].unique())

# Rename the columns
df.columns = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species'
]

# Display the dataset after renaming
print(df.head())
print('Rows and columns:', df.shape)
print('Column names:', df.columns)
print('Target names:', df['species'].unique())

# Select input and output
X = df[['sepal_length', 'sepal_width']].values
y = (df['species'] == 'Iris-setosa').astype(int)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train the SVM model
model = SVC(kernel='linear', C=2)
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Get the model coefficients
w = model.coef_[0]
b = model.intercept_[0]

# Create points for plotting
x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)

# Decision boundary and margins
decision = -(w[0] * x + b) / w[1]
margin1 = -(w[0] * x + b - 1) / w[1]
margin2 = -(w[0] * x + b + 1) / w[1]

# Plot the data points
plt.scatter(
    X[:, 0], X[:, 1],
    c=y, cmap='coolwarm',
    edgecolors='black'
)

# Plot the decision boundary
plt.plot(x, decision, 'k-', label='Decision boundary')

# Plot the margins
plt.plot(x, margin1, 'k--', label='Margin')
plt.plot(x, margin2, 'k--')

# Highlight support vectors
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=158,
    facecolors='none',
    edgecolors='green',
    label='Support vectors'
)

# Add labels and title
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Linear SVM: Setosa vs Non-setosa')
plt.legend()
plt.show()

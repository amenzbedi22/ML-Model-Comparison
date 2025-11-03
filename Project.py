# %%
# Importing necessary libraries for data manipulation and linear algebra
import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)

# Importing libraries for data visualization
import matplotlib.pyplot as plt # Plotting
import seaborn as sns # Statistical data visualization

# Importing machine learning libraries
from sklearn.model_selection import train_test_split # Splitting the dataset into training and testing sets
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors algorithm
from sklearn.linear_model import LogisticRegression # Logistic Regression algorithm
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

# Suppressing warnings for cleaner output
import warnings as wrn
wrn.filterwarnings('ignore')

# Importing Plotly for interactive plotting
import plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True) # Initializing Plotly for offline mode in Jupyter notebooks
import plotly.graph_objs as go # Graph objects in Plotly
import plotly.express as px # Express module for easy plotting in Plotly

# Importing additional machine learning libraries
from sklearn.preprocessing import StandardScaler # Standardizing features by removing the mean and scaling to unit variance
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier

# Re-importing seaborn and matplotlib for potential additional visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Importing metrics for model evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


# %%
data = pd.read_csv('Iris.csv')

# %%
data.head()

# %%
data.info()

# %%
data.describe()

# %%
data.isnull().sum()

# %%
# Create a new figure and axis object for plotting with a size of 12x8 inches
fig, ax = plt.subplots(figsize=(12, 8))

# Use Seaborn's countplot function to plot the counts of observations in the 'Legendary' column of the 'data' DataFrame
# The 'x' parameter specifies the column to plot on the x-axis
# 'data' specifies the DataFrame to use
# 'ax' parameter specifies the axis object to plot on
sns.countplot(x='Species', data=data, ax=ax)

# Display the plot
plt.show()


# %%
fig, ax = plt.subplots(figsize=(12, 8))
sns.countplot(x=data.Species, hue=data["Id"], ax=ax)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.select_dtypes(include=['float64', 'int64']).corr(), annot=True, fmt=".2f", linewidths=1.5, ax=ax)
plt.show()

# %%
sns.pairplot(data)
plt.tight_layout()  # Ensure tight layout
plt.show()

# %%
data.head()

# %%
le = LabelEncoder()
data["Species"] = le.fit_transform(data["Species"])
data.Species = data.Species.astype(int)

# %%
data.head()

# %%
data.tail()

# %%
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# %%
iris = load_iris()
X, y = iris.data, iris.target
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%
predictions = {}
predictions_proba = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test)
    else:
        y_pred_prob = None
    
    predictions[name] = y_pred
    predictions_proba[name] = y_pred_prob


# %%
from sklearn.preprocessing import label_binarize

results = []

for name in models.keys():
    y_pred = predictions[name]
    y_pred_prob = predictions_proba[name]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    if y_pred_prob is not None:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        try:
            auc = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
        except Exception:
            auc = float('nan')
    else:
        auc = float('nan')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc
    })
    results_df = pd.DataFrame(results)

    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision (macro): {precision:.2f}")
    print(f"Recall (macro): {recall:.2f}")
    print(f"F1-Score (macro): {f1:.2f}")
    print(f"AUC: {auc if not np.isnan(auc) else 'N/A'}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=np.arange(len(iris.target_names))+0.5, labels=iris.target_names)
    plt.yticks(ticks=np.arange(len(iris.target_names))+0.5, labels=iris.target_names)
    plt.show()


# %%
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Binarize test labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(14, 10))

for name in models.keys():
    y_pred_prob = predictions_proba[name]
    if y_pred_prob is None:
        print(f"{name} does not support predict_proba, skipping ROC curve.")
        continue
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot micro-average ROC curve for this model
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'{name} micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')

# Plot random guessing diagonal
plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve for Iris Dataset')
plt.legend(loc="lower right")
plt.show()


# Bar plot for comparative performance metrics
results_df.set_index('Model', inplace=True)
results_df.plot(kind='bar', figsize=(14, 8))
plt.title('Comparative Performance of Classification Algorithms')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.show()




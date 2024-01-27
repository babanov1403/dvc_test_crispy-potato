from dvclive import Live
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


iris = load_iris()
X = iris.data
y = iris.target
clf = DecisionTreeClassifier()

with Live() as live:
    live.log_param("epochs", 1)
    for i in range(1, 6):
        for j in range(2, 6):
            for k in range(1, 6):
                clf = DecisionTreeClassifier(max_depth=i
                                             , min_samples_split=j
                                             , min_samples_leaf=k)
                clf.fit(X, y)
                y_pred = clf.predict(X)
                live.log_metric('precision', precision_score(y, y_pred, average = 'micro'))
                live.log_metric('recall', recall_score(y, y_pred, average = 'micro'))
                live.log_sklearn_plot("confusion_matrix", y, y_pred)

                conf_matrix = confusion_matrix(y, y_pred)
                sns_plot = sns.heatmap(conf_matrix, annot=True)
                fig = sns_plot.get_figure()
                fig.savefig('tmp.png')
                plt.close()
                live.log_image(f"img/{live.step}.png", 'tmp.png')
                os.remove('tmp.png')
                live.next_step()
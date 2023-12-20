import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def cm_save(pred, y, run_path):
    cm = confusion_matrix(y, pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    # tick_marks = np.arange(np.unique(y))
    # plt.xticks(tick_marks)
    # plt.yticks(tick_marks)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.savefig(run_path + "confusion_matrix.png")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#akurasi
def Accuracy(y_test, y_pred):
    accuracy = np.round(accuracy_score(y_test, y_pred),2)
    return accuracy

#presisi
def Precision(y_test, y_pred):
    presisi = np.round(precision_score(y_test, y_pred),2)
    return presisi

#recall
def Recall(y_test, y_pred):
    recall = np.round(recall_score(y_test, y_pred),2)
    return recall

#F1 Score
def F1(y_test, y_pred):
    f1 = np.round(f1_score(y_test, y_pred),2)
    return f1

#Confusion matix
def CM(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm

#TN, FP, FN, dan TP
def CMValue(cm):
    # Inisialisasi variabel array untuk menyimpan TN, FP, FN, dan TP
    TN = []
    FP = []
    FN = []
    TP = []
    T = []
    F = []

    # Menghitung nilai TN, FP, FN, dan TP dari setiap matriks CM
    for i in range(len(cm)):
        tn, fp, fn, tp = cm[i].ravel()
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        TP.append(tp)
        T.append(tn+tp)
        F.append(fn+fp)
    return TN, FP, FN, TP, T, F

#metriks evaluasi accuracy, presisi, recall, f1, cm
def Metrics(y_test, predict):
    accuracy, presisi, recall, f1, cm = [], [], [], [], []
    for i in range(predict.shape[1]):
        nilai_accuracy = Accuracy(y_test, predict.iloc[:,i])
        nilai_presisi = Precision(y_test, predict.iloc[:,i])
        nilai_recall = Recall(y_test, predict.iloc[:,i])
        nilai_f1 = F1(y_test, predict.iloc[:,i])
        nilai_cm = CM(y_test, predict.iloc[:,i])
        accuracy.append(nilai_accuracy)
        presisi.append(nilai_presisi)
        recall.append(nilai_recall)
        f1.append(nilai_f1)
        cm.append(nilai_cm)
    return accuracy, presisi, recall, f1, cm

#visualisasi metrik
def MetricsVisualization(metrics, title):
    # Plot grup bar chart
    plt.figure(figsize=(8, 6))
    bar_width = 0.15
    index = np.arange(len(metrics['k']))
    opacity = 0.8

    plt.bar(index, metrics['akurasi'], bar_width, alpha=opacity, label='Akurasi')
    plt.bar(index + bar_width, metrics['presisi'], bar_width, alpha=opacity, label='Presisi')
    plt.bar(index + (2 * bar_width), metrics['recall'], bar_width, alpha=opacity, label='Recall')
    plt.bar(index + (3 * bar_width), metrics['f1'], bar_width, alpha=opacity, label='F1')

    # Menambahkan label pada sumbu x dan y
    plt.xlabel('k')
    plt.ylabel('Nilai')

    # Menambahkan judul
    plt.title(title)

    # Mengatur label sumbu x
    plt.xticks(index + bar_width, metrics['k'])

    # Menampilkan legenda di luar plot dan mengatur posisi legenda
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Menampilkan plot
    plt.show()

#visualisasi confusion matrix
def CMVisualization(cm, k):
    # Mengatur ukuran dan layout
    fig, axes = plt.subplots(nrows=1, ncols=len(cm), figsize=(15, 4))

    # Memvisualisasikan setiap matriks CM
    for i, (matrix, k) in enumerate(zip(cm, k)):
        ax = axes[i]
        sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', cbar=False, ax=ax,
            xticklabels=['0', '1'],   
            yticklabels=['0', '1'])
        ax.set_title(f"Confusion Matrix K={k}")
        ax.set_xlabel ('prediksi')       
        ax.set_ylabel ('aktual')    

    # Menampilkan plot
    plt.tight_layout()
    plt.show()
    print()
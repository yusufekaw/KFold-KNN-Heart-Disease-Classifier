from data.DataPreprocessing import *
from algorithm.KFold import *
from algorithm.KNN import *
from evaluation.EvaluationMetrics import *

# Main program
if __name__ == '__main__':
    
    # Load dataset
    dataset = LoadData() 
    
    # menampilkan dataset
    print ("\n\t\t============================HASIL IMPORT DATASET============================") 
    print (dataset)

    # informasi kolom  dataset
    print("\n\t\t============================INFORMASI KOlOM DATASET============================")
    dataset.info()

    # menampilkan informasi kolom kategorikal (object)
    print("\n\t\t============================INFORMASI KOlOM KATEGORIKAL============================")
    PrintColumnKategorical(dataset)

    # encoding nilai kategorikal menjadi numerikal
    dataset = LabelEncode(dataset) 
    
    # menampilkan dataset yang telah diencoding
    print ("\n\t\t============================HASIL ENCODING DATASET============================")
    print (dataset)

    # Export ke excel
    # nama file untuk menyimpan laporan, file telah disiapkan sebelumnya
    file_name = "data/dataset/report.xlsx"

    #eksport hasil encoding ke excel
    ExportToExcel(dataset, file_name, "label_encoding")

    # Normalisasi dataset
    dataset = MinMax(dataset) 
    
    # menampilkan dataset yang telah dinormalisasi
    print ("\n\t\t============================HASIL NORMALISASI DATA DENGAN MINMAX============================")
    print (dataset)

    #eksport hasil normalisasi min-max ke excel
    ExportToExcel(dataset, file_name, "MinMax")

    #inisiasi variabel fold
    n_split = 9 #menentukan jumlah lipatan
    fold = []
    
    #split datafold
    fold = SplitFold(dataset, n_split)
    
    #inisiasi 6 K berbeda
    k = [3,5,7,9,11,13]
    maks_K = np.max(k)

    #inisiasi variabel metrik evaluasi bernilai 0
    avg_accuracy = avg_precission = avg_recall = avg_f1 = 0
    metrics = 0
    k_total = 0

    #menjalankan algoritma KNN
    for i in range(len(fold)):
        #data training
        print(f"\n\t\t============================DATA TRAINING KE-{i}============================")
        data_training = DataTrainig(fold[i], dataset)
        print(data_training)

        #data testing
        print(f"\n\t\t============================DATA TESTING KE-{i}============================")
        data_testing = fold[i]
        print(data_testing)

        #Split fitur dataset
        column_target = "HeartDisease"
        X_train, X_test, y_train, y_test = Feature(data_training, data_testing, column_target)
        
        #mencari jarak terdekat
        nearest, indeks = Nearest(X_test, X_train, y_train, maks_K)
        # Melakukan klasifikasi menggunakan kNN untuk setiap nilai k
        y_pred = allPredict(k, X_train, y_train, X_test)
        # mencari jarak terdekat
        nearest_result = NearestResult(X_test, y_train, nearest, indeks)    
        #konversi data frame
        nearest_result = pd.DataFrame(nearest_result, columns=["Testing", "Training", "Jarak", "Kelas"])
        print(nearest_result)

        #klasifikasi berdasarkan jarak terdekat
        predict_column = ["K" + str(val) for val in k] # Membuat dictionary kolom
        predict = pd.DataFrame(np.transpose(y_pred), columns=predict_column) # Membuat dataframe
        classification = Classification(X_test, predict, y_test) # Hasil klasifikasi
        print(classification)

        #eksport hasil normalisasi min-max ke excel
        ExportToExcel(classification, file_name, "Class Fold "+ str(i))
        
        #akurasi, presisi, recall, f1, cm
        accuracy, precission, recall, f1, cm = Metrics(y_test, predict)
        #TN, FP, TP, T, F
        TN, FP, FN, TP, T, F = CMValue(cm)
        #ilai CM
        cm_value = pd.DataFrame({'k': k, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'T': T, 'F': F})
        
        #confusion matrix data ke fold ke-i 
        print(f"\n\t\t============================CONFUSION MATRIX DATA FOLD KE-{i}============================")
        print(cm_value)
        #visualisasi confusion matrix
        CMVisualization(cm, k)

        #metrik fold ke-i
        metrics_per_fold = pd.DataFrame({'k': k,'akurasi': accuracy, 'presisi': precission, 'recall': recall, 'f1': f1})
        print(f"\n\t\t============================METRIC EVALUASI DATA FOLD KE-{i}============================")
        print(metrics_per_fold)
        title = f"metrics data fold ke-{i}"
        #viasualisasi metric evaluasi
        MetricsVisualization(metrics_per_fold, title)

        #join semua hasil metric per fold
        if(i==0):
            metrics=metrics_per_fold
        else:
            metrics=pd.concat([metrics, metrics_per_fold], ignore_index=True)

    #rata2 metrics
    #nilai desimal dua digit
    avg_accuracy = round(metrics.groupby('k')['akurasi'].mean(),2)
    avg_accuracy = avg_accuracy.reset_index(drop=True)
    avg_precission = round(metrics.groupby('k')['presisi'].mean(),2)
    avg_precission = avg_precission.reset_index(drop=True)
    avg_recall = round(metrics.groupby('k')['recall'].mean(),2)
    avg_recall = avg_recall.reset_index(drop=True)
    avg_f1 = round(metrics.groupby('k')['f1'].mean(),2)
    avg_f1 = avg_f1.reset_index(drop=True)
    k=pd.DataFrame({'k':k})
    avg_metrics =  pd.concat([k, avg_accuracy, avg_precission, avg_recall, avg_f1], axis=1)
    print("\n\t\t============================RATA-RATA NILAI METRIC EVALUASI============================")
    print(avg_metrics)
    title = "Rata-rata metric"
    MetricsVisualization(avg_metrics, title)

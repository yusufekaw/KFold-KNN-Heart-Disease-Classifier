import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import euclidean_distances

#Pemodelan KNN
def Predict(X_train, y_train, X_test, k):
    # Inisialisasi model KNN
    knn_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # Melatih model dengan data training
    knn_model.fit(X_train, y_train)
    # Melakukan prediksi pada data testing
    y_pred = knn_model.predict(X_test)    
    # Mengembalikan hasil prediksi dan jarak
    return y_pred

#perhitungan jarak
def Distance(X_test, X_train):
     # Menghitung jarak antara data testing dan data training
    jarak = euclidean_distances(X_test, X_train)
    jarak = np.round(jarak, 2)
    jarak = pd.DataFrame(jarak, columns=[X_train.index[i] for i in range(len(X_train))])
    jarak.index = [X_test.index[i] for i in range(len(X_test))]
    return jarak

#Ketetanggan terdekat
def Nearest(X_test, X_train, y_train, k):
    # Mendapatkan jarak dan indeks tetangga terdekat
    knn_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # Melatih model dengan data training
    knn_model.fit(X_train, y_train)

    distances, indices = knn_model.kneighbors(X_test)
    terdekat = np.round(distances,2) #jarak terdekat
    indeks = indices #indeks
    return terdekat, indeks

#hasil ketetanggan terdekat
def NearestResult(X_test, y_train, terdekat, indeks):
    hasil_jarak_terdekat = []
    #klasifikasi = []
    for i in range(len(X_test)):
        #data_klasifikasi = [X_test.index[i], y_pred[i], y_test.values[i]] #kombinasi index data testing dengan hasil prediksi
        #klasifikasi.append(data_klasifikasi) #disimpan dalam variabel klasifikasi
        for j in range(0, len(indeks[i])):
            #print(X_test.index[i], "\t", terdekat[i][j], "\t", indeks[i][j] ,"\t", y_train.iloc[indeks[i][j]])
        #print()
            #mendapatkan nilai indeks, jarak dan kelas
            data_jarak_terdekat = [X_test.index[i], indeks[i][j], terdekat[i][j], y_train.iloc[indeks[i][j]]]
            #menggabungkan nilai diatas
            hasil_jarak_terdekat.append(data_jarak_terdekat)
    return hasil_jarak_terdekat#, klasifikasi

#Hasil Klasifikasi    
def Classification(X_test, prediksi, y_test):
    klasifikasi = []
    for i in range(len(X_test)):
        data_klasifikasi = [X_test.index[i], y_test.values[i]] #kombinasi index data testing dengan hasil prediksi
        klasifikasi.append(data_klasifikasi) #disimpan dalam variabel klasifikasi
    df_klasifikasi = pd.DataFrame(klasifikasi, columns=["#", "target"])
    hasil_klasifikasi = df_klasifikasi.join(prediksi)
    return hasil_klasifikasi

#proses klasifikasi
def allPredict(K, X_train, y_train, X_test):
    y_pred = [] #inisiasi variabel
    #proses klasisifikasi pada setiap K
    for k in K:
        prediksi = Predict(X_train, y_train, X_test, k)
        y_pred.append(prediksi)
    return y_pred

#memisah fitur X dan Y
def Feature(data_training, data_testing, column_target):
    X_train = data_training.drop(columns=[column_target]) #X Traib
    X_test = data_testing.drop(columns=[column_target]) #X test
    y_train = data_training[column_target] #Y train
    y_test = data_testing[column_target] #Y Test 
    return X_train, X_test, y_train, y_test
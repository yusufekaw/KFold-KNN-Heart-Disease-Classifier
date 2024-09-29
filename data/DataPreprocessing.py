import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from openpyxl import *

#fungsi untuk memuat dataset
def LoadData():
    # Path ke file dataset
    # Membaca dataset menggunakan Pandas
    dataset = pd.read_csv("data/dataset/heart.csv")
    return dataset

#Informasi kolom bertipe data object
def ColumnInfo(dataset):
    #mengambil hanya data bertipe object
    object_data_type_kolom = dataset.select_dtypes(include='object').columns
    result = {} #variabel untuk menyimpan hasil
    for kolom in object_data_type_kolom:
        unique_value = dataset[kolom].nunique() #nilai unik dari kolom
        count_value = dataset[kolom].value_counts() #manghitung jumlah nilai dalam kolom
        value_info = {value: count for value, count in count_value.items()}
        result[kolom] = {
            'nilai_unik': unique_value, #nilai dalam kolom
            'nilai_perhitungan': value_info #jumlah nilai unik dalam kolom
        }
    return result

#Label Encoding mengubah nilai kategorikal menjadi numerikal
def LabelEncode(dataset):
    le = LabelEncoder()
    for kolom in dataset.columns:
        if dataset[kolom].dtype == 'object': #mengambil kolom bertipe data objek
            dataset[kolom] = le.fit_transform(dataset[kolom]) #mengganti nilai kategori menjadi angka
    return dataset

#Min-Max Normalisasi
def MinMax(dataset):
    scaler = MinMaxScaler()
    data_normalisasi = dataset.apply(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    data_normalisasi = np.round(data_normalisasi,2)
    data_normalisasi['HeartDisease'] = data_normalisasi['HeartDisease'].astype(int)
    return data_normalisasi

#print kolom kategorikal
def PrintColumnKategorical(dataset):
    #info lkolom
    info_kolom = ColumnInfo(dataset)
    #menghitung jumlah nilai unik setiap kolom
    for kolom, info in info_kolom.items():
        unique_value = info['nilai_unik']
        value_count = info['nilai_perhitungan']
        print("Kolom :", kolom)
        for nilai, hitung in value_count.items():
            print("\t\t",nilai,"\t: ",hitung," data")
        print("\t",unique_value," Nilai")

#export data frame ke excel
def ExportToExcel(data, file_name, sheet_name):
    workbook = load_workbook(file_name)
    #export hanya jika tidak terdapat sheet
    if sheet_name not in workbook.sheetnames:
        with pd.ExcelWriter(file_name, engine="openpyxl", mode="a") as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=True)
    
from sklearn.model_selection import KFold

#mambagi data sejumlah n fold
def SplitFold(dataset, n_split):
    #inisiasi jumlah lipatan
    kfold = KFold(n_splits=n_split, shuffle=False)
    # Variabel untuk menyimpan data setiap fold
    fold = []
    # Lakukan iterasi KFold dan simpan data di setiap fold
    for train_index, test_index in kfold.split(dataset):
        fold_data = dataset.iloc[test_index]
        fold.append(fold_data)
    return fold

#mengambil data training, data fold selain data ke-i
def DataTrainig(fold, dataset):
    # Mendapatkan indeks baris yang terdapat dalam folds[i] / data testing
    idx_fold = fold.index
    # Menampilkan data dari dataset yang tidak termasuk dalam folds[i] / data testing
    data_training = dataset.drop(idx_fold)
    return data_training

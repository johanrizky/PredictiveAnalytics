# Laporan Proyek Machine Learning - Johan Rizky Triosaputra

## Domain Proyek

Prediksi harga mobil telah menjadi bidang penelitian yang menarik untuk dipelajari. Mengutip dari jurnal [*Car price prediction using machine learning techniques*](https://temjournal.com/content/81/TEMJournalFebruary2019_113_118.pdf) oleh Gegic E., menurut Badan Pusat Statistik BiH, 921.456 kendaraan terdaftar pada tahun 2014 dari mana 84% dari mereka adalah mobil untuk penggunaan pribadi. Angka persen ini meningkat sebesar 2,7% sejak 2013 dan kemungkinan hal ini akan terus berlanjut. Hal tersebut dapat terjadi karena mobil adalah kendaraan yang sangat bermanfaat jika memiliki keluarga lebih dari dua orang, sehingga dapat membantu mobilitas terutama sesuai dari tempat Badan Pusat Statistik BiH sendiri yang bertempat di Eropa. Menaiknya persen penggunaan mobil ini tentu banyak orang berminat terhadap mobil, sehingga jika ada orang memiliki penghasilan yang sedikit atau kegunaan untuk komersial tentu pabrik pembuatan mobil akan menyesuaikan fitur- fitur yang ada di dalam mobil tersebut sehingga tetap dapat dijangkau dan dapat menyesuaikan fitur didalamnya sesuai kegunaanya. Oleh karena itu, prediksi harga mobil yang sesuai fiturnya dapat digunakan dalam hal seperti ini. Prediksi harga mobil yang akurat dapat melibatkan seorang praktisi *Machine Learning*, karena harga tergantung dari ciri dan faktor yang khas. Biasanya, kebanyakan yang signifikan adalah model mobil, tahun pembelian, transmisi mobil, jarak yang sudah ditempuh, konsumsi bahan bakar per mil, dan tipe mesin atau ukuran mesin. Dalam laporan ini, saya menerapkan model *Machine Learning Predictive Analytics* yang diharapkan saat akhir dapat menghasilkan output untuk membantu memprediksi harga mobil sesuai dengan fitur- fitur yang ada dalam mobil. 

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Dari beberapa fitur dalam mobil, fitur apa yang berpengaruh terhadap harga mobil?
- Apakah model- model mobil juga mempengaruhi harga, meskipun fiturnya sama?
- Apakah pajak mempengaruhi harga mobil ?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengetahui fitur mobil yang paling berkorelasi dengan harga mobil.
- Membuat model *Machine Learning* yang dapat mengetahui korelasi model mobil dengan harga mobil dimana fitur dalamnya yang hampir sama.
- Mengetahui apakah pajak mempengaruhi harga mobil.
- ### Solution statements
    Menggunakan algoritma K- Nearest Neighbor, Random Forest, dan Boosting Algorithm untuk mengetahui performa prediksi akurasi dan mengetahui mana yang terbaik untuk digunakan dalam kasus ini.

## Data Understanding
Data yang digunakan oleh proyek ini adalah dari [Kaggle](https://www.kaggle.com/datasets) bernama [100,000 UK Used Car Data set](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?select=ford.csv) dimana yang saya gunakan adalah data dari Mobil Ford. Dalam data ford.csv ini memiliki 17966 baris, dan 9 fitur kolom.

### Variabel-variabel pada 100,000 UK Used Car Data set pada data ford.csv adalah sebagai berikut:
- *model* : merupakan tipe mobil dari merk Ford.
- *year* : merupakan tahun pembelian mobil atau pembelian tangan pertama.
- *price* : merupakan harga mobil dalam satuan dollar ($).
- *transmission* : merupakan tipe transmisi yang ada dalam mobil.
- *mileage* : merupakan jarak yang sudah ditempuh oleh mobil tersebut.
- *fuelType* : merupakan tipe bahan bakar yang dapat digunakan oleh tiap- tiap tipe mobil.
- *tax* : merupakan pajak tahunan yang keluar dalam satuan dollar ($).
- *mpg* : merupakan satuan mil yang dapat ditempuh per gallon atau liter.
- *engineSize* : merupakan ukuran mesin dalam satuan unit.

### Exploratory data analysis:

<img src="https://github.com/johanrizky/PredictiveAnalytics/blob/main/read_csv.jpg" alt="drawing" width="200"/> 
Gambar 1. Visualisasi data dengan read_csv



- Visualisasi data untuk melihat isi data csv menggunakan fungsi read_csv dari library pandas seperti pada gambar 1. 


<img src="https://github.com/johanrizky/PredictiveAnalytics/blob/main/info().jpg" alt="drawing" width="200"/>
Gambar 2. Visualisasi dengan fungi info()



- Mengetahui tipe data dengan fungsi info(), seperti pada gambar 2.


<img src="https://github.com/johanrizky/PredictiveAnalytics/blob/main/describe().jpg" alt="drawing" width="400"/>
Gambar 3. Cek deskripsi dengan fungsi describe()



- Mengetahui nilai yang tidak masuk akal seperti 0 pada *engineSize* seperti pada gambar 3.


<img src="https://github.com/johanrizky/PredictiveAnalytics/blob/main/drop%200.jpg" alt="drawing" width="200"/>
Gambar 4. hapus baris dengan nilai 0



- Menangani *missing value* dengan menghapus baris yang memiliki nilai nol seperti pada gambar 4, dengan catatan nilai nol tersebut tidak masuk akal. Selanjutnya data menjadi berukuran 15.768.



<img src="https://github.com/johanrizky/PredictiveAnalytics/blob/main/cek%20outliers.jpg" alt="drawing" width="500"/>
Gambar 5. Menghilangkan *outlier* dengan IQR



- Menangani *outlier* yang berada di luar Q1 dan Q3 dengan metode IQR seperti pada gambar 5.


<img src="https://github.com/johanrizky/PredictiveAnalytics/blob/main/Univariate%20numerical%20feature.jpg" alt="drawing" width="200"/>


Gambar 6. Contoh *univariate* untuk *numerical feature*



- Melakukan proses analisis *univariate* dengan membagi fitur kolom menjadi *numerical* dan *categorical*.

<img src="https://github.com/johanrizky/PredictiveAnalytics/blob/main/multivariate%20featrue%20categorical.jpg" alt="drawing" width="200"/>

Gambar 7. *Multivariate categorical feature*

<img src="https://github.com/johanrizky/PredictiveAnalytics/blob/main/korelasi%20fitur%20numerik%20terhadap%20price.jpg" alt="drawing" width="200"/>
Gambar 8. Korelasi fitur *numeric* terhadap *price*

<img src="https://github.com/johanrizky/PredictiveAnalytics/blob/main/korelasi%20price%20terhadap%20fitur%20mobil%20setelah%20one%20hot%20encoding.jpg" alt="drawing" width="700"/>
Gambar 9. Korelasi *price* terhadap fitur mobil setelah *one-hot-encoding*



- Melakukan proses analisis *multivariate* untuk mengetahui hubungan antara dua atau lebih variabel seperti pada gambar 7, contoh *price* dengan *transmission*. Disini juga akan mengevaluasi skor korelasi fitur kolom yang berisi numerik menggunakan fungsi corr() seperti pada gambar 8, dan mengetahui skor korelasi *price* dengan setiap model mobil dan fitur mobil setelah melakukan *one-hot-encoding* seperti pada gambar 9.

## Data Preparation

### Tahapan Data Preparation :
1. Encoding fitur kategori.
   - Encoding fitur kategori digunakan untuk mendapatkan fitur baru yang sesuai dengan fitur kategorikal yang diubah.
   - Encoding fitur kategori dilakukan agar kategori dalam kolom yang bersifat kategori berubah menjadi value angka, sehingga dapat digunakan untuk menguji korelasi model mobil dengan harga.
2. Reduksi dimensi dengan PCA.
   - Teknik reduksi (pengurangan) dimensi adalah prosedur yang mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data.
   - Principal Component Analysis (PCA) merupakan teknk untuk mereduksi dimensi, mengekstrak fitur, dan menstransformasi data digunakan ketika variabel dalam data memiliki korelasi yang paling tinggi. 
   - Alasan menggunakan PCA adalah karena PCA digunakan untuk mereduksi variabel asli menjadi sejumlah kecil variabel baru yang tidak berkorelasi linier atau disebut Komponen Utama (PC), dimana komponen utama ini dapat menangkap sebagian besar varians dalam variabel asli.
3. Pembagian dataset dengan fungsi train_test_split dari library sklearn.
    - train_test_split merupakan pembagian data latih (train) dan data uji (test) yang biasanya memiliki ratio 8:2.
    - train_test_split diperlukan karena untuk mempertahankan sebagian data yang untuk menguji data yang baru, agar tidak mengotori data uji dengan data latih, dan agar tidak berpotensi menimbulkan kebocoran data (data leakage).
4. Standarisasi.
    - Standarisasi adalah proses standarisasi fitur dengan mengurangkan mean (nilai rata- rata) kemudian membaginya dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.
    - Standarisasi digunakan agar mengubah nilai rata- rata (mean) menjadi 0 dan niai standar deviasi menjadi 1.

## Modeling

### Tahapan modeling yang digunakan :
1. Menyiapkan data frame untuk digunakan sebagai analisi ketiga model nantinya.
2. K-Nearest Neighbor
    - K-NN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain atau dengan objek terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). Cara K-NN adalah menentukan nilai K yaitu di parameter n_neighbors yang akan dipakai pada model KNeighborsRegressor seperti potongan kode berikut berikut : 
        ```
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X_train, y_train)
        ```
        Setelah K ditentukan di parameter KNeighborsRegressor, lalu dipanggil dengan model fit.
    - Kelebihan K-NN adalah mudah digunakan karena dapat menghindari overfit dan underfit bila memilih nilai K yang sesuai.
    - Sedangkan kekurangannya adalah  jika dihadapkan pada jumlah fitur atau dimensi yang besar. Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data.
3. Random Forest
    - Random Forest merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Cara menenggunakan Random forest sebagai berikut :
        ```
        RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
        RF.fit(X_train, y_train)
        ```
        - Menentukan tree yang dibentuk n_estimators semakin banyak semakin baik tetapi program semakin lama.
        - Menentukan ukuran seberapa banyak tree dapat membelah untuk membagi setiap node ke jumlah pengamatan di max_depth.
        - mengontrol random number generator yang digunakan pada random_state.
        - Menentukan jumlah proses yang berjalan / job (pekerjaan yang digunakan secara paralel). n_jobs=-1 artinya proses berjalan secara paralel. 
    - Kelebihan Random Forest dapat dihadapkan dengan fitur yang acak dan data yang banyak.
    - Sedangkan kekurangan Random Forest adalah sulit diterapkan karena membutuhkan hyperparameter yang tepat.
4. Boosting Algorithm
    - Boosting merupakan model yang dilatih secara berurutan atau dalam proses yang iteratif. Teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Cara menggunakan boosting algorithm sebagai berikut :
        ```
        boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
        boosting.fit(X_train, y_train)
        ```      
        - Menentukan bobot yang diterapkan pada masing- masing iterasi boosting pada parameter learning_rate.
        - Mengontrol random number generator yang digunakan pada parameter random_state.
    - Kelebihan algoritma adalah dapat membentuk suatu model yang kuat (strong ensemble learner), powerful dalam meningkatkan akurasi prediksi, dan mengungguli model yang lebih sederhana seperti logistic regression dan random forest.
    - Hampir tidak ada kelemahan kecuali kurang membaca dokumentasi.

### Model yang terbaik digunakan :
Dalam kasus ini, saya menggunakan kasus regresi. Dalam tes program saya sendiri model algoritma boosting adalah yang terbaik, ditandai dengan jarak hasil akurasi train dengan test hampir berdekatan dibanding algoritma K- KN dan Random Forest. Boosting menjadi yang terbaik di model saya, karena boosting dalam program saya tidak memerlukan tambahan hyperparameter tuning lagi.

## Evaluation
1. Metrik evaluasi pertama yang saya gunakan pada prediksi ini adalah MSE atau Mean Squared Erro yang menghitung jumlah selisih kuadrat rata- rata nilai sebenarnya dengan nilai prediksi, nilai MSE disini juga digunakan untuk membandingan ketiga algoritma terbaik yang digunakan. Dalam menghitung MSE perlu dilakukan proses scaling fitur numerik pada data uji agar skala antara data latih dan data uji sama dan kita bisa melakukan evaluasi.
    
    - Langkah yang dilakukan membuat MSE :
        1. Membuat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma.
        2. Membuat dictionary untuk setiap algoritma yang digunakan.
        3. Menghitung Mean Squared Error masing-masing algoritma pada data train dan test.
        4. Memanggil mse
    - Nilai evaluasi dalam tiap Mean Square Error
    Nilai yang diperoleh di latih dan test pada tabel salinan program saya adalah sebagai berikut :
        |                     	|    train    	|     test    	|
        |:-------------------:	|:-----------:	|:-----------:	|
        | K-Nearest Neighbors 	| 2617.599334 	| 3792.114025 	|
        |    Random Forest    	|  1271.62211 	| 3857.129955 	|
        |  Boosting Algorithm 	| 6004.249221 	| 6416.680764 	|

        Dari tabel tersebut MSE yang dihasilkan dengan data latih dan uji masih berdekatan jadi belum termasuk overfit atau underfit. Dimana hasil MSE yang tertinggi adalah algoritma boosting, berdasarkan tabel yang mendapat angka diatas 600. ***Algoritma boosting inilah yang merupakan model terbaik terhadap dataset yang saya gunakan ini***, yaitu untuk prediksi harga mobil berdasarkan fitur- fitur yang ada dalam mobil tersebut.
2. Metrik kedua yang saya gunakan adalah metrics [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) dari sklearn untuk mengetahui detail akurasi yang didapatkan dari algoritma boosting.
    
    Langkah yang dapat dilakukan : 
    1. Melakukan predict testing test dari algoritma boosting
    2. Mengecek performa model dengan r2_score menggunakan library metrics dari sklearn

### Referensi
1. E. Gegic, B. Isakovic, D. Keco, Z. Masetic and J. Kevric, "Car Price Prediction using Machine," Technology, Education, Management, Informatics, vol. 8, no. 1, pp. 113-118, 2019. [source journal](https://temjournal.com/content/81/TEMJournalFebruary2019_113_118.pdf) 
2. [Machine Learning Terapan](https://www.dicoding.com/academies/319/) dari Dicoding
3. Evaluasi metrics [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) skelarn

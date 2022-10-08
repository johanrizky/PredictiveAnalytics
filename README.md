# Laporan Proyek Machine Learning - Johan Rizky Triosaputra

## Domain Proyek

Prediksi harga mobil telah menjadi bidang penelitian yang menarik untuk dipelajari. Sesuai informasi yang didapat dari Badan Pusat Statistik BiH, 921.456 kendaraan terdaftar pada tahun 2014 dari mana 84% dari mereka adalah mobil untuk penggunaan pribadi. Angka persen ini meningkat sebesar 2,7% sejak 2013 dan kemungkinan hal ini akan terus berlanjut. Hal tersebut dapat terjadi karena mobil adalah kendaraan yang sangat bermanfaat jika memiliki keluarga lebih dari dua orang, sehingga dapat membantu mobilitas terutama sesuai dari tempat Badan Pusat Statistik BiH sendiri yang bertempat di Eropa. Menaiknya persen penggunaan mobil ini tentu banyak orang berminat terhadap mobil, sehingga jika ada orang memiliki penghasilan yang sedikit atau kegunaan untuk komersial tentu pabrik pembuatan mobil akan menyesuaikan fitur- fitur yang ada di dalam mobil tersebut sehingga tetap dapat dijangkau dan dapat menyesuaikan fitur didalamnya sesuai kegunaanya. Oleh karena itu, prediksi harga mobil yang sesuai fiturnya dapat digunakan dalam hal seperti ini. Prediksi harga mobil yang akurat dapat melibatkan seorang praktisi Machine Learning, karena harga tergantung dari ciri dan faktor yang khas. Biasanya, kebanyakan yang signifikan adalah model mobil, tahun pembelian, transmisi mobil, jarak yang sudah ditempuh, konsumsi bahan bakar per mil, dan tipe mesin atau ukuran mesin. Dalam laporan ini, saya menerapkan model Machine Learning Predictive Analytics yang diharapkan saat akhir dapat menghasilkan output untuk membantu memprediksi harga mobil sesuai dengan fitur- fitur yang ada dalam mobil. 

Referensi:

1. [Car price prediction using machine learning techniques](https://temjournal.com/content/81/TEMJournalFebruary2019_113_118.pdf) dari Google Scholar
2. [Machine Learning Terapan](https://temjournal.com/content/81/TEMJournalFebruary2019_113_118.pdf) dari Dicoding
3. Evaluasi metrics [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Dari beberapa fitur dalam mobil, fitur apa yang berpengaruh terhadap harga mobil?
- Apakah model- model mobil juga mempengaruhi harga, meskipun fiturnya sama?
- Apakah pajak mempengaruhi harga mobil ?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengetahui fitur mobil yang paling berkorelasi dengan harga mobil.
- Membuat model Machine Learning yang dapat mengetahui korelasi model mobil dengan harga mobil dimana fitur dalamnya yang hampir sama.
- Mengetahui apakah pajak mempengaruhi harga mobil.
- ### Solution statements
    Menggunakan algoritma K- Nearest Neighbor, Random Forest, dan Boosting Algorithm untuk mengetahui performa prediksi akurasi dan mengetahui mana yang terbaik untuk digunakan dalam kasus ini.

## Data Understanding
Data yang digunakan oleh proyek ini adalah dari [100,000 UK Used Car Data set](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?select=ford.csv) dimana yang saya gunakan adalah data dari Mobil Ford. Dalam data ford.csv ini memiliki 17966 baris, dan 9 fitur kolom.

### Variabel-variabel pada 100,000 UK Used Car Data set pada data ford.csv adalah sebagai berikut:
- model : merupakan tipe mobil dari merk Ford.
- year : merupakan tahun pembelian mobil atau pembelian tangan pertama.
- price : merupakan harga mobil dalam satuan dollar ($).
- transmission : merupakan tipe transmisi yang ada dalam mobil.
- mileage : merupakan jarak yang sudah ditempuh oleh mobil tersebut.
- fuelType : merupakan tipe bahan bakar yang dapat digunakan oleh tiap- tiap tipe mobil.
- tax : merupakan pajak tahunan yang keluar dalam satuan dollar ($).
- mpg : merupakan satuan mil yang dapat ditempuh per gallon atau liter.
- engineSize : merupakan ukuran mesin dalam satuan unit.

### Exploratory data analysis:
- Visualisasi data untuk melihat isi data csv menggunakan fungsi read_csv dari library pandas.
- Mengecek informasi pada dataset menggunakan fungsi info(), untuk mengetahui tipe data.
- Mengecek deskripsi statistik data dengan fitur describe(), juga untuk mengetahui nilai minimum yang tidak masuk akal pada data.
- Menangani missing value dengan mendrop baris yang memiliki nilai nol, dengan catatan nilai nol tersebut tidak masuk akal. Contohnya engineSize bernilai nol.
- Menangani outlier yang berada di luar Q1 dan Q3 dengan metode IQR.
- Melakukan proses analisis univariate dengan membagi fitur kolom menjadi numerical dan categorical.
- Melakukan proses analisis multivariate untuk mengetahui hubungan antara dua atau lebih variabel, contoh price dengan transmission. Disini juga akan mengevaluasi skor korelasi fitur kolom yang berisi numerik menggunakan fungsi corr(), dan mengetahui skor korelasi price dengan setiap model mobil dan fitur mobil setelah melakukan one-hot-encoding.

## Data Preparation

### Tahapan Data Preparation :
1. Encoding fitur kategori.
   - Encoding fitur kategori digunakan untuk mendapatkan fitur baru yang sesuai dengan fitur kategorical yang diubah.
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
    - K-NN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k atau objek terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat).
     - Kelebihan K-NN adalah mudah digunakan karena dapat menghindari overfit dan underfit bila memilih nilai K yang sesuai.
    - Sedangkan kekurangannya adalah  jika dihadapkan pada jumlah fitur atau dimensi yang besar. Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data.
3. Random Forest
    - Random Forest merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian.
    - Kelebihan Random Forest dapat dihadapkan dengan fitur yang acak dan data yang banyak.
    - Sedangkan kekurangan Random Forest adalah sulit diterapkan karena membutuhkan hyperparameter yang tepat.
4. Boosting Algorithm
    - Boosting merupakan model yang dilatih secara berurutan atau dalam proses yang iteratif. Teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.
    - Kelebihan algoritma adalah dapat membentuk suatu model yang kuat (strong ensemble learner), powerful dalam meningkatkan akurasi prediksi, dan mengungguli model yang lebih sederhana seperti logistic regression dan random forest.
    - Hampir tidak ada kelemahan kecuali kurang membaca dokumentasi.

### Model yang terbaik digunakan :
Dalam kasus ini, saya menggunakan kasus regresi. Dalam tes program saya sendiri model algoritma boosting adalah yang terbaik, ditandai dengan jarak hasil akurasi train dengan test hampir berdekatan dibanding algoritma K- KN dan Random Forest. Boosting menjadi yang terbaik di model saya, karena boosting dalam program saya tidak memerlukan tambahan hyperparameter tuning lagi.

## Evaluation
1. Metrik evaluasi pertama yang saya gunakan pada prediksi ini adalah MSE atau Mean Squared Erro yang menghitung jumlah selisih kuadrat rata- rata nilai sebenarnya dengan nilai prediksi, nilai MSE disini juga digunakan untuk membandingan ketiga algoritma terbaik yang digunakan. Dalam menghitung MSE perlu dilakukan proses scaling fitur numerik pada data uji agar skala antara data latih dan data uji sama dan kita bisa melakukan evaluasi.
    
    Langkah yang dapat dilakukan :
    1. Membuat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma.
    2. Membuat dictionary untuk setiap algoritma yang digunakan.
    3. Menghitung Mean Squared Error masing-masing algoritma pada data train dan test.
    4. Memanggil mse
2. Metrik kedua yang saya gunakan adalah r2_score untuk mengetahui detail akurasi yang didapatkan dari algoritma boosting.
    
    Langkah yang dapat dilakukan : 
    1. Melakukan predict testing test dari algoritma boosting
    2. Mengecek performa model dengan r2_score menggunakan library metrics dari sklearn

**---Ini adalah bagian akhir laporan---**

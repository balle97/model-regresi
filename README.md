# Laporan Proyek Machine Learning - MUHAMMAD IQBAL

## Domain Proyek

> Domain proyek yang dipilih adalah mengenai bisnis properti, khususnya rumah, dengan judul proyek **"Prediksi Harga Rumah di Kecamatan Tebet"**.

### Latar Belakang Masalah

Rumah adalah salah satu dari tiga kebutuhan dasar manusia, selain kebutuhan sandang dan pangan. Sejalan dengan konsep Hierarki Kebutuhan menurut **Maslow** yang menyebutkan bahwa rumah sebagai bagian dari kebutuhan fisiologis manusia. Seiring perkembangan waktu, kebutuhan fisiologis manusia semakin meningkat, termasuk keinginan untuk memiliki rumah. Dengan meningkatnya daya beli masyarakat, harga rumah terus mengalami peningkatan. Para pengembang properti bersaing untuk membangun properti rumah sebagai bentuk investasi. Namun, fluktuasi harga rumah yang sulit diprediksi dengan pasti meningkatkan risiko investasi. 

Menurut **Albani Musyafa (2013)**, harga rumah dipengaruhi oleh faktor lokasi dan biaya lahan [[1](https://www.researchgate.net/publication/325656226_KOMPOSISI_HARGA_JUAL_RUMAH_TINGGAL_LAYAK_HUNI_DI_YOGYAKARTA_STUDI_KASUS_PEMBANGUNAN_RUMAH_TIPE_90115_DI_LUAR_KOMPLEKS_PERUMAHAN)]. Namun, Menurut **Hendra et al. (2017)** menambahkan bahwa faktor akses transportasi juga berpengaruh [[2](https://jurnal.untan.ac.id/index.php/justin/article/view/18455)]. Selain itu, penelitian dari **Saiful et al. (2021)** menunjukkan bahwa penggunaan machine learning dengan algoritma linear regression dapat memberikan prediksi harga rumah dengan tingkat akurasi yang memuaskan [[3](https://jurnal.mdp.ac.id/index.php/jatisi/article/view/701)]. Namun, penelitian dari **Haryanto et al. (2023)** menunjukkan bahwa algoritma random forest regression memberikan akurasi yang lebih tinggi dalam memprediksi harga rumah dibandingkan dengan metode lainnya [[4](https://ejournal.itn.ac.id/index.php/jati/article/view/6343)]. 

Oleh karena itu, penting bagi pengembang properti untuk memiliki pemahaman mendalam mengenai faktor-faktor yang memengaruhi harga rumah di kecamatan Tebet. Prediksi akan digunakan untuk menentukan berapa harga yang pantas untuk properti rumah berdasarkan pertimbangan faktor-faktor tertentu sehingga pengembang properti bisa mendapatkan profit sebesar mungkin. Tidak adanya acuan harga rumah, seperti acuan harga emas, mendorong kebutuhan akan sistem prediksi yang dapat membantu mengurangi risiko kerugian dalam pengembangan properti. Sehingga, pada proyek ini akan dibuat model prediksi untuk memprediksi harga rumah di kecamatan Tebet dengan algoritma machine learning regresi, seperti SVR, Random Forest dan Gradient Boosting.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang masalah diatas, terdapat rumusan masalah yang dapat diselesaikan dalam proyek ini, antara lain:
- Apa yang menjadi faktor penting yang mempengaruhi harga rumah di kecamatan Tebet?
- Bagaimana memprediksi harga rumah di kecamatan Tebet berdasarkan pertimbangan faktor-faktor tertentu?

### Goals

Untuk menjawab rumusan masalah tersebut, maka akan dibuat predictive modelling dengan tujuan sebagai berikut:
- Mengetahui faktor yang paling berkorelasi dengan harga rumah di kecamatan Tebet.
- Membuat model machine learning yang dapat memprediksi harga rumah di kecamatan Tebet seakurat mungkin berdasarkan pertimbangan faktor-faktor tertentu.

### Solution Statements

Untuk mencapai tujuan tersebut, maka akan diberikan beberapa solusi diantaranya:
- Menganalisis data dengan melakukan penanganan _missing values_, penanganan _duplicates_, penanganan _outliers_, _univariate analysis_ dan _multivariate analysis_ serta penggunaan fungsi _corr_ dan _heatmap_ untuk mengetahui korelasi antar faktor penentu.
- Menggunakan beberapa algoritma machine learning seperti _Support Vector Regression_ (SVR), _Random Forest_, dan _Gradient Boosting untuk mengembangkan model. Dari ketiga model ini, akan dipilih satu model yang memiliki nilai kesalahan prediksi terendah. Intinya, tujuannya adalah menciptakan model yang seakurat mungkin dengan menghasilkan nilai kesalahan yang seminimal mungkin. Menggunakan metrik evaluasi seperti Mean Squared Error (MSE) untuk mengevaluasi kinerja model dalam memprediksi harga. Secara keseluruhan, metrik ini digunakan untuk mengukur seberapa akurat model dalam memperkirakan nilai yang sebenarnya.

## Data Understanding

Data yang digunakan dalam proyek ini adalah **Harga Rumah Tebet dataset** yang dibuat dan dikumpulkan oleh [Wisnu Anggara](https://www.kaggle.com/wisnuanggara) yang di upload ke situs [Kaggle](https://www.kaggle.com/). Dataset ini memiliki 1.010 jenis rumah dengan berbagai faktor penentu. Faktor yang dimaksud disini adalah fitur numerik seperti luas bangunan, luas tanah, kamar tidur, kamar mandi dan garasi. Kelima fitur ini adalah fitur yang akan digunakan dalam menemukan pola pada data, sedangkan harga merupakan fitur target. Dataset tersebut dipilih karena memenuhi kriteria yang dicari untuk membuat model regresi dengan harga rumah sebagai target.

Informasi dataset dapat dilihat pada tabel dibawah ini :
Jenis | Keterangan
--- | ---
Sumber | [Kaggle Dataset: Harga Rumah Tebet dataset](https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah/data?select=DATA+RUMAH.xlsx)
Lisensi | [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
Kategori | Bisnis Properti
Format dan Ukuran File | xlsx (72,24 kB)
Jumlah Data | 1.010 sampel dengan 8 fitur
Tipe Data | 7 fitur bertipe int64 dan 1 fitur bertipe object

### Variabel-variabel pada Harga Rumah Tebet dataset adalah sebagai berikut:
- NO : merupakan fitur yang merepresentasikan nomor urut data.
- NAMA RUMAH : merupakan fitur yang merepresentasikan judul iklan rumah.
- HARGA : merupakan fitur target yang merepresentasikan harga rumah dalam mata uang Rupiah Indonesia (Rp).
- LB : merupakan fitur yang merepresentasikan ukuran luas bangunan.
- LT : merupakan fitur yang merepresentasikan ukuran luas tanah.
- KT : merupakan fitur yang merepresentasikan jumlah kamar tidur.
- KM : merupakan fitur yang merepresentasikan jumlah kamar mandi.
- GRS : merupakan fitur yang merepresentasikan jumlah kapasitas mobil dalam garasi.

### Tahapan yang Diperlukan untuk Memahami Data
1. Data Loading.
2. Exploratory Data Analysis - Description of Variables.
3. Exploratory Data Analysis - Handling Missing Values, Drop Unnecessary Columns, and Duplicates.
4. Exploratory Data Analysis - Visualizing and Handling Outliers.
5. Exploratory Data Analysis - Univariate and Multivariate Analysis.

#### 1. Data Loading
Pada tahap ini, hal pertama yang perlu dilakukan adalah mengimpor library **pandas** untuk memuat dataset berformat Excel ke dalam dataframe menggunakan fungsi `read_excel()` yang tersedia dalam library tersebut. Dataset yang saya gunakan bernama DATA RUMAH.xlsx. Adapun outputnya sebagai berikut: 

![1-data-awal](https://github.com/balle97/model-regresi/assets/128248022/06626c41-1feb-4700-9954-bb0bf2870a34) 

Dari output di atas terlihat bahwa:
* Terdapat 1.010 baris (records atau jumlah pengamatan) dalam dataset.
* Terdapat 8 kolom yaitu: NO, NAMA RUMAH, HARGA, LB, LT, KT, KM, GRS.

#### 2. Exploratory Data Analysis - Description of Variables
Langkah selanjutnya adalah mengecek informasi pada dataset menggunakan fungsi `info()`. Adapun outputnya sebagai berikut: 

![2-info-tipe-data](https://github.com/balle97/model-regresi/assets/128248022/2416687c-0f24-4a33-ac75-a7af0097092b) 

Dari output di atas terlihat bahwa:
* Terdapat 1 kolom dengan tipe data object, yaitu: NAMA RUMAH. Kolom ini merupakan fitur non-numerik, tapi bukan fitur kategorikal.
* Terdapat 6 kolom numerik dengan tipe data int64 yaitu: NO, LB, LT, KT, KM, GRS. Kolom ini merupakan fitur numerik.
* Terdapat 1 kolom numerik dengan tipe data int64, yaitu: HARGA. Kolom ini merupakan target fitur.

Selain itu, mengecek deskripsi statistik dataset menggunakan fungsi `describe()` untuk mengecek apakah ada nilai minimum 0 pada setiap kolom tersebut. 

Fungsi `describe()` memberikan informasi statistik pada masing-masing kolom, antara lain:
* Count : jumlah sampel pada data.
* Mean : nilai rata-rata.
* Std : standar deviasi atau varians.
* Min : nilai minimum setiap kolom.
* 25% : kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
* 50% : kuartil kedua, atau biasa juga disebut median (nilai tengah).
* 75% : kuartil ketiga.
* Max : nilai maksimum.
Adapun outputnya sebagai berikut:

![3-deskripsi-statistik](https://github.com/balle97/model-regresi/assets/128248022/e9c4aa7d-1d89-46b3-873c-15e5c0a98a97)

Dari output di atas terlihat bahwa nilai minimum 0 hanya terdapat pada kolom 'GRS'. Nilai tersebut bermakna bahwa ada beberapa rumah yang tidak memiliki garasi mobil, sehingga itu merupakan hal yang wajar. Jadi, patut diduga bahwa ini merupakan data yang valid atau bukan _missing value_. 

#### 3. Exploratory Data Analysis - Handling Missing Values, Drop Unnecessary Columns, and Duplicates
**a. Handling Missing Values** 

Setelah mengecek nilai minimum 0, langkah selanjutnya adalah menangani _missing value_ menggunakan fungsi `isnull().sum()`. Adapun outputnya sebagai berikut: 

![4-value](https://github.com/balle97/model-regresi/assets/128248022/16658e8e-6b74-4460-967d-df78e8238113) 

Dari output di atas terlihat bahwa semua kolom tidak memiliki _missing value_. 

Ada beberapa teknik untuk mengatasi missing value, antara lain: menghapus atau melakukan drop terhadap data yang hilang, menggantinya dengan mean atau median, serta memprediksi dan mengganti nilainya dengan teknik regresi. Tapi, untuk proyek ini, semua teknik tersebut tidak diperlukan.

**b. Drop Unnecessary Columns** 

Setelah mengecek _missing value_, langkah selanjutnya adalah membuat kolom baru untuk mempermudah melihat harga rumah. Kolom tersebut diberi nama 'HARGA(JUTA)'. Untuk melakukannya, bagi nilai harga dengan nilai 1.000.000 (satu juta) untuk menghilangkan angka 0 sebanyak 6 digit. Adapun outputnya sebagai berikut: 

![5-kolom-baru](https://github.com/balle97/model-regresi/assets/128248022/bb41ad16-28d5-4eb1-9eae-bbffdb7a4b5e) 

Dari output di atas terlihat bahwa kolom pada tabel di atas sekarang berjumlah **9 kolom**.

Setelah itu, menghapus kolom yang tidak diperlukan menggunakan fungsi `drop()`. Kolom yang dimaksud adalah 'NO', 'NAMA HARGA', 'HARGA'. Adapun outputnya sebagai berikut: 

![6-hapus-kolom](https://github.com/balle97/model-regresi/assets/128248022/964d6fe3-15cc-4911-9e33-984ecae261e2) 

Dari output di atas terlihat bahwa kolom pada tabel di atas sekarang tersisa **6 kolom**.

**c. Handling Duplicates** 

Selanjutnya, menangani _duplicate_ data menggunakan fungsi `drop.duplicates()`. Adapun outputnya sebagai berikut: 

![7-duplikasi](https://github.com/balle97/model-regresi/assets/128248022/76133453-ba66-479d-b58c-ab30c8bff815) 

Dari output di atas terlihat bahwa adanya perubahan pada ukuran dataset dari 1.010 baris menjadi **967 baris**. Hal ini menunjukkan bahwa di dalam dataset tersebut terdapat duplikasi data sebesar **43 baris**.

#### 4. Exploratory Data Analysis - Visualizing and Handling Outliers
**a. Visualize Outliers on Numerical Features** 

Pada tahap ini, akan dideteksi outlier menggunakan teknik visualisasi data (*boxplot*). Akan divisualisasikan data numerik dengan *boxplot* secara horizontal. Menurut **Seltman** dalam “*Experimental Design and Analysis*”, *boxplot* menunjukkan ukuran lokasi dan penyebaran, serta memberikan informasi tentang simetri dan *outliers*. Adapun outputnya sebagai berikut: 

![8-outlier-LB](https://github.com/balle97/model-regresi/assets/128248022/291a967a-18bc-4545-b9a0-411359dd4adb) 
![8-outlier-LT](https://github.com/balle97/model-regresi/assets/128248022/a198da21-fd5c-44e7-85ba-6de67335a4af) 
![8-outlier-KT](https://github.com/balle97/model-regresi/assets/128248022/7ee385a7-d1cb-4505-b55a-91e1d85e4265)
![8-outlier-KM](https://github.com/balle97/model-regresi/assets/128248022/fe9b3872-5413-4abb-9808-4e28c93c501e)
![8-outlier-GRS](https://github.com/balle97/model-regresi/assets/128248022/678f5476-7953-46c7-8b59-74f9ef0ab1fd)  

Dari output di atas terlihat bahwa semua kolom memiliki _outliers_. 

**b. Handle Outliers with IQR Method** 

Ada beberapa teknik untuk menangani _outliers_, antara lain: Hypothesis Testing, Z-score method, IQR Method. Untuk proyek ini, akan menggunakan **metode IQR**. Metode IQR akan digunakan untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apapun yang berada di luar batas ini dianggap sebagai outlier.

Menurut **Seltman** dalam “*Experimental Design and Analysis*”, *outliers* yang diidentifikasi oleh boxplot disebut juga **boxplot outliers** yang didefinisikan sebagai data yang nilainya 1.5 QR di atas Q3 atau 1.5 QR di bawah Q1.

Hal pertama yang perlu dilakukan adalah membuat batas bawah dan batas atas. Berikut persamaan matematikanya: 

![1-metode IQR](https://github.com/balle97/model-regresi/assets/128248022/3a12e41a-4c86-4816-82d8-14183b2db0b8) 

Akan diterapkan persamaan di atas ke dalam code berikut: 

`Q1 = df.quantile(0.25)`

`Q3 = df.quantile(0.75)`

`IQR = Q3 - Q1`

`df = df[~((df < (Q1-1.5*IQR)) | (df > (Q3+1.5*IQR))).any(axis=1)]`

Perlu mengecek ukuran dataset setelah di-drop outliers menggunakan fungsi `shape()`. Adapun outputnya sebagai berikut: 

![10-shape](https://github.com/balle97/model-regresi/assets/128248022/fee221e5-8126-460c-9d8b-8a1d48c62699) 

Datasetnya sekarang sudah bersih dan memiliki **667 sampel**.

#### 5. Exploratory Data Analysis - Univariate and Multivariate Analysis
**a. Univariate Analysis** 

Untuk proyek ini, hanya melakukan _univariate analysis_ pada fitur numerik saja. Karena dataset yang digunakan tidak memiliki fitur categorikal. Akan menggunakan histogram untuk melihat masing-masing fitur tersebut. Adapun outputnya sebagai berikut: 

![11-univariate](https://github.com/balle97/model-regresi/assets/128248022/735041c9-c24e-4008-b40d-26619be99c18) 

Lihatlah histogram untuk variabel "HARGA(JUTA)" yang merupakan fitur target (label). Dari histogram "HARGA(JUTA)", dapat diperoleh beberapa informasi, antara lain:
* Peningkatan harga rumah sebanding dengan penurunan jumlah sampel. Hal ini dapat terlihat jelas dari histogram "HARGA(JUTA)" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel pada sumbu y.
* Distribusi harga miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

**b. Multivariate Analysis** 

Untuk proyek ini, hanya melakukan _multivariate analysis_ pada fitur numerik saja. Karena dataset yang digunakan tidak memiliki fitur categorikal. Akan mengamati hubungan antar fitur numerik menggunakan fungsi `pairplot()`. Fungsi pairplot dari library seaborn menunjukkan relasi pasangan dalam dataset. Adapun outputnya sebagai berikut: 

![12-multivariate](https://github.com/balle97/model-regresi/assets/128248022/671147c0-adb0-4595-81db-16df64f11dab) 

Dari output di atas, dapat terlihat relasi antara semua fitur numerik dengan fitur target yaitu ‘HARGA(JUTA)’. 

Untuk membacanya, lihat fitur pada sumbu y, cari fitur target ‘HARGA(JUTA)’, dan lihat grafik relasi antara semua fitur pada sumbu x dengan fitur target ‘HARGA(JUTA)’ pada sumbu y. Dalam hal ini, fitur ‘HARGA(JUTA)’ berada pada baris terakhir pada sumbu y. Jadi, cukup dilihat relasi antar fitur numerik dengan fitur target ‘HARGA(JUTA)’ pada baris tersebut saja.

Pada pola sebaran data grafik pairplot diatas, terlihat ‘LB’, ‘LT’ memiliki korelasi yang tinggi dengan fitur "HARGA(JUTA)". Oleh karena itu, perlu melakukan evaluasi skor korelasi antara fitur numerik dengan fitur target menggunakan fungsi `corr()` dan fungsi `heatmap()`. Adapun outputnya sebagai berikut: 

![13-correlation-matrix](https://github.com/balle97/model-regresi/assets/128248022/d7b6572d-f797-4984-8f82-a4875adafce9) 

Dari output diatas terlihat bahwa koefisien korelasi berkisar antara 0 dan 1. Nilai ini untuk mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah.  

Arah korelasi antara dua variabel bisa bernilai positif (nilai kedua variabel cenderung meningkat bersama-sama) maupun negatif (nilai salah satu variabel cenderung meningkat ketika nilai variabel lainnya menurun).

Lihatlah grafik korelasi di atas. Jika diamati, fitur ‘LB’, ‘LT’ memiliki skor korelasi yang besar (di atas 0.6) dengan fitur target ‘HARGA(JUTA)’. Artinya, fitur 'HARGA(JUTA)' berkorelasi tinggi dengan kedua fitur tersebut.

Selanjutnya, mengecek fitur yang memiliki korelasi tinggi tersebut menggunakan fungsi `pairplot()`. Adapun outputnya sebagai berikut: 

![14-korelasi-tinggi](https://github.com/balle97/model-regresi/assets/128248022/bff09e32-f43c-4052-b12f-9ee841ce1aa3)

## Data Preparation

Ada beberapa tahapan yang umum dilakukan pada data preparation, antara lain, seleksi fitur, transformasi data, _feature engineering_, dan _dimensionality reduction_. Untuk proyek ini, akan dilakukan 2 tahap persiapan data, yaitu:
* Pembagian dataset dengan fungsi train_test_split.
* Transformasi dataset dengan fungsi minmaxscaller.

### 1. Splitting Dataset with Train-Test-Split
Sebelum membuat model, ada tahap yang harus dilakukan terlebih dulu yaitu membagi (_splitting_) dataset menjadi data latih (_train_) dan data uji (_test_). Perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. 

Karena data uji berperan sebagai data baru, maka perlu melakukan semua proses transformasi di dalam data latih. Akan dibagi dataset sebelum melakukan transformasi apapun agar tidak mengotori data uji dengan informasi yang didapat dari data latih. Tujuan dari data uji adalah untuk mengukur kinerja model pada data baru.

Pada proyek ini, akan menggunakan proporsi pembagian sebesar 90:10 dengan fungsi `train_test_split()` dari sklearn. Untuk mengecek jumlah sampel pada masing-masing bagian, gunakan code berikut: 

`print(f'Total sampel pada seluruh dataset: {len(X)}')`

`print(f'Total sampel pada training set: {len(X_train)}')`

`print(f'Total sampel pada testing set: {len(X_test)}')`

Adapun outputnya sebagai berikut: 

![15-splitting](https://github.com/balle97/model-regresi/assets/128248022/a18c1414-cfc4-40bc-b935-d5f83518d39f) 

Dari output di atas terlihat bahwa dataset berjumlah **667 sampel** dari 1.010 sampel. Jumlahnya berkurang karena sudah melewati _Data Cleaning_ pada tahap sebelumnya. Dengan proporsi pembagian 90:10, maka data uji akan berjumlah **67 sampel**. Tentu ini merupakan jumlah yang sudah cukup karena tidak perlu jumlah sampel yang banyak hanya untuk proses pengujian. 

### 2. Scalling Dataset with MinMaxScaller
Setelah membagi dataset, langkah selanjutnya adalah melakukan proses transformasi dataset menggunakan fungsi `MinMaxScaler()` dari library Scikitlearn. MinMaxScaler adalah salah satu teknik preprocessing data yang digunakan dalam machine learning untuk menormalkan atau menstandarisasi fitur-fitur numerik dalam rentang tertentu, biasanya antara 0 dan 1.

Cara kerjanya adalah dengan mengubah setiap nilai fitur dalam dataset menjadi nilai yang sesuai dengan rentang yang ditentukan, di mana nilai terkecil dalam setiap fitur akan diubah menjadi 0, sedangkan nilai terbesar akan diubah menjadi 1, dan nilai-nilai lainnya akan diubah secara proporsional sesuai dengan skala yang ditentukan.

MinMaxScaler membantu mengatasi perbedaan skala antara fitur-fitur yang berbeda dalam dataset, sehingga memungkinkan algoritma machine learning untuk bekerja lebih baik dengan data tersebut. Adapun outputnya sebagai berikut: 

![16-scalling-train](https://github.com/balle97/model-regresi/assets/128248022/5121bc7b-021b-4e4b-9ae2-d742eb808cdf) 

Dari output di atas terlihat bahwa skala fitur sudah relatif sama. Sehingga dataset sudah siap digunakan dalam algoritma machine learning.

## Modeling

Pada tahap ini, akan menggunakan algoritma machine learning untuk menjawab _problem statement_ dari tahap _business understanding_. Akan mengembangkan model machine learning menggunakan tiga algoritma yang berbeda. Setelah itu, Akan mengevaluasi kinerja masing-masing algoritma dan menentukan algoritma yang memberikan hasil prediksi terbaik. Algoritma-algoritma yang akan digunakan adalah sebagai berikut:
* Support Vector Regression
* Random Forest
* Gradient Boosting

### 1. Support Vector Regression (SVR)
Support Vector Regression (SVR) adalah algoritma supervised learning yang digunakan untuk memprediksi nilai variabel kontinu atau digunakan untuk kasus regresi. Algoritma ini menggunakan prinsip yang sama dengan SVM, namun SVM biasanya digunakan pada kasus klasifikasi. 

Pada SVM, algoritma ini berusaha untuk menemukan sebuah "jalan" yang memisahkan sampel-sampel dari kelas berbeda dengan sejauh mungkin. Sementara pada SVR, algoritma ini berusaha untuk menemukan sebuah "jalan" yang dapat menampung sebanyak mungkin sampel di sekitar "jalan" tersebut, dengan tetap memperhatikan margin kesalahan. Berbeda dengan SVM di mana support vector adalah 2 sampel dari 2 kelas berbeda yang memiliki jarak paling dekat. Pada SVR, support vector adalah sampel yang menjadi pembatas jalan yang dapat menampung seluruh sampel pada data. 

Berikut ini beberapa parameter yang digunakan pada model SVR, antara lain:
* `C`: Parameter ini mengontrol penalti terhadap error yang melebihi toleransi epsilon. Nilai yang lebih besar akan meningkatkan penalti, sehingga model lebih cenderung mengikuti data training.

* `epsilon`: Parameter ini menentukan toleransi error yang diizinkan pada model. Nilai yang lebih kecil akan menghasilkan model yang lebih akurat, tetapi bisa menyebabkan overfitting.

* `gamma`: Parameter ini menentukan seberapa sensitif model terhadap perubahan data. Nilai yang tinggi menghasilkan model yang lebih sensitif terhadap perubahan data, dan dapat menyebabkan overfitting, sedangkan nilai yang rendah  menghasilkan model yang kurang sensitif terhadap perubahan data, dan dapat menghasilkan underfitting.

* `kernel`: Parameter ini menentukan fungsi transformasi data yang digunakan untuk memetakan data ke ruang dimensi yang lebih tinggi. Kernel yang umum digunakan dalam SVR:
`'linear'`: Kernel ini cocok untuk data yang memiliki hubungan linear. 
`'poly'`: Kernel ini cocok untuk data yang memiliki hubungan non-linear yang kompleks. 
`'rbf'`: Kernel ini cocok untuk data yang memiliki hubungan non-linear yang halus.

**a. Kelebihan SVR**
* Model keputusan dapat dengan mudah diperbarui dengan data baru.
* Dibandingkan teknik regresi lain, SVR umumnya memiliki kebutuhan komputasi yang lebih rendah.
* Memiliki kemampuan generalisasi yang baik, sehingga performanya optimal pada data baru.
* Bisa digunakan untuk memodelkan hubungan non-linear dengan menggunakan fungsi kernel.
* Secara alami membantu mencegah overfitting dengan meminimalkan margin error.

**b. Kekurangan SVR**
* Performanya dapat menurun, jika jumlah fitur melebihi jumlah sampel data.
* Tidak optimal dalam menangani data yang memiliki banyak noise yaitu kelas target tumpang tindih.
* Tidak cocok untuk dataset yang besar, karena kompleksitas komputasi yang meningkat.
* Lebih sulit untuk diinterpretasi dibandingkan dengan model regresi lainnya.

### 2. Random Forest
Random forest adalah algoritma supervised learning yang digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Algoritma ini termasuk ke dalam kategori ensemble (group) learning. Ada dua teknik pendekatan dalam membuat model ensemble, yaitu **bagging dan boosting**. Bagging atau bootstrap aggregating adalah teknik yang melatih model dengan sampel random. Dalam teknik bagging, sejumlah model dilatih dengan teknik sampling with replacement (proses sampling dengan penggantian). Algoritma yang cocok untuk teknik bagging ini adalah **decision tree**.

Random Forest pada dasarnya adalah versi bagging dari algoritma decision tree. Model decision tree masing-masing memiliki hyperparameter yang berbeda dan dilatih pada beberapa bagian (subset) data yang berbeda juga. Teknik pembagian data pada algoritma decision tree adalah memilih sejumlah fitur dan sejumlah sampel secara acak dari dataset yang terdiri dari _n_ fitur dan _m_ sampel. Pada kasus regresi, prediksi akhir adalah rata-rata prediksi seluruh pohon dalam model ensemble. Jadi, algoritma Random Forest disusun dari banyak algoritma pohon (decision tree) yang pembagian data dan fiturnya dipilih secara acak.

Berikut ini beberapa parameter yang digunakan pada model Random Forest, antara lain:
* `n_estimators`: Parameter ini menentukan jumlah pohon yang akan dibuat dalam hutan. Semakin banyak pohon, semakin akurat modelnya, tetapi juga semakin lama waktu pelatihannya.

* `max_depth`: Parameter ini menentukan seberapa banyak pohon dapat membelah (_splitting_) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. Semakin dalam pohon, semakin kompleks modelnya, tetapi juga semakin mudah untuk overfitting.

* `n_jobs`: Parameter ini menentukan jumlah pekerjaan paralel yang akan digunakan untuk melatih model. Nilai yang lebih tinggi akan mempercepat pelatihan, tetapi membutuhkan lebih banyak memori.

* `random_state`: Parameter ini mengontrol nilai seed acak yang digunakan untuk membangun pohon. Menetapkan nilai yang sama akan menghasilkan model yang sama setiap kali dilatih.

**a. Kelebihan Random Forest**
* Kuat terhadap data outlier (pencilan data).
* Bekerja dengan baik dengan data non-linear.
* Risiko overfitting lebih rendah.
* Berjalan secara efisien pada kumpulan data yang besar.
* Akurasi yang lebih baik daripada algoritma regresi lainnya.

**b. Kekurangan Random Forest**
* Cenderung bias saat berhadapan dengan variabel kategorikal.
* Waktu komputasi pada dataset berskala besar relatif lambat.
* Tidak cocok untuk metode linier dengan banyak fitur sparse (data dengan banyak nilai missing).
* Berpotensi mengalami overfitting pada subset data tertentu, terutama jika jumlah data yang tersedia relatif kecil.
* Membutuhkan memori yang cukup besar untuk menyimpan model, terutama pada dataset yang besar.

### 3. Gradient Boosting
Gradient Boosting adalah sebuah teknik machine learning yang sering digunakan untuk menyelesaikan masalah regresi dan klasifikasi. Teknik ini menggabungkan beberapa model yang lemah (_weak model_) menjadi sebuah model yang kuat. Model lemah ini sering disebut dengan _weak learners_, dan dapat berupa model regresi atau klasifikasi sederhana seperti **Decision Tree**. 

Algoritma ini menggunakan pendekatan iteratif, di mana setiap iterasi bertujuan untuk meningkatkan model sebelumnya dengan menambahkan model baru. Pada setiap iterasi, Gradient Boosting akan menambahkan _weak learner_ baru dan mengoreksi prediksi sebelumnya dengan memperhitungkan kesalahan pada prediksi tersebut. Dalam setiap iterasi, Gradient Boosting memperbarui _residual error_ dengan mengurangi hasil prediksi dari target, lalu menambahkan _weak learner_ baru yang menyelesaikan masalah _residual error_ yang dihasilkan.

Secara matematis, Gradient Boosting mengoptimalkan suatu fungsi objektif dengan mengevaluasi gradient pada setiap titik. Fungsi objektif yang umum digunakan dalam Gradient Boosting adalah fungsi **Mean Squared Error (MSE)** untuk regresi dan fungsi Log-Loss untuk klasifikasi.

Berikut ini beberapa parameter yang digunakan pada model Gradient Boosting, antara lain:
* `learning_rate`: Parameter ini mengontrol pengaruh setiap pohon terhadap hasil akhir. Nilai yang lebih kecil menghasilkan model yang lebih stabil, tetapi membutuhkan waktu lebih lama untuk dilatih.

* `n_estimators`: Parameter ini menentukan jumlah pohon yang akan dibangun dalam model. Semakin banyak pohon, semakin kompleks modelnya dan semakin akurat potensinya, tetapi dapat menyebabkan overfitting.

* `max_depth`: Parameter ini menentukan kedalaman maksimal dari setiap pohon yang dibangun. Pohon yang lebih dalam dapat menangkap pola yang lebih kompleks, tetapi juga lebih rentan terhadap overfitting.

* `subsample`: Parameter ini menentukan proporsi data yang digunakan untuk melatih setiap pohon. Nilai yang lebih kecil dapat membantu mencegah overfitting dan meningkatkan kinerja model pada data baru.

**a. Kelebihan Gradient Boosting**
* Mampu menghasilkan model yang akurat dan kuat, terutama pada data yang kompleks dan tidak terstruktur.
* Mampu menangani data dengan banyak fitur dan interaksi yang kompleks.
* Dapat digunakan untuk berbagai jenis tugas machine learning, seperti klasifikasi, regresi, dan ranking.
* Dapat dikombinasikan dengan berbagai jenis model dasar (_weak learner_) untuk menghasilkan model yang optimal untuk masalah tertentu.

**b. Kekurangan Gradient Boosting**
* Memerlukan tuning parameter yang cermat untuk mendapatkan model yang optimal. Proses tuning bisa memakan waktu dan membutuhkan keahlian.
* Berisiko mengalami overfitting, di mana model terlalu fokus pada data latih dan tidak dapat digeneralisasikan dengan baik pada data baru.
* Lebih kompleks dibandingkan dengan algoritma machine learning lainnya, sehingga membutuhkan waktu dan sumber daya komputasi yang lebih besar.
* Bisa lebih lambat dibandingkan dengan algoritma machine learning lainnya, terutama pada data berukuran besar.

> Saya memilih Random Forest sebagai model terbaik untuk memprediksi harga rumah karena memiliki nilai kesalahan prediksi terendah, sehingga memiliki hasil prediksi yang mendekati nilai harga aslinya.

## Evaluation

Pada tahap ini, akan dilakukan evaluasi model yang sudah dilakukan proses training sebelumnya. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Maka, semua metrik mengukur seberapa kecil nilai eror tersebut.

Metrik yang digunakan di proyek ini adalah **MSE (Mean Squared Error)** yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Berikut persamaannya: 

![2-mse](https://github.com/balle97/model-regresi/assets/128248022/7d15ef95-90a6-4bc2-be25-5da28205ab97)

### 1. Scalling Dataset with MinMaxScaller
Sebelum menghitung nilai MSE dalam model, perlu dilakukan proses scaling fitur numerik pada data uji. Sebelumnya, sudah melakukan proses scaling pada data latih untuk menghindari kebocoran data. Hal ini harus dilakukan agar skala antara data latih dan data uji sama dan agar bisa dilakukan tahap evaluasi model. Adapun outputnya sebagai berikut: 

![17-scalling-test](https://github.com/balle97/model-regresi/assets/128248022/38ea368d-8eda-4740-8824-055106fa03be) 

Dari output di atas terlihat bahwa skala fitur sudah relatif sama. Sehingga dataset sudah bisa digunakan untuk tahap evaluasi model.

### 2. Evaluate Model with MSE
Setelah selesai scalling data uji, langkah selanjutnya adalah dilakukan evaluasi model menggunakan metrik MSE yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Saat menghitung nilai **MSE (Mean Squared Error)** pada data latih dan data uji, akan dibagi dengan nilai 1e3 agar nilai MSE berada dalam skala yang tidak terlalu besar. Adapun outputnya sebagai berikut: 

![18-metrik-mse](https://github.com/balle97/model-regresi/assets/128248022/8bc92771-44ae-4615-93b4-4e9f3fdb070c)

Selanjutnya, membuat plot metrik mse dengan bar chart. Adapun outputnya sebagai berikut: 

![19-plot-metrik-mse](https://github.com/balle97/model-regresi/assets/128248022/da1df910-a0fb-48a0-8842-23c13ca500ff) 

Dari output di atas terlihat bahwa model Random Forest dan Gradient Boosting menghasilkan nilai eror yang paling kecil. Sedangkan model dengan algoritma SVR menghasilkan nilai eror yang paling besar.

Selanjutnya, melakukan proses uji model dengan memprediksi nilai asli(y_true). Adapun outputnya sebagai berikut: 

![20-testing](https://github.com/balle97/model-regresi/assets/128248022/e9aca260-0bba-402f-b0cf-7690354787b7)

Dari output di atas terlihat bahwa prediksi dengan model **Random Forest** dan **Gradient Boosting** memberikan hasil yang paling mendekati dengan harga aslinya. Jadi, Model **Random Forest** yang akan dipilih sebagai model terbaik untuk melakukan prediksi harga rumah.

## Referensi
[[1](https://www.researchgate.net/publication/325656226_KOMPOSISI_HARGA_JUAL_RUMAH_TINGGAL_LAYAK_HUNI_DI_YOGYAKARTA_STUDI_KASUS_PEMBANGUNAN_RUMAH_TIPE_90115_DI_LUAR_KOMPLEKS_PERUMAHAN)] Musyafa, A. (2013). Komposisi Harga Jual Rumah Tinggal Layak Huni Di Yogyakarta (Studi Kasus Pembangunan Rumah Tipe 90/115 di Luar Kompleks Perumahan) (004K). Konferensi Nasional Teknik Sipil 7 (KoNTekS 7), 7-12.

[[2](https://jurnal.untan.ac.id/index.php/justin/article/view/18455)] Hendra, Tursina, & Nyoto, R. D. (2017). Case Base Reasoning Penentuan Harga Rumah dengan Menggunakan Metode Tversky (Studi Kasus : Kota Pontianak). Jurnal Sistem Dan Teknologi Informasi (JUSTIN), 5(2), 75–79.

[[3](https://jurnal.mdp.ac.id/index.php/jatisi/article/view/701)] Saiful, A., Andryana, S. & Gunaryati, A. (2021). Prediksi Harga Rumah Menggunakan Web Scrapping Dan Machine Learning Dengan Algoritma Linear Regression. Jurnal Teknik Informatika dan Sistem Informasi (JATISI), 8(1), 41-50. https://doi.org/10.35957/jatisi.v8i1.701.

[[4](https://ejournal.itn.ac.id/index.php/jati/article/view/6343)] Haryanto, C., Rahaningsih, N. & Basysyar, F. M. (2023). Komparasi Algoritma Machine Learning dalam Memprediksi Harga Rumah. JATI (Jurnal Mahasiswa Teknik Informatika), 7(1), 533-539. https://doi.org/10.36040/jati.v7i1.6343.

# Laporan Proyek Machine Learning - MUHAMMAD IQBAL

## Domain Proyek

> Domain proyek yang dipilih adalah mengenai bisnis properti, khususnya rumah, dengan judul proyek **"Prediksi Harga Rumah di Kecamatan Tebet"**.


### Latar Belakang Masalah

Rumah adalah salah satu dari tiga kebutuhan dasar manusia, selain kebutuhan sandang dan pangan. Sejalan dengan konsep Hierarki Kebutuhan menurut **Maslow** yang menyebutkan bahwa rumah sebagai bagian dari kebutuhan fisiologis manusia. Seiring perkembangan waktu, kebutuhan fisiologis manusia semakin meningkat, termasuk keinginan untuk memiliki rumah. Dengan meningkatnya daya beli masyarakat, harga rumah terus mengalami peningkatan. Para pengembang properti bersaing untuk membangun properti rumah sebagai bentuk investasi. Namun, fluktuasi harga rumah yang sulit diprediksi dengan pasti meningkatkan risiko investasi. 

Menurut **Albani Musyafa (2013)**, harga rumah dipengaruhi oleh faktor lokasi dan biaya lahan [[1](https://www.researchgate.net/publication/325656226_KOMPOSISI_HARGA_JUAL_RUMAH_TINGGAL_LAYAK_HUNI_DI_YOGYAKARTA_STUDI_KASUS_PEMBANGUNAN_RUMAH_TIPE_90115_DI_LUAR_KOMPLEKS_PERUMAHAN)]. Namun, Menurut **Hendra et al. (2017)** menambahkan bahwa faktor akses transportasi juga berpengaruh [[2](https://jurnal.untan.ac.id/index.php/justin/article/view/18455)]. Selain itu, penelitian dari **Saiful et al. (2021)** menunjukkan bahwa penggunaan machine learning dengan algoritma linear regression dapat memberikan prediksi harga rumah dengan tingkat akurasi yang memuaskan [[3](https://jurnal.mdp.ac.id/index.php/jatisi/article/view/701)]. Namun, penelitian dari **Haryanto et al. (2023)** menunjukkan bahwa algoritma random forest regression memberikan akurasi yang lebih tinggi dalam memprediksi harga rumah dibandingkan dengan metode lainnya [[4](https://ejournal.itn.ac.id/index.php/jati/article/view/6343)]. 

Oleh karena itu, penting bagi pengembang properti untuk memiliki pemahaman mendalam mengenai faktor-faktor yang memengaruhi harga rumah di kecamatan Tebet. Prediksi akan digunakan untuk menentukan berapa harga yang pantas untuk properti rumah berdasarkan pertimbangan faktor-faktor tertentu sehingga pengembang properti bisa mendapatkan profit sebesar mungkin. Tidak adanya acuan harga rumah, seperti acuan harga emas, mendorong kebutuhan akan sistem prediksi yang dapat membantu mengurangi risiko kerugian dalam pengembangan properti. Sehingga, pada proyek ini akan dibuat model prediksi untuk memprediksi harga rumah di kecamatan Tebet dengan algoritma machine learning regresi, seperti _Support Vector Regression_ (SVR), _Random Forest_, dan _Gradient Boosting_.


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

Untuk mencapai tujuan tersebut, beberapa solusi akan diberikan, diantaranya:
- Menganalisis data dengan melakukan penanganan _missing values_, penanganan _duplicates_, penanganan _outliers_, _univariate analysis_ dan _multivariate analysis_ serta penggunaan fungsi _corr_ dan _heatmap_ untuk mengetahui korelasi antar faktor penentu.
- Menggunakan beberapa algoritma machine learning seperti _Support Vector Regression_ (SVR), _Random Forest_, dan _Gradient Boosting_ untuk mengembangkan model. Dari ketiga model ini, akan dipilih satu model yang memiliki nilai kesalahan prediksi terendah. Intinya, tujuannya adalah menciptakan model yang seakurat mungkin dengan menghasilkan nilai kesalahan yang seminimal mungkin. Menggunakan metrik evaluasi seperti _Mean Squared Error (MSE)_ untuk mengevaluasi kinerja model dalam memprediksi harga. Secara keseluruhan, metrik ini digunakan untuk mengukur seberapa akurat model dalam memperkirakan nilai yang sebenarnya.


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
Pada tahap ini, yang perlu dilakukan pertama kali adalah memuat dataset berformat Excel ke dalam dataframe dengan mengimpor _library_ **pandas** menggunakan fungsi `read_excel()` yang tersedia dalam _library_ tersebut. Dataset yang digunakan bernama **DATA RUMAH.xlsx**. Adapun outputnya sebagai berikut: 

Tabel 1. Memuat Dataset Berformat Excel ke dalam Dataframe Menggunakan Fungsi `read_excel()`
|      | NO   | NAMA RUMAH                                        | HARGA       | LB  | LT  | KT  | KM  | GRS |
|------|------|---------------------------------------------------|-------------|-----|-----|-----|-----|-----|
| 0    | 1    | Rumah Murah Hook Tebet Timur, Tebet, Jakarta S... | 3800000000  | 220 | 220 | 3   | 3   | 0   |
| 1    | 2    | Rumah Modern di Tebet dekat Stasiun, Tebet, Ja... | 4600000000  | 180 | 137 | 4   | 3   | 2   |
| 2    | 3    | Rumah Mewah 2 Lantai Hanya 3 Menit Ke Tebet, T... | 3000000000  | 267 | 250 | 4   | 4   | 4   |
| 3    | 4    | Rumah Baru Tebet, Tebet, Jakarta Selatan          | 430000000   | 40  | 25  | 2   | 2   | 0   |
| 4    | 5    | Rumah Bagus Tebet komp Gudang Peluru lt 350m, ... | 9000000000  | 400 | 355 | 6   | 5   | 3   |
| ...  | ...  | ...                                               | ...         | ... | ... | ... | ... | ... |
| 1005 | 1006 | Rumah Strategis Akses Jalan 2mobil Di Menteng ... | 9000000000  | 450 | 550 | 10  | 10  | 3   |
| 1006 | 1007 | Tebet Rumah Siap Huni Jln 2 Mbl Nyaman            | 4000000000  | 160 | 140 | 4   | 3   | 2   |
| 1007 | 1008 | Di Kebun Baru Rumah Terawat, Area Strategis       | 4000000000  | 139 | 230 | 4   | 4   | 1   |
| 1008 | 1009 | Dijual Cepat Rumah Komp Depkeu Dr Soepomo Tebe... | 19000000000 | 360 | 606 | 7   | 4   | 0   |
| 1009 | 1010 | Dijual Rumah Kokoh Di Gudang Peluru               | 10500000000 | 420 | 430 | 7   | 4   | 2   |

1010 rows × 8 columns


Dari output di atas, terlihat bahwa:
* Terdapat 1.010 baris (_records_ atau jumlah pengamatan) dalam dataset.
* Terdapat 8 kolom yaitu: NO, NAMA RUMAH, HARGA, LB, LT, KT, KM, GRS.


#### 2. Exploratory Data Analysis - Description of Variables
Pada tahap ini, informasi pada dataset akan diperiksa dengan menggunakan fungsi `info()`. Adapun outputnya sebagai berikut: 

Tabel 2. Informasi pada Dataset Diperiksa Menggunakan Fungsi `info()`
| # | Column     | Non-Null Count | Dtype  |
|---|------------|----------------|--------|
| 0 | NO         | 1010 non-null  | int64  |
| 1 | NAMA RUMAH | 1010 non-null  | object |
| 2 | HARGA      | 1010 non-null  | int64  |
| 3 | LB         | 1010 non-null  | int64  |
| 4 | LT         | 1010 non-null  | int64  |
| 5 | KT         | 1010 non-null  | int64  |
| 6 | KM         | 1010 non-null  | int64  |
| 7 | GRS        | 1010 non-null  | int64  |

dtypes: int64(7), object(1)


Dari output di atas, terlihat bahwa:
* Terdapat 1 kolom dengan tipe data object, yaitu: NAMA RUMAH. Kolom ini merupakan fitur non-numerik, tapi bukan fitur kategorikal.
* Terdapat 6 kolom numerik dengan tipe data int64 yaitu: NO, LB, LT, KT, KM, GRS. Kolom ini merupakan fitur numerik.
* Terdapat 1 kolom numerik dengan tipe data int64, yaitu: HARGA. Kolom ini merupakan target fitur.


Deskripsi statistik dataset diperiksa dengan menggunakan fungsi `describe()` untuk memastikan apakah ada nilai minimum 0 pada setiap kolom tersebut. 

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

Tabel 3. Deskripsi Statistik Dataset Diperiksa Menggunakan Fungsi `describe()`
|       | NO          | HARGA        | LB          | LT          | KT          | KM          | GRS         |
|-------|-------------|--------------|-------------|-------------|-------------|-------------|-------------|
| count | 1010.000000 | 1.010000e+03 | 1010.000000 | 1010.000000 | 1010.000000 | 1010.000000 | 1010.000000 |
| mean  | 505.500000  | 7.628987e+09 | 276.539604  | 237.432673  | 4.668317    | 3.607921    | 1.920792    |
| std   | 291.706188  | 7.340946e+09 | 177.864557  | 179.957604  | 1.572776    | 1.420066    | 1.510998    |
| min   | 1.000000    | 4.300000e+08 | 40.000000   | 25.000000   | 2.000000    | 1.000000    | 0.000000    |
| 25%   | 253.250000  | 3.262500e+09 | 150.000000  | 130.000000  | 4.000000    | 3.000000    | 1.000000    |
| 50%   | 505.500000  | 5.000000e+09 | 216.500000  | 165.000000  | 4.000000    | 3.000000    | 2.000000    |
| 75%   | 757.750000  | 9.000000e+09 | 350.000000  | 290.000000  | 5.000000    | 4.000000    | 2.000000    |
| max   | 1010.000000 | 6.500000e+10 | 1126.000000 | 1400.000000 | 10.000000   | 10.000000   | 10.000000   |

Dari output di atas, terlihat bahwa nilai minimum 0 hanya ditemukan pada kolom 'GRS'. Nilai tersebut menunjukkan bahwa ada beberapa rumah yang tidak memiliki garasi mobil, sehingga hal tersebut dianggap wajar. Oleh karena itu, dapat disimpulkan bahwa ini merupakan data yang valid atau bukan _missing value_. 


#### 3. Exploratory Data Analysis - Handling Missing Values, Drop Unnecessary Columns, and Duplicates 

**a. Handling Missing Values** 

Setelah nilai minimum 0 diperiksa, langkah selanjutnya adalah menangani _missing value_ dengan menggunakan fungsi `isnull().sum()`. Adapun outputnya sebagai berikut: 

![4-value](https://github.com/balle97/model-regresi/assets/128248022/eeb8996c-8db1-4a31-987e-156cffb7e399) 

Gambar 1. Menangani _Missing Value_ Menggunakan Fungsi `isnull().sum()`


Dari output di atas, terlihat bahwa semua kolom tidak memiliki _missing value_. 

Ada beberapa teknik untuk mengatasi _missing value_, antara lain: menghapus atau melakukan drop terhadap data yang hilang, menggantinya dengan mean atau median, serta memprediksi dan mengganti nilainya dengan teknik regresi. Tapi, untuk proyek ini, semua teknik tersebut tidak diperlukan.


**b. Drop Unnecessary Columns** 

Setelah _missing value_ diperiksa, langkah selanjutnya adalah pembuatan kolom baru untuk mempermudah visualisasi harga rumah. Kolom tersebut diberi nama 'HARGA(JUTA)'. Untuk melakukan ini, nilai harga dibagi dengan nilai 1.000.000 (satu juta) untuk menghilangkan angka 0 sebanyak 6 digit. Adapun outputnya sebagai berikut: 

Tabel 4. Membuat Kolom Baru untuk Mempermudah Visualisasi Harga Rumah
|      | NO   | NAMA RUMAH                                        | HARGA       | LB  | LT  | KT  | KM  | GRS | HARGA(JUTA) |
|------|------|---------------------------------------------------|-------------|-----|-----|-----|-----|-----|-------------|
| 0    | 1    | Rumah Murah Hook Tebet Timur, Tebet, Jakarta S... | 3800000000  | 220 | 220 | 3   | 3   | 0   | 3800.0      |
| 1    | 2    | Rumah Modern di Tebet dekat Stasiun, Tebet, Ja... | 4600000000  | 180 | 137 | 4   | 3   | 2   | 4600.0      |
| 2    | 3    | Rumah Mewah 2 Lantai Hanya 3 Menit Ke Tebet, T... | 3000000000  | 267 | 250 | 4   | 4   | 4   | 3000.0      |
| 3    | 4    | Rumah Baru Tebet, Tebet, Jakarta Selatan          | 430000000   | 40  | 25  | 2   | 2   | 0   | 430.0       |
| 4    | 5    | Rumah Bagus Tebet komp Gudang Peluru lt 350m, ... | 9000000000  | 400 | 355 | 6   | 5   | 3   | 9000.0      |
| ...  | ...  | ...                                               | ...         | ... | ... | ... | ... | ... | ...         |
| 1005 | 1006 | Rumah Strategis Akses Jalan 2mobil Di Menteng ... | 9000000000  | 450 | 550 | 10  | 10  | 3   | 9000.0      |
| 1006 | 1007 | Tebet Rumah Siap Huni Jln 2 Mbl Nyaman            | 4000000000  | 160 | 140 | 4   | 3   | 2   | 4000.0      |
| 1007 | 1008 | Di Kebun Baru Rumah Terawat, Area Strategis       | 4000000000  | 139 | 230 | 4   | 4   | 1   | 4000.0      |
| 1008 | 1009 | Dijual Cepat Rumah Komp Depkeu Dr Soepomo Tebe... | 19000000000 | 360 | 606 | 7   | 4   | 0   | 19000.0     |
| 1009 | 1010 | Dijual Rumah Kokoh Di Gudang Peluru               | 10500000000 | 420 | 430 | 7   | 4   | 2   | 10500.0     | 

1010 rows × 9 columns


Dari output di atas, terlihat bahwa kolom pada tabel di atas sekarang berjumlah **9 kolom**.


Kemudian, kolom yang tidak diperlukan dihapus menggunakan fungsi `drop()`. Kolom yang dimaksud adalah 'NO', 'NAMA HARGA', 'HARGA'. Adapun outputnya sebagai berikut: 

Tabel 5. Kolom yang Tidak Diperlukan Dihapus Menggunakan Fungsi `drop()`
|      | LB  | LT  | KT  | KM  | GRS | HARGA(JUTA) |
|------|-----|-----|-----|-----|-----|-------------|
| 0    | 220 | 220 | 3   | 3   | 0   | 3800.0      |
| 1    | 180 | 137 | 4   | 3   | 2   | 4600.0      |
| 2    | 267 | 250 | 4   | 4   | 4   | 3000.0      |
| 3    | 40  | 25  | 2   | 2   | 0   | 430.0       |
| 4    | 400 | 355 | 6   | 5   | 3   | 9000.0      |
| ...  | ... | ... | ... | ... | ... | ...         |
| 1005 | 450 | 550 | 10  | 10  | 3   | 9000.0      |
| 1006 | 160 | 140 | 4   | 3   | 2   | 4000.0      |
| 1007 | 139 | 230 | 4   | 4   | 1   | 4000.0      |
| 1008 | 360 | 606 | 7   | 4   | 0   | 19000.0     |
| 1009 | 420 | 430 | 7   | 4   | 2   | 10500.0     | 

1010 rows × 6 columns


Dari output di atas, terlihat bahwa kolom pada tabel di atas sekarang tersisa **6 kolom**.


**c. Handling Duplicates** 

Selanjutnya, _duplicate_ data akan ditangani dengan menggunakan fungsi `drop_duplicates()`. Adapun outputnya sebagai berikut: 

![7-duplikasi](https://github.com/balle97/model-regresi/assets/128248022/76133453-ba66-479d-b58c-ab30c8bff815) 

Gambar 2. Menangani Duplikasi Data Menggunakan Fungsi `drop_duplicates()`


Dari output di atas, terlihat bahwa terjadi perubahan pada ukuran dataset dari 1.010 baris menjadi **967 baris**. Hal ini menunjukkan bahwa terdapat duplikasi data sebesar **43 baris** dalam dataset tersebut.


#### 4. Exploratory Data Analysis - Visualizing and Handling Outliers 

**a. Visualize Outliers on Numerical Features** 

Pada tahap ini, _outlier_ akan dideteksi menggunakan teknik visualisasi data (_boxplot_). Data numerik akan divisualisasikan dengan _boxplot_ secara horizontal. Menurut **Seltman** dalam “_Experimental Design and Analysis_”, _boxplot_ menunjukkan ukuran lokasi dan penyebaran, serta memberikan informasi tentang simetri dan _outliers_. Adapun outputnya sebagai berikut: 

![8-outlier-LB](https://github.com/balle97/model-regresi/assets/128248022/291a967a-18bc-4545-b9a0-411359dd4adb) 
![8-outlier-LT](https://github.com/balle97/model-regresi/assets/128248022/a198da21-fd5c-44e7-85ba-6de67335a4af) 
![8-outlier-KT](https://github.com/balle97/model-regresi/assets/128248022/7ee385a7-d1cb-4505-b55a-91e1d85e4265)
![8-outlier-KM](https://github.com/balle97/model-regresi/assets/128248022/fe9b3872-5413-4abb-9808-4e28c93c501e)
![8-outlier-GRS](https://github.com/balle97/model-regresi/assets/128248022/678f5476-7953-46c7-8b59-74f9ef0ab1fd)  

Gambar 3. Outlier Dideteksi Menggunakan Teknik Visualisasi Data (*Boxplot*)


Dari output di atas, terlihat bahwa semua kolom memiliki _outliers_. 


**b. Handle Outliers with IQR Method** 

Ada beberapa teknik untuk menangani _outliers_, antara lain: Hypothesis Testing, Z-score method, IQR Method. Dalam proyek ini, **Metode IQR** akan digunakan untuk mengidentifikasi _outlier_ yang berada di luar Q1 dan Q3. Nilai apapun yang berada di luar batas ini dianggap sebagai _outlier_.

Menurut **Seltman** dalam “*Experimental Design and Analysis*”, *outliers* yang diidentifikasi oleh boxplot disebut juga **boxplot outliers** yang didefinisikan sebagai data yang nilainya 1.5 QR di atas Q3 atau 1.5 QR di bawah Q1.

Batas bawah dan batas atas perlu dibuat terlebih dahulu. Berikut adalah persamaan matematika yang digunakan untuk itu: 

$$Batas Bawah =  Q1 - 1.5 * IQR$$

$$Batas Atas =  Q3 + 1.5 * IQR$$


Ukuran dataset perlu diperiksa setelah dilakukan penghapusan _outlier_ dengan menggunakan fungsi `shape()`. Adapun outputnya sebagai berikut: 

![10-shape](https://github.com/balle97/model-regresi/assets/128248022/fee221e5-8126-460c-9d8b-8a1d48c62699) 

Gambar 4. Ukuran Dataset Diperiksa Menggunakan Fungsi `shape()`


Sekarang, datasetnya sudah bersih dan memiliki **667 sampel**.


#### 5. Exploratory Data Analysis - Univariate and Multivariate Analysis

**a. Univariate Analysis** 

Untuk proyek ini, hanya akan dilakukan _univariate analysis_ pada fitur numerik saja, karena dataset yang digunakan tidak memiliki fitur kategorikal. Histogram akan digunakan untuk melihat masing-masing fitur tersebut. Adapun outputnya sebagai berikut: 

![11-univariate](https://github.com/balle97/model-regresi/assets/128248022/735041c9-c24e-4008-b40d-26619be99c18) 

Gambar 5. Histogram Digunakan untuk Melihat Masing-Masing Fitur Numerik


Dari output di atas, histogram untuk variabel "HARGA(JUTA)" yang merupakan fitur target (label) bisa dilihat. Dari histogram "HARGA(JUTA)", beberapa informasi dapat diperoleh, antara lain:
* Peningkatan harga rumah sebanding dengan penurunan jumlah sampel. Hal ini dapat terlihat jelas dari histogram "HARGA(JUTA)" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel pada sumbu y.
* Distribusi harga miring ke kanan (_right-skewed_). Hal ini akan berimplikasi pada model.


**b. Multivariate Analysis** 

Untuk proyek ini, hanya akan dilakukan _multivariate analysis_ pada fitur numerik saja, karena dataset yang digunakan tidak memiliki fitur kategorikal. Hubungan antar fitur numerik akan diamati menggunakan fungsi `pairplot()`. Relasi pasangan dalam dataset ditunjukkan oleh fungsi pairplot dari library seaborn. Adapun outputnya sebagai berikut: 

![12-multivariate](https://github.com/balle97/model-regresi/assets/128248022/671147c0-adb0-4595-81db-16df64f11dab) 

Gambar 6. Hubungan Antar Fitur Numerik Diamati Menggunakan Fungsi `pairplot()`.


Dari output di atas, relasi antara semua fitur numerik dengan fitur target 'HARGA(JUTA)' dapat terlihat. Untuk membacanya, fitur target 'HARGA(JUTA)' dapat ditemukan dengan melihat fitur pada sumbu y, dan lihat grafik relasi antara semua fitur pada sumbu x dengan fitur target ‘HARGA(JUTA)’ pada sumbu y. Dalam hal ini, fitur ‘HARGA(JUTA)’ berada pada baris terakhir pada sumbu y. Sehingga, relasi antar fitur numerik dengan fitur target 'HARGA(JUTA)' hanya perlu dilihat pada baris tersebut saja.


Dalam pola sebaran data grafik pairplot di atas, terlihat bahwa 'LB' dan 'LT' memiliki korelasi yang tinggi dengan fitur "HARGA(JUTA)". Oleh karena itu, evaluasi skor korelasi antara fitur numerik dengan fitur target perlu dilakukan menggunakan fungsi `corr()` dan fungsi `heatmap()`. Adapun outputnya sebagai berikut: 

![13-correlation-matrix](https://github.com/balle97/model-regresi/assets/128248022/d7b6572d-f797-4984-8f82-a4875adafce9) 

Gambar 7. Evaluasi Skor Korelasi antara Fitur Numerik dengan Fitur Target Menggunakan Fungsi `corr()` dan Fungsi `heatmap()`


Dari output diatas, terlihat bahwa koefisien korelasi berkisar antara 0 dan 1. Nilai ini digunakan untuk mengetahui kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilai korelasi ke 1, semakin kuat korelasinya. Sementara itu, semakin dekat nilai korelasi ke 0, semakin lemah korelasinya. 

Arah korelasi antara dua variabel dapat bernilai positif (di mana nilai kedua variabel cenderung meningkat bersama-sama) maupun negatif (di mana nilai salah satu variabel cenderung meningkat ketika nilai variabel lainnya menurun). 

Lihatlah grafik korelasi di atas. Jika diamati, fitur ‘LB’, ‘LT’ memiliki skor korelasi yang besar (di atas 0.6) dengan fitur target ‘HARGA(JUTA)’. Artinya, fitur 'HARGA(JUTA)' berkorelasi tinggi dengan kedua fitur tersebut.


## Data Preparation

Ada beberapa tahapan yang umum dilakukan pada data preparation, antara lain, seleksi fitur, transformasi data, _feature engineering_, dan _dimensionality reduction_. Dalam proyek ini, dua tahap persiapan data akan dilakukan, yaitu:
* Pembagian dataset menggunakan fungsi train_test_split.
* Transformasi dataset menggunakan fungsi MinMaxScaler.


### 1. Splitting Dataset with Train-Test-Split
Sebelum model dibuat, tahap yang harus dilakukan terlebih dahulu adalah membagi (_splitting_) dataset menjadi data latih (_train_) dan data uji (_test_). Perlunya mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. 

Semua proses transformasi perlu dilakukan di dalam data latih karena data uji berperan sebagai data baru. Dataset akan dibagi sebelum melakukan transformasi apapun agar data uji tidak terkontaminasi dengan informasi yang diperoleh dari data latih. Tujuan dari data uji adalah untuk mengukur kinerja model pada data baru.

Proporsi pembagian sebesar 90:10 akan digunakan dalam proyek ini dengan menggunakan fungsi `train_test_split()` dari sklearn. Adapun outputnya sebagai berikut: 

![15-splitting](https://github.com/balle97/model-regresi/assets/128248022/a18c1414-cfc4-40bc-b935-d5f83518d39f) 

Gambar 8. Proporsi Pembagian Dataset Sebesar 90:10 Menggunakan Fungsi `train_test_split()`


Dari output di atas, terlihat bahwa dataset berjumlah **667** sampel dari 1.010 sampel. Jumlahnya berkurang karena sudah melewati _Data Cleaning_ pada tahap sebelumnya. Dengan proporsi pembagian 90:10, maka data uji akan berjumlah **67 sampel**. Tentu ini merupakan jumlah yang sudah cukup karena tidak perlu jumlah sampel yang banyak hanya untuk proses pengujian. 


### 2. Scalling Dataset with MinMaxScaller
Setelah dataset dibagi, langkah selanjutnya adalah melakukan proses transformasi dataset dengan menggunakan fungsi `MinMaxScaler()` dari library Scikit-learn. 

MinMaxScaler adalah salah satu teknik preprocessing data yang digunakan dalam machine learning untuk menormalkan atau menstandarisasi fitur numerik dalam rentang tertentu, biasanya antara 0 dan 1.

Cara kerjanya adalah dengan mengubah setiap nilai fitur dalam dataset menjadi nilai yang sesuai dengan rentang yang ditentukan, di mana nilai terkecil dalam setiap fitur akan diubah menjadi 0, sedangkan nilai terbesar akan diubah menjadi 1, dan nilai-nilai lainnya akan diubah secara proporsional sesuai dengan skala yang ditentukan.

Perbedaan skala antara fitur-fitur yang berbeda dalam dataset dapat diatasi dengan bantuan MinMaxScaler, sehingga memungkinkan algoritma machine learning untuk bekerja lebih baik dengan data tersebut. Adapun outputnya sebagai berikut: 

Tabel 6. Transformasi Dataset Menggunakan Fungsi `MinMaxScaler()`
|     | LB       | LT       | KT       | KM       | GRS      |
|-----|----------|----------|----------|----------|----------|
| 410 | 0.133898 | 0.073826 | 0.000000 | 0.000000 | 0.666667 |
| 85  | 0.498305 | 0.348993 | 0.333333 | 0.666667 | 0.333333 |
| 184 | 0.328814 | 0.241611 | 0.666667 | 0.333333 | 0.666667 |
| 47  | 0.501695 | 0.561521 | 0.333333 | 0.333333 | 0.000000 |
| 96  | 0.752542 | 0.774049 | 1.000000 | 1.000000 | 1.000000 |
| ... | ...      | ...      | ...      | ...      | ...      |
| 98  | 0.210169 | 0.163311 | 0.333333 | 0.666667 | 0.333333 |
| 148 | 0.328814 | 0.239374 | 0.000000 | 0.333333 | 0.666667 |
| 378 | 0.244068 | 0.203579 | 0.000000 | 0.333333 | 0.666667 |
| 643 | 0.515254 | 0.192394 | 0.333333 | 0.666667 | 0.666667 |
| 142 | 0.210169 | 0.174497 | 0.333333 | 0.333333 | 0.666667 |

600 rows × 5 columns


Dari output di atas, terlihat bahwa skala fitur sudah relatif sama. Sehingga dataset sudah siap digunakan dalam algoritma machine learning.


## Modeling

Pada tahap ini, algoritma machine learning akan digunakan untuk menjawab _problem statement_ dari tahap _business understanding_. Model machine learning akan dikembangkan menggunakan tiga algoritma yang berbeda. Kinerja masing-masing algoritma akan dievaluasi dan algoritma yang memberikan hasil prediksi terbaik akan ditentukan. Algoritma-algoritma yang akan digunakan adalah sebagai berikut:
* Support Vector Regression
* Random Forest
* Gradient Boosting


### 1. Support Vector Regression (SVR)
_Support Vector Regression (SVR)_ adalah algoritma _supervised learning_ yang digunakan untuk memprediksi nilai variabel kontinu atau digunakan untuk kasus regresi. Algoritma ini menggunakan prinsip yang sama dengan _SVM_, namun _SVM_ biasanya digunakan pada kasus klasifikasi. 

Pada **SVM**, algoritma ini berusaha untuk menemukan sebuah "jalan" yang memisahkan sampel-sampel dari kelas berbeda dengan sejauh mungkin. Sementara pada **SVR**, algoritma ini berusaha untuk menemukan sebuah "jalan" yang dapat menampung sebanyak mungkin sampel di sekitar "jalan" tersebut, dengan tetap memperhatikan margin kesalahan. Berbeda dengan **SVM** di mana _support vector_ adalah 2 sampel dari 2 kelas berbeda yang memiliki jarak paling dekat. Pada **SVR**, _support vector_ adalah sampel yang menjadi pembatas jalan yang dapat menampung seluruh sampel pada data. 

Berikut ini beberapa parameter yang digunakan pada model _SVR_, antara lain:
* `C`: Parameter ini mengontrol penalti terhadap error yang melebihi toleransi epsilon. Nilai yang lebih besar akan meningkatkan penalti, sehingga model lebih cenderung mengikuti data training.
* `epsilon`: Parameter ini menentukan toleransi error yang diizinkan pada model. Nilai yang lebih kecil akan menghasilkan model yang lebih akurat, tetapi bisa menyebabkan overfitting.
* `gamma`: Parameter ini menentukan seberapa sensitif model terhadap perubahan data. Nilai yang tinggi menghasilkan model yang lebih sensitif terhadap perubahan data, dan dapat menyebabkan overfitting, sedangkan nilai yang rendah  menghasilkan model yang kurang sensitif terhadap perubahan data, dan dapat menghasilkan underfitting.
* `kernel`: Parameter ini menentukan fungsi transformasi data yang digunakan untuk memetakan data ke ruang dimensi yang lebih tinggi. Kernel yang umum digunakan dalam SVR:
`'linear'`: Kernel ini cocok untuk data yang memiliki hubungan linear. 
`'poly'`: Kernel ini cocok untuk data yang memiliki hubungan non-linear yang kompleks. 
`'rbf'`: Kernel ini cocok untuk data yang memiliki hubungan non-linear yang halus.

**a. Kelebihan SVR**
* Model keputusan dapat dengan mudah diperbarui dengan data baru.
* Dibandingkan teknik regresi lain, _SVR_ umumnya memiliki kebutuhan komputasi yang lebih rendah.
* Memiliki kemampuan generalisasi yang baik, sehingga performanya optimal pada data baru.
* Bisa digunakan untuk memodelkan hubungan non-linear dengan menggunakan fungsi kernel.
* Secara alami membantu mencegah _overfitting_ dengan meminimalkan _margin error_.

**b. Kekurangan SVR**
* Performanya dapat menurun, jika jumlah fitur melebihi jumlah sampel data.
* Tidak optimal dalam menangani data yang memiliki banyak _noise_ yaitu kelas target tumpang tindih.
* Tidak cocok untuk dataset yang besar, karena kompleksitas komputasi yang meningkat.
* Lebih sulit untuk diinterpretasi dibandingkan dengan model regresi lainnya.


### 2. Random Forest
_Random Forest_ adalah algoritma _supervised learning_ yang digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Algoritma ini termasuk ke dalam kategori _ensemble (group) learning_. Ada dua teknik pendekatan dalam membuat model _ensemble_, yaitu **bagging dan boosting**. _Bagging atau bootstrap aggregating_ adalah teknik yang melatih model dengan _sampel random_. Dalam teknik _bagging_, sejumlah model dilatih dengan teknik _sampling with replacement_ (proses sampling dengan penggantian). Algoritma yang cocok untuk teknik _bagging_ ini adalah **decision tree**.

_Random Forest_ pada dasarnya adalah versi _bagging_ dari algoritma _decision tree_. Model _decision tree_ masing-masing memiliki _hyperparameter_ yang berbeda dan dilatih pada beberapa bagian (_subset_) data yang berbeda juga. Teknik pembagian data pada algoritma _decision tree_ adalah memilih sejumlah fitur dan sejumlah sampel secara acak dari dataset yang terdiri dari _n_ fitur dan _m_ sampel. Pada kasus regresi, prediksi akhir adalah rata-rata prediksi seluruh pohon dalam model _ensemble_. Jadi, algoritma _Random Forest_ disusun dari banyak algoritma pohon (_decision tree_) yang pembagian data dan fiturnya dipilih secara acak.

Berikut ini beberapa parameter yang digunakan pada model _Random Forest_, antara lain:
* `n_estimators`: Parameter ini menentukan jumlah pohon yang akan dibuat dalam hutan. Semakin banyak pohon, semakin akurat modelnya, tetapi juga semakin lama waktu pelatihannya.
* `max_depth`: Parameter ini menentukan seberapa banyak pohon dapat membelah (_splitting_) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. Semakin dalam pohon, semakin kompleks modelnya, tetapi juga semakin mudah untuk overfitting.
* `n_jobs`: Parameter ini menentukan jumlah pekerjaan paralel yang akan digunakan untuk melatih model. Nilai yang lebih tinggi akan mempercepat pelatihan, tetapi membutuhkan lebih banyak memori.
* `random_state`: Parameter ini mengontrol nilai seed acak yang digunakan untuk membangun pohon. Menetapkan nilai yang sama akan menghasilkan model yang sama setiap kali dilatih.

**a. Kelebihan Random Forest**
* Kuat terhadap data _outlier_ (pencilan data).
* Bekerja dengan baik dengan data non-linear.
* Risiko _overfitting_ lebih rendah.
* Berjalan secara efisien pada kumpulan data yang besar.
* Akurasi yang lebih baik daripada algoritma regresi lainnya.

**b. Kekurangan Random Forest**
* Cenderung bias saat berhadapan dengan variabel kategorikal.
* Waktu komputasi pada dataset berskala besar relatif lambat.
* Tidak cocok untuk metode linier dengan banyak fitur sparse (data dengan banyak nilai missing).
* Berpotensi mengalami _overfitting_ pada _subset_ data tertentu, terutama jika jumlah data yang tersedia relatif kecil.
* Membutuhkan memori yang cukup besar untuk menyimpan model, terutama pada dataset yang besar.


### 3. Gradient Boosting
_Gradient Boosting_ adalah sebuah teknik machine learning yang sering digunakan untuk menyelesaikan masalah regresi dan klasifikasi. Teknik ini menggabungkan beberapa model yang lemah (_weak model_) menjadi sebuah model yang kuat. Model lemah ini sering disebut dengan _weak learners_, dan dapat berupa model regresi atau klasifikasi sederhana seperti **Decision Tree**. 

Algoritma ini menggunakan pendekatan iteratif, di mana setiap iterasi bertujuan untuk meningkatkan model sebelumnya dengan menambahkan model baru. Pada setiap iterasi, Gradient Boosting akan menambahkan _weak learner_ baru dan mengoreksi prediksi sebelumnya dengan memperhitungkan kesalahan pada prediksi tersebut. Dalam setiap iterasi, Gradient Boosting memperbarui _residual error_ dengan mengurangi hasil prediksi dari target, lalu menambahkan _weak learner_ baru yang menyelesaikan masalah _residual error_ yang dihasilkan.

Secara matematis, _Gradient Boosting_ mengoptimalkan suatu fungsi objektif dengan mengevaluasi _gradient_ pada setiap titik. Fungsi objektif yang umum digunakan dalam _Gradient Boosting_ adalah fungsi **Mean Squared Error (MSE)** untuk regresi dan fungsi Log-Loss untuk klasifikasi.

Berikut ini beberapa parameter yang digunakan pada model _Gradient Boosting_, antara lain:
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
* Memerlukan _tuning parameter_ yang cermat untuk mendapatkan model yang optimal. Proses _tuning_ bisa memakan waktu dan membutuhkan keahlian.
* Berisiko mengalami _overfitting_, di mana model terlalu fokus pada data latih dan tidak dapat digeneralisasikan dengan baik pada data baru.
* Lebih kompleks dibandingkan dengan algoritma machine learning lainnya, sehingga membutuhkan waktu dan sumber daya komputasi yang lebih besar.
* Bisa lebih lambat dibandingkan dengan algoritma machine learning lainnya, terutama pada data berukuran besar.

> Saya memilih **Gradient Boosting** sebagai model terbaik untuk memprediksi harga rumah karena memiliki nilai eror yang paling kecil, sehingga memberikan hasil yang paling mendekati dengan harga aslinya.


## Evaluation

Pada tahap ini, dilakukan evaluasi model yang sudah melewati proses _training_ sebelumnya. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Oleh karena itu, semua metrik mengukur seberapa kecil nilai eror tersebut.

Metrik yang digunakan dalam proyek ini adalah **MSE (Mean Squared Error)**. Berikut persamaannya: 

$$MSE = \frac{1}{N} \Sigma_{i=1}^N({y_i}- y\_pred_i)^2$$

Keterangan:
* N = jumlah dataset
* yi = nilai sebenarnya
* y_pred = nilai prediksi


### 1. Scalling Dataset with MinMaxScaller
Sebelum menghitung nilai _MSE_ dalam model, perlu dilakukan proses _scalling_ fitur numerik pada data uji. Sebelumnya, sudah dilakukan proses _scalling_ pada data latih untuk menghindari kebocoran data. Hal ini harus dilakukan agar skala antara data latih dan data uji sama dan agar bisa dilakukan tahap evaluasi model. Adapun outputnya sebagai berikut: 

Tabel 7. Scalling Fitur Numerik pada Data Uji Menggunakan Fungsi `MinMaxScaler()` 
|     | LB       | LT       | KT       | KM       | GRS      |
|-----|----------|----------|----------|----------|----------|
| 993 | 0.108475 | 0.111857 | 0.333333 | 0.333333 | 0.333333 |
| 217 | 0.193220 | 0.205817 | 0.333333 | 0.333333 | 0.666667 |
| 954 | 0.244068 | 0.170022 | 0.666667 | 0.666667 | 0.333333 |
| 336 | 0.413559 | 0.460850 | 0.333333 | 0.666667 | 1.000000 |
| 955 | 0.667797 | 0.572707 | 0.666667 | 1.000000 | 0.333333 |
| ... | ...      | ...      | ...      | ...      | ...      |
| 4   | 0.583051 | 0.673378 | 1.000000 | 1.000000 | 1.000000 |
| 226 | 0.413559 | 0.785235 | 1.000000 | 0.666667 | 0.000000 |
| 903 | 0.294915 | 0.214765 | 0.666667 | 0.666667 | 0.666667 |
| 998 | 0.328814 | 0.340045 | 0.666667 | 0.666667 | 0.333333 |
| 873 | 0.108475 | 0.080537 | 0.333333 | 0.666667 | 0.333333 |

67 rows × 5 columns


Dari output di atas, terlihat bahwa skala fitur sudah relatif sama. Sehingga dataset sudah bisa digunakan untuk tahap evaluasi model.


### 2. Evaluate Model with MSE
Setelah _scalling_ data uji selesai, langkah selanjutnya adalah melakukan evaluasi model dengan metrik _MSE_ yang menghitung jumlah selisih kuadrat rata-rata antara nilai sebenarnya dengan nilai prediksi. Ketika nilai _MSE (Mean Squared Error)_ dihitung untuk data latih dan data uji, nilai tersebut akan dibagi dengan **1e3** agar skala _MSE_ tidak terlalu besar. Adapun outputnya sebagai berikut: 

Tabel 8. Hasil Evaluasi dengan Metrik MSE
|                   | train       | test        |
|-------------------|-------------|-------------|
| SVR               | 2596.551455 | 4848.200286 |
| Random Forest     | 520.333336  | 4374.319259 |
| Gradient Boosting | 535.080088  | 3624.57515  |

Selanjutnya, membuat plot metrik mse dengan bar chart. Adapun outputnya sebagai berikut: 

![19-plot-metrik-mse](https://github.com/balle97/model-regresi/assets/128248022/da1df910-a0fb-48a0-8842-23c13ca500ff) 

Gambar 9. Membuat Plot Metrik MSE dengan Bar Chart


Dari output di atas, terlihat bahwa model **Random Forest** dan **Gradient Boosting** menghasilkan nilai eror yang paling kecil. Sedangkan model dengan algoritma **SVR** menghasilkan nilai eror yang paling besar.


### 3. Testing Model
Selanjutnya, melakukan proses uji model dengan memprediksi nilai asli(y_true). Adapun outputnya sebagai berikut: 

Tabel 9. Uji Model dengan Memprediksi Nilai Asli(y_true)
|     | y_true | prediksi_SVR | prediksi_Random Forest | prediksi_Gradient Boosting |
|-----|--------|--------------|------------------------|----------------------------|
| 993 | 3250.0 | 3237.4       | 3250.0                 | 3242.2                     |

Dari output di atas, terlihat bahwa prediksi dengan model **Random Forest** dan **Gradient Boosting** memberikan hasil yang paling mendekati dengan harga aslinya. 


> Berdasarkan hasil evaluasi dengan metrik MSE dan korelasi faktor-faktor penting dengan harga rumah, dapat dikatakan bahwa proyek ini berhasil mencapai goals yang telah ditetapkan, antara lain:
>* Identifikasi faktor penting yang mempengaruhi harga rumah, yaitu faktor **luas bangunan** dan **luas tanah**.
>* Pengembangan model machine learning yang dapat memprediksi harga rumah dengan akurasi yang cukup baik. Memilih model **Gradient Boosting** sebagai model terbaik berdasarkan nilai MSE yang terendah. Meskipun masih ada ruang untuk peningkatan, namun secara keseluruhan proyek ini memberikan kontribusi yang signifikan dalam memahami dan memprediksi harga rumah di kecamatan Tebet.


## Referensi
[[1](https://www.researchgate.net/publication/325656226_KOMPOSISI_HARGA_JUAL_RUMAH_TINGGAL_LAYAK_HUNI_DI_YOGYAKARTA_STUDI_KASUS_PEMBANGUNAN_RUMAH_TIPE_90115_DI_LUAR_KOMPLEKS_PERUMAHAN)] Musyafa, A. (2013). Komposisi Harga Jual Rumah Tinggal Layak Huni Di Yogyakarta (Studi Kasus Pembangunan Rumah Tipe 90/115 di Luar Kompleks Perumahan) (004K). Konferensi Nasional Teknik Sipil 7 (KoNTekS 7), 7-12.

[[2](https://jurnal.untan.ac.id/index.php/justin/article/view/18455)] Hendra, Tursina, & Nyoto, R. D. (2017). Case Base Reasoning Penentuan Harga Rumah dengan Menggunakan Metode Tversky (Studi Kasus : Kota Pontianak). Jurnal Sistem Dan Teknologi Informasi (JUSTIN), 5(2), 75–79.

[[3](https://jurnal.mdp.ac.id/index.php/jatisi/article/view/701)] Saiful, A., Andryana, S. & Gunaryati, A. (2021). Prediksi Harga Rumah Menggunakan Web Scrapping Dan Machine Learning Dengan Algoritma Linear Regression. Jurnal Teknik Informatika dan Sistem Informasi (JATISI), 8(1), 41-50. https://doi.org/10.35957/jatisi.v8i1.701.

[[4](https://ejournal.itn.ac.id/index.php/jati/article/view/6343)] Haryanto, C., Rahaningsih, N. & Basysyar, F. M. (2023). Komparasi Algoritma Machine Learning dalam Memprediksi Harga Rumah. JATI (Jurnal Mahasiswa Teknik Informatika), 7(1), 533-539. https://doi.org/10.36040/jati.v7i1.6343.

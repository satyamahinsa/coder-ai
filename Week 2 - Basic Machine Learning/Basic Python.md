# Basic Python Programming

Python adalah bahasa pemrograman yang populer karena kemudahan penggunaannya dan fleksibilitasnya. Banyak digunakan di berbagai bidang, seperti pengembangan web, data science, kecerdasan buatan, dan banyak lagi.

Berikut adalah beberapa konsep dasar Python

## 1. Input dan Output

- **Input:** Mengambil data dari pengguna.

  _Contoh:_

  ```python
  name = input("Enter your name: ")
  print(f"Hello, {name}")
  ```

- **Output:** Menampilkan data ke layar.

  _Contoh:_

  ```python
  print("Hello, World!")
  ```

## 2. Variabel dan Tipe Data

Variabel digunakan untuk menyimpan data. Python memiliki tipe data yang dinamis, artinya kita tidak perlu mendefinisikan tipe data secara eksplisit.

### 2.1 Tipe Data Dasar di Python

- **int:** Bilangan bulat, contoh: `x = 5`
- **float:** Bilangan desimal, contoh: `y = 3.14`
- **str:** String atau teks, contoh: `name = "Satya"`
- **bool:** Boolean, hanya memiliki dua nilai, yaitu `True` atau `False`

_Contoh:_

```python
x = 5 # int
y = 3.14 # float
name = "Satya" # str
is_student = True # bool
```
### 2.2 Operasi pada Variabel
Python mendukung berbagai operasi matematika dan string.

#### Operasi Matematika
```python
a = 10
b = 3
print(a + b)  # Penjumlahan
print(a - b)  # Pengurangan
print(a * b)  # Perkalian
print(a / b)  # Pembagian
```

#### Operasi String
```python
greeting = "Hello"
name = "Satya"
message = greeting + " " + name  # Penggabungan string
print(message)  # Output: Hello Satya
```


## 3. Struktur Kontrol

Struktur kontrol digunakan untuk mengontrol alur eksekusi program.

### 3.1 If-Else Statement

If-else digunakan untuk melakukan pengambilan keputusan berdasarkan kondisi.

_Contoh:_

```python
nilai = 100
if nilai >= 90:
    print("Excellent")
else:
    print("Try Again!")
```

### 3.2 Looping (Perulangan)

- **For Loop:** Digunakan untuk iterasi melalui koleksi elemen, seperti list atau string.

  _Contoh:_

  ```python
  for i in range(3):
      print(i)  # Output: 0, 1, 2
  ```

- **While Loop:** Perulangan berbasis kondisi. Selama kondisi benar, perulangan akan terus dijalankan.

  _Contoh:_

  ```python
  count = 0
  while count < 5:
      print(count)
      count += 1  # Output: 0, 1, 2, 3, 4
  ```

## 4. Fungsi

Fungsi adalah blok kode yang dirancang untuk melakukan tugas tertentu. Fungsi dapat menerima argumen dan mengembalikan nilai.

_Contoh:_

```python
def greeting(name):
    print(f"Hello, {name}!")

greeting("Satya")
```

## 5. List dan Dictionary

- **List:** Kumpulan elemen yang berurutan, dapat diakses menggunakan indeks.

  _Contoh:_

  ```python
  people = ["Budi", "Andi", "Eko"]
  print(fruits[0])  # Output: Budi
  print(fruits[1])  # Output: Andi
  ```
  List juga mendukung berbagai metode, seperti append() untuk menambah elemen.
  ```python
  people.append("Rani")
  print(people) # Output: ["Budi", "Andi", "Eko", "Rani"]
  ```

- **Dictionary:** Kumpulan pasangan key-value, di mana setiap key unik dan mengarah pada value tertentu.

  _Contoh:_

  ```python
  person = {"name": "Satya", "age": 20}
  print(person["name"])  # Output: Satya
  ```

## 6. Import dan Modul

Python memiliki ribuan modul bawaan dan eksternal yang bisa digunakan untuk berbagai keperluan. Modul dapat diimpor menggunakan kata kunci `import`.

```python
import math

print(math.sqrt(16))  # Output: 4.0
```

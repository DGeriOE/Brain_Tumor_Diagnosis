# 🧠 Agyi Tumor Diagnosztika MRI Képek Alapján (Brain Tumor Diagnosis)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classification-2F80ED?style=flat-square)](https://xgboost.readthedocs.io/)

Ez a projekt az **Óbudai Egyetem Gépivilágítás (GepiLatas)** tárgyának keretein belül készült. A cél egy olyan gépi tanulási csővezeték (pipeline) kidolgozása, amely képes hagyományos képfeldolgozási és gépi tanulási módszerek segítségével osztályozni agyi MRI felvételeket négy kategóriába:
1. **Glioma** (Agydaganat típus)
2. **Meningioma** (Agyhártyadaganat)
3. **No Tumor** (Egészséges agy / Nincs daganat)
4. **Pituitary** (Agyalapi mirigy daganat)

---

## 📂 Projekt Mappaszerkezet

```text
BrainTumorDiagnosis/
│
├── data/                            # Az adathalmazok könyvtára (git által mellőzve)
│   ├── agyikepek_3_osztaly/        # Opcionális 3 osztályos adathalmaz
│   └── agyikepek_4_osztaly/        # A projektben használt 4 osztályos adathalmaz
│       ├── Training/                # Tanító képek kategóriák szerint (5599 kép)
│       └── Testing/                 # Tesztelő képek kategóriák szerint (1311 kép)
│
├── environment.yml                  # Conda környezet definíciós fájl
├── requirements.txt                 # Pip függőségek listája
├── projekt.ipynb                    # A fő Jupyter Notebook a kóddal és elemzésekkel
├── .gitignore                       # Git mellőzési szabályok
└── README.md                        # Ez a dokumentáció
```

## 🔧 Telepítés és futtatás (Conda)

A projekt egy dedikált Conda-környezetet használ a függőségek kezelésére.

1. **Repozitórium klónozása:**
   ```bash
   git clone https://github.com/DGeriOE/Brain_Tumor_Diagnosis.git
   cd Brain_Tumor_Diagnosis
   ```

2. **Környezet létrehozása:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Környezet aktiválása:**
   ```bash
   conda activate brain_tumor_diagnosis
   ```

4. **Jupyter notebook indítása (és a [projekt.ipynb](projekt.ipynb) megnyitása):**
   ```bash
   jupyter notebook
   ```

> [!NOTE]
> **Alternatív telepítés (Pip):** Ha nem Condát használsz, közvetlenül a `requirements.txt` alapján is telepítheted a függőségeket egy Python 3.11 környezetben: `pip install -r requirements.txt`

---

## ⚙️ A feldolgozási pipeline

A projekt a képek beolvasásától a végső osztályozásig az alábbi lépéseket követi:

### 1. Kép Előfeldolgozás és Szegmentáció
A bemeneti szürkeárnyalatos MRI képeken az alábbi transzformációkat hajtjuk végre a zajok kiszűrésére és a lényeges részek kiemelésére:
* **Gauss-szűrés:** $5 \times 5$-ös maggal végzett simítás a nagyfrekvenciás zajok csökkentésére.
* **Otsu-féle automatikus küszöbölés:** A háttér és az agyszövet szétválasztása egy bináris maszk segítségével.
* **Háttér-maszkolás:** A maszk alkalmazásával eltávolítjuk a koponyán kívüli zajokat és nem releváns képpontokat.
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Lokális kontrasztjavítás a szegmentált régión (`clipLimit=2.0`, `tileGridSize=(8, 8)`), hogy a daganatos területek kontúrjai jobban kirajzolódjanak.
* **Átméretezés:** A képeket fix $128 \times 128$ képpontos méretre skálázzuk a fix méretű jellemzővektor eléréséhez.

### 2. Jellemzőkinyerés
* **HOG (Histogram of Oriented Gradients):** Irányított gradiens hisztogramok kinyerése (`orientations=9`, `pixels_per_cell=(16, 16)`, `cells_per_block=(2, 2)`). Ez hatékonyan leírja a formákat és a textúrák irányait.
* **Intenzitás statisztikák:** A HOG vektor végéhez hozzáfűzzük a szegmentált kép átlagos fényerejét és szórását is, ezzel megőrizve a globális intenzitásbeli különbségeket.

### 3. Párhuzamosított Adatbetöltés
A több ezer kép feldolgozása egy szálon lassú lenne, ezért a projekt a `joblib` könyvtárat használja a képek párhuzamos feldolgozására (`n_jobs=-1`, azaz az összes CPU magot használva). Ez drasztikusan lerövidíti az adatok betöltésének és előfeldolgozásának idejét.

---

## 📊 Modellek és Abációs Vizsgálat

A projekt során megvizsgáltuk a jellemzők standardizálásának (StandardScaler) hatását a különböző gépi tanulási modellek pontosságára. A modelleket a tesztelő halmazon (1311 kép) értékeltük ki az alábbi metrikák alapján: **Accuracy**, **Macro F1-Score** és **ROC-AUC (One-vs-Rest)**.

### Összehasonlító táblázat

| Modell | Előfeldolgozás | Accuracy | Macro F1-Score | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | Nem standardizált / Standardizált | **0.9359** | **0.9310** | **0.9912** |
| **Support Vector Machine (SVM)** | Standardizált | **0.9176** | **0.9122** | **0.9896** |
| **Random Forest** | Nem standardizált / Standardizált | **0.9123** | **0.9057** | **0.9841** |
| **CalibratedClassifierCV (LinearSVC)** | Nem standardizált | 0.9047 | 0.8976 | 0.9789 |
| **CalibratedClassifierCV (LinearSVC)** | Standardizált | 0.9008 | 0.8934 | 0.9752 |
| **Logistic Regression** | Standardizált | 0.8940 | 0.8862 | 0.9770 |
| **K-Nearest Neighbors (KNN)** | Standardizált | 0.8863 | 0.8790 | 0.9834 |
| **Logistic Regression** | Nem standardizált | 0.8780 | 0.8706 | 0.9770 |
| **K-Nearest Neighbors (KNN)** | Nem standardizált | 0.8307 | 0.8182 | 0.9695 |
| **Support Vector Machine (SVM)** | Nem standardizált | 0.6453 | 0.6393 | 0.8424 |

### Főbb megállapítások:
1. **Az előfeldolgozás fontossága:** Az **SVM** teljesítménye kritikus mértékben függ a standardizálástól. Standardizált adatokkal kiemelkedő, **91.76%**-os pontosságot ért el, míg anélkül mindössze **64.53%**-ot.
2. **Skálázás-invariáns modellek:** A döntési fákon alapuló modellek (mint az **XGBoost** és a **Random Forest**) teljesítményét nem befolyásolta a standardizálás, mindkét esetben azonos eredményt adtak.
3. **Legjobb Modell:** A legjobb eredményt az **XGBoost** nyújtotta **93.59%**-os pontossággal és **0.9912**-es ROC-AUC értékkel, szorosan követve az SVM-mel és a Random Foresttel.



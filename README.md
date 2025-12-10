# Recommendation System

_Built on [Coursera scraper data](https://github.com/cmd-HMN/Course_Scraper). Models: KNN, Baseline, NN (GPU only)_  

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?logo=pytorch) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn) ![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-lightblue?logo=numpy) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-teal?logo=pandas)

---

## 📖 Overview  

Recommendation System is a CLI tool built to train and save recommendation models using data scraped from Coursera.
It supports multiple models including **KNN**, **Baseline**, and **Neural Networks** (PyTorch).

**Features:**
- Train models easily via CLI.
- Watchdog mode to monitor data changes.
- GPU support for Neural Networks.

> [!NOTE]
> **Models:** Currently supports `knn`, `bsl` (baseline), and `nn` (neural network).

---

## ⚡ Installation    

**Clone via Git:**  

```bash
git clone https://github.com/cmd-HMN/Recomendation-System-Coursi.git
cd Recomendation-System-Coursi
```

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> `torch-gpu` is needed

---
## 🖥️ Running  

> [!TIP]
> I have stepup the .toml file u can try that (haven't tested that myself)

You can run the recommendation CLI in two ways: *via shell script* or *directly with Python*.  

### 1. Run using Shell Script 

```bash
# Make the script executable
chmod +x ./run.sh  

# Run the CLI with a model
./run.sh --model knn
```

###  2. Run using Python

```bash
# Run CLI
python cli.py --model="nn"
```

---

### ⚙️ Arguments  

The CLI accepts the following arguments:  

##### Model Related Arguments 
- `--model` (Required) → Name of the model to train. Options: `knn`, `bsl`, `nn`.
- `--params` → Dictionary of parameters for the model (default: `{}`).

###### WatchDog related Arguments
- `--watchit` → Enable watchdog to monitor data path changes (default: `False`).
- `--watch_path` → Path to watch for changes (default: User Data Path).

#### Example  

```bash
python cli.py --model="nn" --params={'nn_epochs': 50} --watchit=True
```

---

## 📜 License  

This project is licensed under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for more details.

---

## 🌟 Acknowledgements  

- Built with ❤️ using Python, PyTorch, Scikit-Learn, NumPy, and Pandas.
- Using data from **Coursera Scraper**.
- If you find a bug or issue, just hit me up ✌️

---

_Have a Nice Day_

<img src="https://img.shields.io/badge/OS-Arch%20Linux-blue?logo=arch-linux" alt="Arch Linux Badge" />

# **📌 Poetry & Jupyter Notebook Setup Guide**

## **1️⃣ Install Poetry**
```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc  # or `source ~/.zshrc` if using Zsh
```
Verify installation:
```bash
poetry --version
```

---

## **2️⃣ Configure Poetry to Use Local Virtual Environments**
```bash
poetry config virtualenvs.in-project true
```

---

## **3️⃣ Set Up Python & Create the Virtual Environment**
Ensure Python 3.10 or higher is installed:
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```
Create the environment:
```bash
poetry env use /usr/bin/python3.10
```
Verify:
```bash
poetry env info
```
Expected:
```
Path: /your/project/.venv
Python: /your/project/.venv/bin/python
```

---

## **4️⃣ Install Dependencies**
```bash
poetry install
```
To install additional packages:
```bash
poetry add <package_name>
```

---

## **5️⃣ Make the Project Editable**
```bash
poetry install --no-root
```

---

## **6️⃣ Activate the Environment**
Manually:
```bash
source .venv/bin/activate
```
Deactivate:
```bash
deactivate
```
Or use:
```bash
poetry run <command>
```
Example:
```bash
poetry run python
```

---

## **7️⃣ Set Up Jupyter Notebook**
Install Jupyter and the kernel:
```bash
poetry add jupyter ipykernel
poetry run python -m ipykernel install --user --name=marovipipelines --display-name "Python (MaroviPipelines)"
```
Verify kernel installation:
```bash
jupyter kernelspec list
```
Expected:
```
Available kernels:
  marovipipelines    /home/youruser/.local/share/jupyter/kernels/marovipipelines
```

Run Jupyter:
```bash
poetry run jupyter notebook
```
or set `marovipipelines` as default:
```bash
jupyter notebook --NotebookApp.kernel_name="marovipipelines"
```

---

## **8️⃣ Final Checks**
✔ **Ensure system Python is used**:
```bash
which python3
```
✔ **Ensure Poetry uses `.venv/`**:
```bash
poetry env info
```
✔ **Run Python inside Poetry**:
```bash
poetry run python
```
✔ **Start Jupyter Notebook**:
```bash
poetry run jupyter notebook
``` 
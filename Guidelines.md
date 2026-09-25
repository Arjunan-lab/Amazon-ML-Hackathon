#  Team Environment Guidelines: Poetry

This project uses **Poetry** to manage Python packages and environments. This ensures everyone runs the exact same code setup without requiring heavy system resources or slow background virtual machines.

---

##  Core Team Rules

1. **Required Python Version**: This project strictly requires **Python 3.11**. Ensure you have it installed locally before building the environment.
2. **Never use `pip install`**: Always use `poetry add <package>` instead. This keeps our environment tracking file updated.
3. **Never commit environment folders**: The actual virtual environment (`.venv` or hidden folders) must never be pushed to Git.
4. **Always commit lockfiles**: Always push `pyproject.toml` and `poetry.lock` to Git when you add or update a package.

---

##  Step-by-Step Onboarding Setup

Follow these steps to set up your local environment.

### 1. Ensure Python 3.11 is Installed
Make sure you have Python 3.11 installed on your system. 
* You can download it from the official Python website or use a version manager like `pyenv` or `asdf`.
* Verify it by running: `python3 --version` (or `python --version` on Windows).

### 2. Install Poetry
Run the appropriate command in your terminal to install Poetry on your machine:

* **Mac / Linux / Windows (WSL):**
  ```bash
  curl -sSL https://python-poetry.org | python3 -
  ```
* **Windows (PowerShell):**
  ```powershell
  (Invoke-WebRequest -string "https://python-poetry.org" -UseBasicParsing).Content | python -
  ```

>  **Important:** Restart your terminal after installation so the `poetry` command becomes available.

### 3. Clone the Repository
Navigate to your preferred directory and clone the project:
```bash
git clone <your-repo-url>
cd <project-folder-name>
```

### 4. Configure & Build the Environment
Tell Poetry to use your local Python 3.11 installation, point its environments inside the project folder, and install the packages:
```bash
poetry env use python3.11
poetry config virtualenvs.in-project true
poetry install
```
Poetry will read the `poetry.lock` file and download the exact package versions required natively.

---

## 🔄 Daily Development Workflow

### How to Run Your Code
Because Poetry keeps the environment isolated, you must execute your scripts through Poetry:
```bash
poetry run python main.py
```
*Alternatively, you can type `poetry shell` to enter the environment completely, run your commands normally, and type `exit` when done.*

### How to Add a New Package
If your task requires a new library (e.g., `requests`), add it using:
```bash
poetry add requests
```
Once verified, commit and push both the `pyproject.toml` and `poetry.lock` files so the rest of the team gets the update.

### How to Sync Your Environment
Whenever you run `git pull` and notice that a teammate updated the `poetry.lock` file, update your local setup by running:
```bash
poetry install
```

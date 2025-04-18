
# Genre Classification System

This is a **Django-based web application** that uses machine learning for classifying book descriptions into various genres, such as **Sci-Fi**, **Romance**, **Mystery**, and **Fantasy**. The project leverages **TensorFlow** for model training and inference, with a backend powered by **Django**.

---

## Features
- Classifies book descriptions into genres.
- Built with **Django** for the backend.
- **TensorFlow** model using **LSTM (Long Short-Term Memory)** architecture.
- **TextVectorization** layer for text preprocessing.
- Support for easy deployment with **Conda** environment for managing dependencies.

---

## Installation

### Clone the repository
To get started, clone this repository to your local machine:

```bash
git clone https://github.com/ayukndip40/BookClassifier.git
cd <your-repo-name>
```

### Set up the Conda Environment
This project uses **Conda** to manage dependencies. To set up the required environment, follow these steps:

1. **Install Conda**:
   - Download and install **Miniconda** or **Anaconda**.
   - [Miniconda Download](https://docs.conda.io/en/latest/miniconda.html)
   - [Anaconda Download](https://www.anaconda.com/products/distribution)

2. **Create the Conda Environment**:
   Use the provided `environment.yml` file to create the environment:

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the Conda Environment**:
   Once the environment is created, activate it:

   ```bash
   conda activate genre-django
   ```

### Install Dependencies (Using Pip)
If you prefer to use `pip` instead of Conda, you can install the dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Setup

### Database Setup
Run the following command to set up the database:

```bash
python manage.py migrate
```

### Running the Development Server
Start the Django development server:

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your web browser to view the application.

---

## How It Works

1. **Text Preprocessing**:
   - The **TextVectorization** layer preprocesses the book descriptions into a sequence that can be fed into the model.
   
2. **Model**:
   - The machine learning model is based on **LSTM (Long Short-Term Memory)** architecture and trained on a set of book descriptions and their corresponding genres.
   
3. **Prediction**:
   - When a user enters a description, the system uses the trained model to predict the genre of the book, returning the most likely genre and its confidence score.

---

## Files in the Project

- **`genre_project/`**: Main Django project files.
- **`classifier/`**: The Django app that handles genre classification.
- **`book_genre_model.h5`**: Trained TensorFlow model for genre classification.
- **`text_vectorizer/`**: Saved TensorFlow model for text preprocessing.
- **`label_encoder.pkl`**: Pickled label encoder to map predictions to genre labels.
- **`environment.yml`**: Conda environment file to recreate the environment.
- **`requirements.txt`**: List of Python dependencies (for pip-based environments).

---

## Contributing

Feel free to fork this project and submit pull requests. If you encounter any issues or bugs, please open an issue on GitHub, and I will address it as soon as possible.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Django**: For building the web framework.
- **TensorFlow**: For the machine learning framework used in the project.
- **Conda**: For environment management.
- **LabelEncoder**: For encoding labels in the genre classification task.

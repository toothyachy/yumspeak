# YUMSPEAK - Speak it, Eat it!
Our product is your ultimate dining wingman, delivering instant, spot-on recommendations for restaurants and eateries that align with your unique tastes and adventurous spirit when you are in Singapore.

## Description
### Dataset Acquisition
Dataset was obtained by web scraping for restaurants in Singapore.
Restaurant names were scraped using an extension by cuisine types and put into the reviews scraper to get the full datset of reviews.

### Data Cleaning
Dataset needed extensive cleaning to remove irrelevant business that were non-dining.

### Feature Engineering
Geographical information (e.g: latitude and longitude) were extracted from the link.

### Modelling
- Word2Vec was used to train the model to predict the relevant restaurants.
- K-Nearest Neighbours (KNN) was used to locate the 5 nearest restaurants based on the location.

### Challenges
- Cleaning of the scraped dataset was time consuming.
- The BERT model has a better prediction but more computational power was required.

## Installation instructions
### 1. Clone the Repository
Start by cloning the repository to your local machine. Open a terminal and run:
```bash
git clone https://github.com/toothyachy/yumspeak.git
cd yumspeak
```

### 2. Setting Up a Virtual Environment with pyenv
It's recommended to manage Python versions and virtual environments using pyenv. This ensures that project dependencies do not interfere with system-wide Python packages. If you haven't already, install pyenv by following the instructions on pyenv's GitHub repository.

After installing pyenv, follow these steps to set up a virtual environment for the project:
##### 1. Install Python 3.10.6 using pyenv (skip this step if you already have this version installed):
```bash
pyenv install 3.10.6
```
##### 2. Create a virtual environment named yumspeak (or another name of your choice) using Python 3.10.6:
```bash
pyenv virtualenv 3.10.6 yumspeak
```
##### 3. Activate the virtual environment. Navigate to your project's directory, then set the local Python version to your newly created virtual environment:
```bash
cd path/to/yumspeak
pyenv local yumspeak
```
This step will create a .python-version file in your project directory, automatically activating the yumspeak virtual environment whenever you navigate to this directory.
##### 4. Verify that the virtual environment is activated by checking the Python version:
```bash
python --version
```
This command should output Python 3.10.6, indicating that the correct version of Python is being used.

### 3. Install Dependencies
Install all the dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```
Ensure you have a requirements.txt file in your repository with all the necessary libraries, including Pandas, WordCloud, Matplotlib, Seaborn, Plotly, Scikit-learn, NLTK, and Streamlit.

### 4. Set Up the Streamlit Application
To run the Streamlit application, navigate to the directory containing your Streamlit script (e.g., app.py) and execute:
```bash
streamlit run app.py
```

## Credits
Karen Ann Leong
Jason Chia Huat Soon
Lee Yao Guang
Ronald Lin Ziwei
Song Cai Yin Tricia

Special thanks to Andrii Gegliuk for the guidance throughout this project.

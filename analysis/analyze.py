# load required modules
from joblib import load
import pandas as pd
import fasttext
# from googletrans import Translator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
import nltk
from nltk.corpus import stopwords
import os


# Initialize the lemmatizer and stopwords
all_names = None
lemmatizer = None
stop_words = None

# Set up the directory where the models are stored
cur_dir = os.getcwd() + "/analysis"

# Define global variables to hold the loaded models
spam_model, spam_vector = None, None
viol_model, viol_vector = None, None
pos_model, pos_vector = None, None
lang_model = None


class ModelsNotLoadedError(Exception):
    pass


def safe_nltk_download(resource):
    try:
        nltk.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])

def load_models():
    """
    Function to load models once and reuse them
    """
    global spam_model, spam_vector, viol_model, viol_vector, pos_model, pos_vector, lang_model, all_names, lemmatizer, stop_words

    # Load Spam model and corresponding vectorizer
    spam_model = load(fr"{cur_dir}/model/lr_spam.pkl")
    spam_vector = load(fr"{cur_dir}/model/vectorizer_spam.pkl")

    # Load violence model and corresponding vectorizer
    viol_model = load(fr"{cur_dir}/model/lr_violence.pkl")
    viol_vector = load(fr"{cur_dir}/model/vectorizer_violence.pkl")

    # Load positivity model and corresponding vectorizer
    pos_model = load(fr"{cur_dir}/model/svm.pkl")
    pos_vector = load(fr"{cur_dir}/model/vectorizer.pkl")

    # Load language detection model
    lang_model = fasttext.load_model(fr'{cur_dir}/model/lid.176.ftz')

    safe_nltk_download("corpora/names")
    safe_nltk_download("corpora/wordnet")
    safe_nltk_download("corpora/stopwords")

    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')


# load spacy model and pytextrank to spacy pipe
# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("textrank")

# initialize google translator object
# translator = Translator()

def cleaned_text(docs):
    """
    Function to remove stopwords from text
    :param docs: string containing stopwords
    :return: string without stopwords
    """
    docs_cleaned = list()
    for doc in docs:
        doc = str(doc)
        doc = doc.lower()
        doc_cleaned = ' '.join(
            lemmatizer.lemmatize(word) for word in doc.split() if word not in all_names and word not in stop_words)
        docs_cleaned.append(doc_cleaned)
    return docs_cleaned


# def extract_keywords(text, n = 5):
#     """
#     Function to extract keywords from a sentence
#     :param text: String from which keywords are extracted
#     :param n: no. of keywords returned
#     :return: list of n most frequent keywords
#     """
#     return nlp(text)[:5]


def pred_pos(df: pd.DataFrame, cleaned_data):
    """
    Predict results of positivity model
    :param df: DataFrame containing comments field
    :return: df with positivity field as result of prediction
    """
    data_transformed = pos_vector.transform(cleaned_data)
    df["positivity"] = pos_model.predict(data_transformed)
    return df


def pred_spam(df: pd.DataFrame, cleaned_data):
    """
    Predict results of Spam model
    :param df: DataFrame containing comments field
    :return: df with spam field as result of prediction
    """
    data_transformed = spam_vector.transform(cleaned_data)
    df["spam"] = spam_model.predict(data_transformed)

    return df


def pred_viol(df: pd.DataFrame, cleaned_data):
    """
    Predict results of violence model
    :param df: DataFrame containing comments field
    :return: df with violence field as result of prediction
    """
    data_transformed = viol_vector.transform(cleaned_data)
    df["violence"] = viol_model.predict(data_transformed)

    return df


def func_lang(text):
    result = lang_model.predict(text, k=1)
    if result[1][0] > 0.8:
        return result[0][0][-2:]
    return "en"


def lang_detect(df: pd.DataFrame):
    """
    Detect language in comments
    :param df: DataFrame containing comments field
    :return: df with language field for corresponding comments
    """
    df["language"] = df.comments.apply(func_lang)

    return df


# def translate_text(text, dest):
#     text = text.rstrip()
#     text = text.lstrip()
#     try:
#         print(text)
#         return translator.translate(text, dest=dest).text
#     except IndexError or TypeError:
#         return "NA"


# def translate_comments(df: pd.DataFrame, target="en"):
#     """
#     Translate comments into targeted language
#     :param df: DataFrame containing comments field
#     :param target: Language of translated comments
#     :return: df with translated field for corresponding comments
#     """
#     df['translated_Text'] = df['comments'].apply(lambda x: translate_text(x, dest=target))
#     return df


# def apply_keywords(df: pd.DataFrame):
#     """
#     Function for keyword extraction of text
#     :param df: DataFrame containing comments field
#     :return: df with keywords field corresponding to each comment
#     """
#     df["Keywords"] = df.comments.apply(extract_keywords)
#     return df


def compute(df: pd.DataFrame, video_id) -> pd.DataFrame:
    if spam_model and spam_vector and pos_model and pos_vector and viol_model and viol_vector and lang_model == None:
        raise ModelsNotLoadedError({"Message": "Machine Learning Models are not loaded"})
    df: pd.DataFrame = df
    print(f"{video_id}: Starting Preprocessing")
    cleaned_data = cleaned_text(df.comments)
    df = pred_pos(df, cleaned_data)
    print(f"{video_id}: Positivity Prediction done")
    df = pred_spam(df, cleaned_data)
    print(f"{video_id}: Spam Prediction done")
    df = pred_viol(df, cleaned_data)
    print(f"{video_id}: Violence Prediction done")
    df = lang_detect(df)
    # df_en = translate_comments(df[df.language != "en"][:150])   # sending a subset of dataset due to api limits
    return df


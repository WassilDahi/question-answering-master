import eel
from similarite import calcule_cos,calcule_eucl
import numpy as np
from Attention import qa
from allennlp.predictors.predictor import Predictor
from keras_question_and_answering_system.library.seq2seq import Seq2SeqQA
from keras_question_and_answering_system.library.utility.squad import SquADDataSet
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BertTokenizer, BertModel
from bert_qa import bert_reponse,bert_layer
import glob
import os

predictor = Predictor.from_path("bidaf.tar.gz")
start_end_attention = [0,0]
seq2seq = Seq2SeqQA()
data_set = SquADDataSet(data_path=None)
qa_list = list()
seq2seq.load_model(model_dir_path='demo/models')

tokenizer_bert = AutoTokenizer.from_pretrained("csarron/bert-base-uncased-squad-v1") 
model_bert = AutoModelForQuestionAnswering.from_pretrained("csarron/bert-base-uncased-squad-v1")
model_version = 'bert-base-uncased'
do_lower_case = True
model_bert2 = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer_bert2 = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)


@eel.expose
def testt(text,question,selected):
    if selected == 'Distance cosinus':
        sentences,cos = calcule_cos(text,question)
    else:
        sentences,cos = calcule_eucl(text,question)
    liste = dict()
    for i in range(len(sentences)):
        liste[sentences[i]] = cos[i]
    answer = sentences[np.argmin(cos)]
    return liste

@eel.expose
def attention_lstm(text,question):
    global start_end_attention
    text,start_end_attention = qa(text,question)
    return text
@eel.expose
def seq2seq_lstm(text,question):
    reponse = seq2seq.reply(text,question)
    return reponse

@eel.expose
def bert_resp(text,question):
    global start_end_attention
    resp ,start_end_attention = bert_reponse(text,question)
    return resp

@eel.expose
def liste_image():
    resp = os.listdir("C:\\Users\\acer\\Desktop\\pfe\\pfe\\templates\\attention")
    reponse =[]
    for image in resp:
        new_image = "attention/"+image
        reponse.append(new_image)
    return reponse
@eel.expose
def reponse(doctionnaire):
    sentences = []
    cos = []
    for dd in doctionnaire.keys():
       sentences.append(dd)
       cos.append(doctionnaire[dd])
    answer = sentences[np.argmin(cos)]
    return answer

@eel.expose
def affiche_layer(text,question,nombre):
    resp = bert_layer(text,question,nombre)
    return resp

@eel.expose
def liste_image_bert():
    resp = os.listdir("C:\\Users\\acer\\Desktop\\pfe\\pfe\\templates\\bert")
    reponse =[]
    for image in resp:
        new_image = "bert/"+image
        reponse.append(new_image)
    return reponse
@eel.expose
def start_attention():
    global start_end_attention
    return start_end_attention[0]
@eel.expose
def end_attention():
    global start_end_attention
    return start_end_attention[1]
eel.init('templates')

eel.start("home.html",size=(1024,900))
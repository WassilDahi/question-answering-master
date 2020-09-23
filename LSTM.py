from keras_question_and_answering_system.library.seq2seq import Seq2SeqQA
from keras_question_and_answering_system.library.utility.squad import SquADDataSet
from allennlp.predictors.predictor import Predictor
import allennlp_models.rc


def lstm(question_context,question):
    ans = seq2seq.reply(question_context, question)
    return ans


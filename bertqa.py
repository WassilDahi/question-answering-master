import torch
import sys
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from IPython.core.display import display, HTML, Javascript


from transformers import BertTokenizer, BertModel
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))

tokenizer_bert = AutoTokenizer.from_pretrained("csarron/bert-base-uncased-squad-v1") 
model_bert = AutoModelForQuestionAnswering.from_pretrained("csarron/bert-base-uncased-squad-v1")
model_version = 'bert-base-uncased'
do_lower_case = True
model_bert2 = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer_bert2 = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

def bert_reponse(context,question,nombre):
    ino = glob.glob("C:\\Users\\abdeslem\\application\\templates\\bert\\*")
    for fichier in ino:
        os.remove(fichier)
    inputs = tokenizer_bert.encode_plus(question, context, return_tensors="pt") 
    answer_start_scores, answer_end_scores = model_bert(**inputs)
    answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1 
    io = answer_start.item()
    print(io,answer_end)
    sentence_a = question
    sentence_b = context
    inputs = tokenizer_bert2.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    attention = model_bert2(input_ids, token_type_ids=token_type_ids)[-1]
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer_bert2.convert_ids_to_tokens(input_id_list)
    result=tokenizer_bert2.convert_tokens_to_string(tokenizer_bert2.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    call_html()
    attention=attention[0].detach().numpy()
    attention=attention[0]
    i=1
    tito =int(len(tokens)/10)
    a= attention[nombre]
    
    io=0
    for io in range(tito):
        matrice = []

        for u in range(io*10,(io+1)*10):
            ligne = []
            for y in range(io*10,(io+1)*10):
                ligne.append(a[u][y])
            matrice.append(ligne)
        p = np.array(matrice,float)
        plt.figure()
        fig, ax = plt.subplots()
        im, cbar = heatmap(p, tokens[io*10:(io+1)*10], tokens[io*10:(io+1)*10], ax=ax,cmap="YlGn", cbarlabel="Matrice d'attention Texte-Question")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")
        fig.tight_layout()
        del p
        plt.savefig('templates/bert/attention_bert'+str(nombre)+str(io)+'.png')
        plt.close()
        
    if len(tokens) > tito*10:
        matrice = []

        for u in range(tito*10,len(tokens)):
            ligne = []
            for y in range(tito*10,len(tokens)):
                ligne.append(a[u][y])
            matrice.append(ligne)
        p = np.array(matrice,float)
        plt.figure()
        fig, ax = plt.subplots()
        im, cbar = heatmap(p, tokens[tito*10:len(tokens)], tokens[tito*10:len(tokens)], ax=ax,cmap="YlGn", cbarlabel="Matrice d'attention Texte-Question")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")
        fig.tight_layout()
        del p
        io+=1
        plt.savefig('templates/bert/attention_bert'+str(nombre)+str(io)+'.png')
        plt.close()

    return result

bert_reponse("Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary.Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legendVenite Ad Me Omnes. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the maindrive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.","What is in front of the Notre Dame Main Building?",11)
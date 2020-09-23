var app = new Vue({
    el: '#app',
    data: {
      text: '' ,
      question : '',
      reponse: '',
      selected:'similarité cosinus',
      exemplechange:'',
      tab :1,
      time:'',
    },

    methods:{
        sim:async function (event) {
          var firstTimestamp = new Date().getTime(); 
          var n = await eel.testt(this.text,this.question,this.selected)();
          var secondTimestamp = new Date().getTime(), 
          this.time = secondTimestamp - firstTimestamp; 
            this.reponse = n;
        },
        select_tab: function(selected) {
          this.tab = selected;
          this.text='';
          this.question ='';
          this.reponse = '';
        },
        attention:async function(event) {
          var firstTimestamp = new Date().getTime(); 
          var m = await eel.attention_lstm(this.text,this.question)();
          var secondTimestamp = new Date().getTime(), 
          this.time = secondTimestamp - firstTimestamp;
          this.reponse = m;
        },
        lstm:async function() {
          var firstTimestamp = new Date().getTime(); 
          var r = await eel.seq2seq_lstm(this.text,this.question)();
          var secondTimestamp = new Date().getTime(), 
          this.time = secondTimestamp - firstTimestamp;
          this.reponse = r;
        },
        transfert:async function() {
          var firstTimestamp = new Date().getTime(); 
          var rm = await eel.bert_resp(this.text,this.question)();
          var secondTimestamp = new Date().getTime(), 
          this.time = secondTimestamp - firstTimestamp;
          this.reponse = rm;
        },
        test: function() {
          vm = this;
          if(vm.selected =='similarité cosinus'){
            alert("elle fonctionne");
          }else{
            alert("un autre essaie")
          }
          
        },
    },
    watch: {
      exemplechange(newValue,oldValue){
                if(newValue == 1){
                  this.text = "Architecturally, the school has a Catholic character. "+
                  "Atop the Main Building's gold dome is a golden statue of the Virgin Mary."+
                  "Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend"+ 
                  "Venite Ad Me Omnes. Next to the Main Building is the Basilica of the Sacred Heart. "+
                  "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection."+
                  "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette "+
                  "Soubirous in 1858. At the end of the main"+ 
                  "drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.";
                  this.question = "What is in front of the Notre Dame Main Building?";

                  this.text = this.text.replace("a copper statue of christ","<h1> a copper statue of christ </h1>");
                }else{
                  this.text="un text";
                  this.question = "une question";
                }
            }},

  })
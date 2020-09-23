var app = new Vue({
    el: '#app',
    data: {
      text: '' ,
      question : '',
      reponse: '',
      selected:'Distance cosinus',
      tab :1,
      exemplechange:'Choisissez un exemple',
      reponse_popup:'',
      popup:0,
      time:'',
      patienter:0,
      image: [],
      sentences : [],
      layer:1,
      affiche:false,
      answer_start:0,
      answer_end:0,
      print_layer:0,
    },

    methods:{
        sim:async function (event) {
          var tito = this.text;
          var qiqo = this.question;
          if(tito.trim() == ""){
            alert("Vous avez laisser le champs de texte vide !");
          }else if(qiqo.trim() == ""){
            alert("Vous avez laisser le champs de question vide !");
          }else{
          this.patienter =1;
          this.popup =0;
          this.reponse='';
          var firstTimestamp = new Date().getTime(); 
          var op = await eel.testt(this.text,this.question,this.selected)();
          var n = await eel.reponse(op)();
          this.sentences = op;
          var secondTimestamp = new Date().getTime();
          this.time = secondTimestamp - firstTimestamp;
            this.reponse = n ;
            this.reponse_popup = this.text.replace(n,"<span style=\"color:red\">"+n+"</span>");
            this.popup =1;
            this.patienter =0;
          }
        },
        select_tab: function(selected) {
          this.tab = selected;
          this.text='';
          this.question ='';
          this.reponse = '';
          this.exemplechange='Choisissez un exemple';
          this.reponse='';
          this.reponse_popup='';
          this.popup =0;
          this.patienter =0;
        },
        attention:async function(event) {
          this.image= [];
          var tito = this.text;
          var qiqo = this.question;
          if(tito.trim() == ""){
            alert("Vous avez laisser le champs de texte vide !");
          }else if(qiqo.trim() == ""){
            alert("Vous avez laisser le champs de question vide !");
          }else{
          this.patienter =1;
          this.popup =0;
          this.reponse='';
          var firstTimestamp = new Date().getTime(); 
          var m = await eel.attention_lstm(this.text,this.question)();
          this.answer_start = await eel.start_attention()();
          this.answer_end = await eel.end_attention()();
          var secondTimestamp = new Date().getTime();
          this.time = secondTimestamp - firstTimestamp;
          this.reponse = m;
          this.reponse_popup = this.text.toLowerCase(); 
          this.reponse_popup = this.reponse_popup.replace(m,"<span style=\"color:red\">"+m+"</span>");
          this.popup =1;
          this.patienter =0;
          this.image = await eel.liste_image()();
          }
        },
        lstm:async function() {
          var tito = this.text;
          var qiqo = this.question;
          if(tito.trim() == ""){
            alert("Vous avez laisser le champs de texte vide !");
          }else if(qiqo.trim() == ""){
            alert("Vous avez laisser le champs de question vide !");
          }else{
          this.patienter =1;
          this.popup =0;
          this.reponse='';
          var firstTimestamp = new Date().getTime(); 
          var r = await eel.seq2seq_lstm(this.text,this.question)();
          var secondTimestamp = new Date().getTime();
          this.time = secondTimestamp - firstTimestamp;
          this.reponse = r;
          this.text = this.text.toLowerCase();
          this.reponse_popup = this.text.replace(r,"<span style=\"color:red\">"+r+"</span>");
          this.popup =1;
          this.patienter =0;
        }
        },
        transfert:async function() {
          this.print_layer =0;
          var tito = this.text;
          var qiqo = this.question;
          if(tito.trim() == ""){
            alert("Vous avez laisser le champs de texte vide !");
          }else if(qiqo.trim() == ""){
            alert("Vous avez laisser le champs de question vide !");
          }else{
          this.affiche = false;
          this.patienter =1;
          this.popup =0;
          this.reponse='';
          var firstTimestamp = new Date().getTime(); 
          var rm = await eel.bert_resp(this.text,this.question)();
          var secondTimestamp = new Date().getTime();
          this.answer_start = await eel.start_attention()();
          this.answer_end = await eel.end_attention()();
          this.time = secondTimestamp - firstTimestamp;
          this.reponse = rm;
          var reg = "<span style=\"color:red\">" + rm + "</span>";
          var text_min = this.text.toLowerCase();
          this.reponse_popup = text_min.replace(rm,reg);
          this.popup =1;
          this.patienter =0;}
        },
        layer_afficher:async function() {
          this.image= [];
          this.print_layer =1;
          this.affiche = false;
          var justee = await eel.affiche_layer(this.text,this.question,parseInt(this.layer))();
          this.image = await eel.liste_image_bert()();
          this.affiche = true;
          this.print_layer =0;
         
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
                }if(newValue ==2){
                  this.text="Philosophers in antiquity used the concept of force in the study of stationary and moving objects and simple machines, but thinkers such as Aristotle and Archimedes retained fundamental errors in understanding force. In part this was due to an incomplete understanding of the sometimes non-obvious force of friction, and a consequently inadequate view of the nature of natural motion. A fundamental error was the belief that a force is required to maintain motion, even at a constant velocity. Most of the previous misunderstandings about motion and force were eventually corrected by Galileo Galilei and Sir Isaac Newton. With his mathematical insight, Sir Isaac Newton formulated laws of motion that were not improved-on for nearly three hundred years. By the early 20th century, Einstein developed a theory of relativity that correctly predicted the action of forces on objects with increasing momenta near the speed of light, and also provided insight into the forces produced by gravitation and inertia.";
                  this.question = "How long did it take to improve on Sir Isaac Newton's laws of motion?";
                }
                if(newValue ==3){
                  this.text=" The French and Indian War (1754–1763) was the North American theater of the worldwide Seven Years' War. The war was fought between the colonies of British America and New France, with both sides supported by military units from their parent countries of Great Britain and France, as well as Native American allies. At the start of the war, the French North American colonies had a population of roughly 60,000 European settlers, compared with 2 million in the British North American colonies. The outnumbered French particularly depended on the Indians. Long in conflict, the metropole nations declared war on each other in 1756, escalating the war from a regional affair into an intercontinental conflict.";
                  this.question = " How many people were in British North American Colonies?";
                }
                if(newValue ==4){
                  this.text=" Imperialism is a type of advocacy of empire. Its name originated from the Latin word \"imperium\", which means to rule over large territories. Imperialism is \"a policy of extending a country's power and influence through colonization, use of military force, or other means\". Imperialism has greatly shaped the contemporary world. It has also allowed for the rapid spread of technologies and ideas. The term imperialism has been applied to Western (and Japanese) political and economic dominance especially in Asia and Africa in the 19th and 20th centuries. Its precise meaning continues to be debated by scholars. Some writers, such as Edward Said, use the term more broadly to describe any system of domination and subordination organised with an imperial center and a periphery.";
                  this.question = "Imperialism is responsible for the rapid spread of what?";
                }
                if(newValue ==5){
                  this.text="A prime number (or a prime) is a natural number greater than 1 that has no positive divisors other than 1 and itself. A natural number greater than 1 that is not a prime number is called a composite number. For example, 5 is prime because 1 and 5 are its only positive integer factors, whereas 6 is composite because it has the divisors 2 and 3 in addition to 1 and 6. The fundamental theorem of arithmetic establishes the central role of primes in number theory: any integer greater than 1 can be expressed as a product of primes that is unique up to ordering. The uniqueness in this theorem requires excluding 1 as a prime because one can include arbitrarily many instances of 1 in any factorization, e.g., 3, 1 · 3, 1 · 1 · 3, etc. are all valid factorizations of 3.";
                  this.question = "Any number larger than 1 can be represented as a product of what?";
                }

            }},

  })
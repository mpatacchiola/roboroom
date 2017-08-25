
import tensorflow as tf
import shutil #libreria per rimuove la directory di log

#Creo un dataset con i valori di input
#ed i relativi valori di output ottimali (target)
dataset_input = [[0, 0],[0, 1],[1, 0],[1, 1]]
dataset_target = [[0.5],[0],[1],[0.5]]

#Dichiaro due tensori vuoti (placeholder) che saranno utilizzati come
#contenitori per i nostri valori di input ed i nostri target ottimali
#Per ricordare che si tratta di placeholder uso il prefisso _ prima
#del nome della variabile.
_input = tf.placeholder(tf.float32, shape=[4,2], name = 'input')
_target = tf.placeholder(tf.float32, shape=[4,1], name = 'target')

#Dichiaro le variabili relative alle connessioni W1 e W2
#Posso mettere i due valori di W1 e W2 dentro una sola variabile W
#rappresentata da un vettore con 2 righe ed 1 colonna
W = tf.Variable(tf.truncated_normal([2,1], mean=0.0, stddev=0.1), name = "W")
#Dichiaro la variabile di Bias B
B = tf.Variable(tf.zeros([1]), name = "B")

#Creo il perceptron. Utilizzo la sigmoide come funzione
#di attivazione. l'input viene moltiplicato per le
#connessioni W, in seguito il valore del Bias B viene 
#aggiunto al risultato ed il tutto viene passato alla
#funzione di attivazione. 
with tf.name_scope("perceptron") as scope:
	output = tf.sigmoid(tf.matmul(_input, W) + B)

#Questa parte del codice definisce il tipo di costo da
#ridurre durante l'addestramento. Viene calcolata la 
#differenza tra il target e l'output della rete (al quadrato).
#Il risultato e' un valore numerico che costituisce il metro
#di giudizio per capire se la rete sta apprendendo o meno.
#Se il training sta funzionando il valore del costo scende. 
with tf.name_scope("cost") as scope:
	cost = tf.reduce_mean(tf.nn.l2_loss(_target - output))
        tf.scalar_summary("cost", cost)

#Definisco il tipo di optimizer ed il relativo learning rate
#In questo caso utilizzo il classico Gradient Descent optimizer
#Il valore del learning rate viene generalmente trovato per prove 
#successive. Nel nostro caso un valore di 0.01 fa il suo dovere.
with tf.name_scope("optimizer") as scope:
	optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

#Inizializza tutte le variabili di tipo tf
#che abbiamo dichiarato sopra.
init = tf.initialize_all_variables()

#Dichiara un oggetto di tipo "Session" che ci
#servira' per gestire la nostra sessione di lavoro.
sess = tf.Session()

#Il writer serve per pubblicare le informazioni relative
#alla nostra rete in Tensorboard. Per visualizzare queste 
#informazioni dal vostro browser dovete aprire un terminale 
#e lanciare il comando: tensorboard --logdir=/tmp/tf_log
#quindi aprire il vostro browser inserendo nella barra
#il seguente indirizzo: http://0.0.0.0:6006
#Potrebb essere necessario attendere alcuni secondi prima di
#cominciare a vedere qualcosa in EVENTS, il grafo della rete
#e' invece immediatamente accessibile in GRAPH
shutil.rmtree("/tmp/tf_log") #cancella la directory precedente
summary_writer = tf.train.SummaryWriter("/tmp/tf_log", sess.graph_def)
merged_summaries = tf.merge_all_summaries()

#Avvia la sessione TensorFlow.
sess.run(init)

#Avvio il processo di apprendimento.
#Il numero totale di epoche viene definito
#nella variabile tot_epoch.
tot_epoch = 10000
for epoch in range(tot_epoch):
        #Questa linea di codice racchiude tutta la logica dietro TensorFlow.
        #La sessione e' stata avviate e tutte le variabili allocate in memoria.
        #Posso interrogare il grafo specificando i valori che voglio vengano
        #calcolati, in questo caso: [optimizer, merged_summaries, output, cost, W, B]
        #Come valori di ingresso devo dare: {_input: dataset_input, _target: dataset_target}
        #Salvo i valori di uscita nelle variabili locali: _, my_summary, my_output, my_cost, my_weights, my_bias 
	_, my_summary, my_output, my_cost, my_weights, my_bias = sess.run([optimizer, merged_summaries, output, cost, W, B], 
                                                                          feed_dict={_input: dataset_input, _target: dataset_target})
        summary_writer.add_summary(my_summary, epoch) #aggiunge al sommario le variabili monitorate

        #Stampo sul terminale i valori di tutte le variabili ogni 1000 epoche.
	if epoch % 1000 == 0:
                print("")
		print("Epoch ..... " + str(epoch))
		print("Cost ..... " + str(my_cost))
                print("Target: " + str(dataset_target[0]) + str(dataset_target[1]) + str(dataset_target[2]) + str(dataset_target[3]))
		print("Output: " + str(my_output[0]) + str(my_output[1]) + str(my_output[2]) + str(my_output[3]))
		print("W: " + str(my_weights[0]) + str(my_weights[1]))
		print("B: " + str(my_bias))

        #Stampo un sommario dell'addestramento una volta giunto all'ultima epoca.
	if epoch == (tot_epoch-1):
                print("")
		print('Training complete!')
                print("Tot Epochs ..... " + str(tot_epoch))
		print("Final Cost ..... " + str(my_cost))
		print("Estimated Weights: " + str(my_weights[0]) + str(my_weights[1]))
		print("Estimated Bias: " + str(my_bias))




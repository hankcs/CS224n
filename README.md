# CS224n
CS224n: Natural Language Processing with Deep Learning Assignments Winter, 2017

#### Requirements
* Python 2.7
* TensorFlow r1.2


## Assignment #1 

1. Softmax
2. Neural Network Basics
3. word2vec
![q3_word_vectors](http://wx2.sinaimg.cn/large/006Fmjmcly1fgydqi2vq4j30m80godgi.jpg)
4. Sentiment Analysis
![q4_reg_v_acc](http://wx1.sinaimg.cn/large/006Fmjmcly1fgydrwwnsbj30m80godgn.jpg)
![q4_dev_conf](http://wx1.sinaimg.cn/large/006Fmjmcly1fgydrmd0wtj30m80gojrx.jpg)

## Assignment #2

1. Tensorflow Softmax
2. Neural Transition-Based Dependency Parsing

```
924/924 [==============================] - 49s - train loss: 0.0631    
Evaluating on dev set - dev UAS: 88.54
New best dev UAS! Saving model in ./data/weights/parser.weights
================================================================================
TESTING
================================================================================
Restoring the best model weights found on the dev set
Final evaluation on test set - test UAS: 88.92
Writing predictions
Done!
```

3. Recurrent Neural Networks: Language Modeling
![unrolled_rnn](http://wx3.sinaimg.cn/large/006Fmjmcly1fgzqfm9p4xj30p60bbdgu.jpg)

## Assignment #3

1. A window into NER

```
DEBUG:Token-level confusion matrix:
go\gu   PER     ORG     LOC     MISC    O    
PER     2968    26      84      16      55   
ORG     147     1621    131     65      128  
LOC     48      88      1896    26      36   
MISC    37      40      54      1030    107  
O       42      46      18      39      42614
DEBUG:Token-level scores:
label   acc     prec    rec     f1   
PER     0.99    0.92    0.94    0.93 
ORG     0.99    0.89    0.77    0.83 
LOC     0.99    0.87    0.91    0.89 
MISC    0.99    0.88    0.81    0.84 
O       0.99    0.99    1.00    0.99 
micro   0.99    0.98    0.98    0.98 
macro   0.99    0.91    0.89    0.90 
not-O   0.99    0.89    0.87    0.88 
INFO:Entity level P/R/F1: 0.82/0.85/0.84
```

2. Recurrent neural nets for NER

```
DEBUG:Token-level confusion matrix:
go\gu   PER     ORG     LOC     MISC    O    
PER     2987    32      47      12      71   
ORG     136     1684    90      70      112  
LOC     39      83      1907    21      44   
MISC    43      45      47      1031    102  
O       36      56      15      34      42618
DEBUG:Token-level scores:
label   acc     prec    rec     f1   
PER     0.99    0.92    0.95    0.93 
ORG     0.99    0.89    0.80    0.84 
LOC     0.99    0.91    0.91    0.91 
MISC    0.99    0.88    0.81    0.85 
O       0.99    0.99    1.00    0.99 
micro   0.99    0.98    0.98    0.98 
macro   0.99    0.92    0.89    0.91 
not-O   0.99    0.90    0.88    0.89 
INFO:Entity level P/R/F1: 0.85/0.86/0.85
```

3. Grooving with GRUs

![q3-noclip-rnn](http://wx2.sinaimg.cn/large/006Fmjmcly1fh6mpycoobj30hs0dcmxt.jpg)
![q3-clip-rnn](http://wx1.sinaimg.cn/large/006Fmjmcly1fh6mq3kxzqj30hs0dcdgh.jpg)
![q3-noclip-gru](http://wx2.sinaimg.cn/large/006Fmjmcly1fh6mq9pbitj30hs0dcgmc.jpg)
![q3-clip-gru](http://wx2.sinaimg.cn/large/006Fmjmcly1fh6mqdhyb7j30hs0dcjs6.jpg)

```
DEBUG:Token-level confusion matrix:
go\gu	PER  	ORG  	LOC  	MISC 	O    
PER  	2920 	41   	57   	12   	119  
ORG  	101  	1716 	73   	64   	138  
LOC  	22   	95   	1908 	16   	53   
MISC 	37   	45   	53   	1017 	116  
O    	21   	67   	14   	39   	42618

DEBUG:Token-level scores:
label	acc  	prec 	rec  	f1   
PER  	0.99 	0.94 	0.93 	0.93 
ORG  	0.99 	0.87 	0.82 	0.85 
LOC  	0.99 	0.91 	0.91 	0.91 
MISC 	0.99 	0.89 	0.80 	0.84 
O    	0.99 	0.99 	1.00 	0.99 
micro	0.99 	0.98 	0.98 	0.98 
macro	0.99 	0.92 	0.89 	0.90 
not-O	0.99 	0.91 	0.88 	0.89 

INFO:Entity level P/R/F1: 0.86/0.85/0.85
```

4. Easter Egg Hunt!
    - Run `python q3_gru.py dynamics` to unfold your candy eggs


## References

CS224n official website

* http://web.stanford.edu/class/cs224n/index.html

Many code snippets come from

* https://github.com/rymc9384/DeepNLP_CS224N
* https://github.com/gxlzj/cs224n-hw3​


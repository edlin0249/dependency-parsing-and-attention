OS environment
===
Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-130-generic x86_64)

Prerequisites
===

## Download Multinli 0.9
```
wget https://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip
```

## Download Stanford Corenlp
```
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
```

## Download Java 1.8+
(Check with command: `java -version`) ([Download Page](http://www.oracle.com/technetwork/cn/java/javase/downloads/jdk8-downloads-2133151-zhs.html))

Requirement
===
```
tensorflow == 1.10
tqdm
stanfordcorenlp
nltk
jsonlines
```

Execute
===
## modify an absolute path for stanford corenlp in ***DependencyParsing.py***
```python=5
        self.nlp = StanfordCoreNLP('/home/edlin0249/iis_summer_intern/stanford-corenlp-full-2018-02-27') # modify your absolute path for stanford corenlp
```

## run
```
python3 main.py
```

Experiment
===
## baseline
### train
Acc = None
Cross Entropy Loss = None

### dev
Acc = None
Cross Entropy Loss = None

### test
Acc = None
Cross Entropy Loss = None

## w/ dependency parsing
### train
Acc = None
Cross Entropy Loss = None

### dev
Acc = None
Cross Entropy Loss = None

### test
Acc = None
Cross Entropy Loss = None

## w/ attention
### train
Acc = None
Cross Entropy Loss = None

### dev
Acc = None
Cross Entropy Loss = None

### test
Acc = None
Cross Entropy Loss = None
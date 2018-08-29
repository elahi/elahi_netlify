---
layout: single
title: Doctors without binders - data science consulting as an Insight Fellow
date: "2018-06-26T21:49:57-07:00"
category: 
tags:
excerpt:
---

<img src="/post/2018-06-26-phraze_consulting_files/tired_doctor_binders.png" alt="Tired doc." width="700px"/>

Physicians are overburdened by their binders full of paperwork. Consequently, doctors are often [staring at their computer screen](https://www.wsj.com/articles/is-your-doctor-getting-too-much-screen-time-1450118616) instead of interacting with their patient. In fact, the burden is so great that a new profession - [medical scribe](https://www.scribeamerica.com/) - has emerged. Medical scribes accompany physicians and manually transcribe their conversations and enter annotated notes into the electronic medical record. [Bank and Gage (2015)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4598196/) demonstrated that cardiovascular physicians with scribes:

  - saw 10% more patients per hour than physicians without scribes
  - increased revenue by $1.3 million
  - for a hefty annual salary of $99k

As a fellow with [Insight Data Science](https://www.insightdatascience.com/), I did a three-week consulting project with the start-up [Phraze](https://www.phraze.co/). Phraze is seeking to remove the burden of manual annotation with machine learning. That is, they are seeking to build a speech to medical record pipeline. The relevant bit here is that the transcribed speech needs to be tagged with the appropriate clinical codes (e.g., physical exam). 

My primary deliverable was to develop a supervised learning model to classify utterances between health care providers and consumers. An utterance, technically *speaking*, is a spoken word, statement, or vocal sound. Importantly, for our purposes, it is also a *unit of linguistic analysis*. Or a *phraze*, if you will. In this blog I describe how I classified utterances using natural language processing in `Python`. 

### The dataset

The dataset I obtained from Phraze were originally from an academic research group, and consisted of nearly 200 patient visits. Each visit was recorded and the audio was transcribed to text. The text was then entered into a spreadsheet with nearly 90k rows. Each row corresponded to a single utterance, and the columns described various features of each utterance (e.g., the role of the speaker; patient, doctor, nurse, mother, etc.). These features were annotated manually in 2013-2014, and importantly, these are the *true* labels I used for supervised learning. 

### Data cleaning and feature engineering

In `R`, I used the `tidyverse` to:

- Remove rows without any utterances ('Text')
- Remove the first two rows and last three rows of each visit, because these did not contain meaningful utterances. Instead, these were typically bracketed text with procedural information (e.g., name of the transcriber). 
- I removed patient visits with fewer than 50 utterances (rows); this resulted in 183 unique patient visits. The mean and median number of utterances per visit was 426 and 470, respectively.

Next, I had to extract useful information from the raw features. In short, I used a combination of different features that are standard in the medical coding of clinical encounters. I won't go into the details here, but the final categories reflect typical entries on an electronic medical record: 

<img src="/post/2018-06-26-phraze_consulting_files/phraze_classes.png" alt="phraze classes" width="700px"/>

I also collapsed the speaker roles into three categories:

> Patient  
Provider  (includes doctors and nurses)  
Other (includes other family members)
> 

Below is an example of a series of utterances between a doctor (bold) and patient (italics). In parentheses, I have labeled each utterances with the labeled class. Note that some utterances compose a single speaker turn:

> **I'm going to take a listen to you** (Physical exam)  
**all right?** (Physical exam)  
**Big breaths** (Physical exam)  
*There's just a wheezing from my back* (Patient history)  
**Yeah, you are wheezing** (Assessment)
> 

### Natural language processing and classification

In brief, the analysis pipeline was:

1. Balance the classes. There are many ways to do this, but I chose to keep it simple and sample rows randomly from each of the seven classes. The sample size of the least frequent class (physical exam) was used - therefore, for this class - all rows were used. 

<img src="/post/2018-06-26-phraze_consulting_files/balance_classes.png" alt="balance classes" width="700px"/>

2. Use natural language processing tools (`nltk`) to:
  - strip special characters (which were often not words, but appeared to be data entry mistakes)
  - tokenize words using regular expressions
  - and a Snowball stemmer reduce words into shorter root forms
  - create a feature matrix 
  
<img src="/post/2018-06-26-phraze_consulting_files/nlp_steps.png" alt="" width="700px"/>  

3. Use logistic regression for classification (`sci-kit learn`). I also tried multinomical naive bayes and random forest classifiers - but these resulted in equivalent, or slightly worse, accuracy. Moreover, the resulting coefficients from logistic regression are easily interpretable, and I can identify the words that are associated with each category easily.

I used grid search cross validation to tune the hyperparameters of the model - namely, which ngrams I should use (e.g., single token, or single token AND pairs of tokens), and whether to use bag of words or TFIDF. The grid search return ngram(1,2) and TFIDF. TFIDF stands for *term frequency inverse document frequency* - essentially, this approach deemphasizes frequently observed words across the classes. This was particularly important in my case, because I chose not to remove [stop words](https://pythonspot.com/nltk-stop-words/) - words like *the, is, are*. I did not remove stop words (e.g., by using a pre-existing library of stop words) because many of my utterances were short (1-10 words) to begin with - so I did not want to strip them down to nothing (in some cases). Moreover, there is good reason to keep those stop words - for example, doctors may be more likely to use the word *you*, whereas patients might say the word *I* more often.  

When I applied the final (tuned) model to the remaining 20% of data in the testing set, overall accuracy was 45%. If the 7 categories were balanced, we would expect 14% accuracy. So, this represents a 3-factor improvement over random. A reasonable start.  

Accuracy for each class ranged from ~30-65%. The classifier was particularly good at identifying phrases associated with physical exam. Below I plot accuracy by class, with the associated words for three of the classes:

<img src="/post/2018-06-26-phraze_consulting_files/class_accuracy.png" alt="" width="700px"/>

### Summary

In 3 weeks, I delivered proof of concept to Phraze - machine learning can be used to automatically classify the individual phrases of doctor-patient conversations. There are a number of possible steps to improve the classification - e.g., use an oversampling algorithm (e.g., SMOTE) to balance the classes without sacrificing data, modify the NLP steps, try different algorithms, and change the classes altogether with the new information in hand. It may also be useful to include the speaker's role in the classification. For example, doctors are more likely to speak about the physical exam, while 'other people' (e.g., the patient's mother) are more likely so speak about 'other things':

<img src="/post/2018-06-26-phraze_consulting_files/class_role.png" alt="" width="700px"/>

Ultimately, the text classifier will streamline the process of data entry in the electronic medical record - which means your doctor will be happy, and you'll be happy too. 

<img src="/post/2018-06-26-phraze_consulting_files/happy_doctor_patient.jpg" alt="" width="700px"/>
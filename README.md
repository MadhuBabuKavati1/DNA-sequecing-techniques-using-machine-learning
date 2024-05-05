# DNA-sequecing-techniques-using-machine-learning

**INTRODUCTION**
DNA sequencing is the process of determining the precise order of nucleotides (adenine, cytosine, guanine, and thymine) within a DNA molecule.
There are several techniques for DNA sequencing, including Sanger sequencing (also known as chain-termination sequencing), next-generation sequencing (NGS), and third-generation sequencing technologies like PacBio and Oxford Nanopore sequencing.

**DNA Structure**
DNA, or deoxyribonucleic acid, adopts a double-helix structure composed of two polynucleotide chains wound around each other. Each chain is made up of nucleotides containing a phosphate group, a deoxyribose sugar molecule, and one of four nitrogenous bases: adenine (A), thymine (T), cytosine (C), or guanine (G). Adenine pairs with thymine, and cytosine pairs with guanine, forming complementary base pairs connected by hydrogen bonds. The two polynucleotide chains run antiparallel to each other, with one strand oriented in the 5' to 3' direction and the other in the 3' to 5' direction. This double helix structure, elucidated by James Watson and Francis Crick in 1953, is fundamental to the storage and transmission of genetic information in living organisms. DNA carries the genetic instructions necessary for the development, functioning, and reproduction of organisms, serving as the template for processes such as DNA replication, transcription, and translation. In eukaryotic cells, DNA is organized into chromosomes within the nucleus, collectively constituting the genome of an organism. Understanding the structure of DNA is crucial for unraveling the mechanisms underlying heredity, genetics, and molecular biology.

**Past Techniques**
Sanger Sequencing: Sanger and Coulson developed a sequencing method known as enzyme DNA sequencing or Sanger sequencing in the ’70s this will be utilizing DNA polymerase which is different from the previous non-enzymatic approach.
NGS Bioinformatics: When compared to the old techniques the data has increased a lot and it is required to continue to develop informatics algorithms that will be translating the results of the DNA sequencing and the NGS platform will be analyzing and manage the data which is generated using statistical methods and bioinformatics tools.

**Evolution of Techniques**
Third-Generation Sequencing: The Third Generation Sequencing can sequence single molecules of DNA without the need for clonal amplification prior to sequencing. This will be avoiding the introduction of artifacts from the PCR and this will require less manipulation of the sample.
Applications of Next-generation sequencing: There are many types of sequencing such as whole genome resequencing, targeted resequencing, gene expression analysis with whole transcriptome analysis, small RNA sequencing, and methylation analysis.

**Flowchart**
<img width="387" alt="Screenshot 2024-05-04 at 10 49 45 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/222f44a5-300d-44ff-a74f-2eac3de039c2">

**Dataset Description**
We take an exploratory dataset (Chauhan, DNA sequence dataset 2021) from Kaggle. It consists of 3 organisms - Human, Chimpanzee, and Dog and seven gene families as stated below:
<img width="451" alt="Screenshot 2024-05-04 at 10 51 24 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/5059805b-ac65-4300-b81d-b1c78d88d905">

**Data Preprocessing**
Standardizing file formats and displaying DNA sequences in classes. For example below we see the class Distribution of Human DNA from our dataset.
<img width="555" alt="Screenshot 2024-05-04 at 10 52 56 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/dd5d155b-b68c-4643-aafd-f165080c6ba6">

**Data Cleaning**
Small letters should be used for all sequence characters. If DNA contains any characters that are not in (A, C, G, or T), put them in as (z) characters.
<img width="740" alt="Screenshot 2024-05-04 at 10 54 20 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/c78f777f-b7c4-448a-b295-d39b7d1baa09">

**Feature Engineering**
A metaphorical comparison to the language of life would be DNA and protein sequences. For the molecules that are present in all living things, language also carries instructions for how they should function. With the genome serving as the book, subsequences (genes and gene families) serving as the sentences and chapters, k-mers and peptides (motifs) serving as the words, and nucleotide bases and amino acids serving as the alphabet, the sequence language analogy continues. Given how appropriate the parallel looks, it makes sense that the incredible work done in the field of natural language processing would also apply to the natural language of DNA and protein sequences.

Here, we employ a straightforward and easy strategy. We first divide the lengthy biological sequence into overlapping k-mer length "words." If we use "words" of length 6 (hexamers), for instance, "ATGCATGTCA" becomes "TGCATG", ATGCAT," "GCATGC," and "CATGCA." In light of this, our example sequence is divided into 4 hexamer words. These types of modifications are referred to as "k-mer counting" in genomics, which is the process of counting the occurrences of each potential k-mer sequence.
<img width="609" alt="Screenshot 2024-05-04 at 10 55 18 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/79f313f4-bfa9-4242-9710-0d85bed92c4b">

We next use the Scikit-Learn natural language processing tools to perform the k-mer counting; however, in order to apply the count vectorizer, we must first convert the lists of k-mers for each gene into string sentences of words.

The Bag of Words model is now developed using CountVectorizer. Earlier testing established the n-gram size of 4.

We have 4380 genes for humans that have been transformed into 232414 feature vectors of uniform length measuring 4-gram k-mer (length 6) counts. With 1682 and 820 genes respectively, chimpanzees and dogs both share the same number of features.
<img width="579" alt="Screenshot 2024-05-04 at 10 56 01 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/0b7a244c-d3e4-4ad6-a923-a444218b47fb">

**Splitting the human dataset into the training set and test set**
The final preprocessing step is to separate the data into training and test sets once it has been separated into inputs and labels. The "train test split" method from the "model selection" library of the "Scikit-Learn library" enables us to easily split data into training and test sets with 80% Train dataset & 20% Test dataset.
Thus we have 3504 sequences in the training data and 876 sequences in the testing data.
<img width="523" alt="Screenshot 2024-05-04 at 10 56 51 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/35763e67-e687-4300-8a60-8c463e8baf6d">

**Model Selection**
Our next step is to build a Multinomial Naive Bayes classifier. The most effective n-gram size and model alpha are 4 and 0.1, according to prior parameter tuning results.
<img width="631" alt="Screenshot 2024-05-04 at 10 58 43 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/38922785-3395-42e0-af73-8db492f77434">

**Evaluation Metrics**
<img width="587" alt="Screenshot 2024-05-04 at 10 59 24 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/9124393a-573a-4be9-9578-e595c8185e72">

**Result Analysis**
<img width="740" alt="Screenshot 2024-05-04 at 11 00 15 PM" src="https://github.com/MadhuBabuKavati1/DNA-sequecing-techniques-using-machine-learning/assets/162259846/11df4c13-c302-4ee8-b54d-bbf77565168e">

**Future Scope**
Technology is increasing and the techniques for DNA sequencing are increasing.
Epigenomic variation is an extension of genome resequencing applications and this will also be investigated using the next-generation sequencing approaches which will enable the ascertainment of genome-wide patterns of methylation and the organism’s development will be based on the patterns that change through the course.
If the DNA sequencing is done rapidly then the experiment results will enhance the potential to combine the results of different experiments and this is the most exciting possibility. The secrets of the cell can be unlocked with the power of correlative analysis, correlative analysis of the genome-wide methylation, histone binding patterns, and gene expression for instance owing to the data that is produced similarly.\

**Conclusion**
It is evident that the New-Age DNA Sequencing techniques are exponentially more efficient than the past techniques. The 98.4% accuracy of our model is staggering proof of this claim. Hence it can be noted that the field of bioinformatics and genomics has made substantial progress.

This small step of Machine Learning incorporation into the Sequencing techniques can very well be a giant leap forward in solving major global illnesses. This along with the rapid development of the Genetic Engineering Domain could be the key to discovering a cure for the threats such as Cancer or hereditary disorders.

**References**
Mohamed, O. (2021, April 5). DNA sequencing with Machine Learning. DataValley. Retrieved December 20, 2022, from https://datavalley.technology/dna-sequencing-with-machine-learning/
Chauhan, N. S. (2021, January 15). DNA sequence dataset. Kaggle. Retrieved December 20, 2022, from https://www.kaggle.com/datasets/nageshsingh/dna-sequence-dataset � Gu, M. (2021, September 6). DNA sequence classification based on Milvus. Milvus. Retrieved December 21, 2022, from https://milvus.io/blog/dna-sequence-classification-based-on-milvus.md
Dixit, P., & Prajapati, G. I. (2015). Machine Learning in Bioinformatics: A novel approach for DNA sequencing. 2015 Fifth International Conference on Advanced Computing & Communication Technologies. https://doi.org/10.1109/acct.2015.73
Sarkar, S., Mridha, K., Ghosh, A., Shaw, R.N. (2022). Machine Learning in Bioinformatics: New Technique for DNA Sequencing Classification. In: Shaw, R.N., Das, S., Piuri, V., Bianchini, M. (eds) Advanced Computing and Intelligent Technologies. Lecture Notes in Electrical Engineering, vol 914. Springer, Singapore. https://doi.org/10.1007/978-981-19-2980-9_27
Shendure, J., Balasubramanian, S., Church, G. M., Gilbert, W., Rogers, J., Schloss, J. A., & Waterston, R. H. (2017). DNA sequencing at 40: Past, present and future. Nature, 550(7676), 345–353. https://doi.org/10.1038/nature24286
Mardis, E. R. (2008). Next-generation DNA sequencing methods. Annual Review of Genomics and Human Genetics, 9(1), 387–402. https://doi.org/10.1146/annurev.genom.9.081307.164359
Morey, M., Fernández-Marmiesse, A., Castiñeiras, D., Fraga, J. M., Couce, M. L., & Cocho, J. A. (2013). A glimpse into past, present, and future DNA sequencing. Molecular Genetics and Metabolism, 110(1-2), 3–24. https://doi.org/10.1016/j.ymgme.2013.04.024










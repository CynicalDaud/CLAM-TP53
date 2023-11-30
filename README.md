# Computational Pathology

## 1 Tissue Segmentation

Tissue segmentation is a task in computational pathology that involves the automatic identification and separation
of tissue regions from other structures in digital pathology images. It is a crucial step in many image analysis
applications, including diagnosis, grading, and prognosis of various diseases such as cancer. Tissue segmentation
algorithms typically involve the use of various image processing techniques such as thresholding, edge detection,
morphological operations, clustering, and machine learning algorithms such as deep neural networks. These algo-
rithms aim to separate the regions of the image that correspond to tissue from those that correspond to other
structures such as background, artifacts, and non-tissue regions. The accurate segmentation of tissue regions is
important for many downstream analysis tasks in computational pathology such as nuclei segmentation, cell count-
ing, and tissue classification. Tissue segmentation can also aid in reducing the computational burden associated
with analyzing large datasets by excluding non-tissue regions from the analysis.

In order to carry out tissue segmentation, the TIAToolBox library was used. WSIReader.open()is used to
load asvsfile containing each whole-slide image. Thesavetilesfunction generates image tiles from a whole
slide image. The function takes several arguments such as the output directory where the tiles will be saved, the
objective value at which the tiles will be generated, the size of each tile, the file format in which the tiles will be
saved, and a verbose flag for printing output. The function calculates the tile read size, slide height, and slide
width, and sets up the output directory. It then calculates the number of vertical and horizontal tiles needed to
cover the slide and iterates through each tile. For each tile, it calculates the start and end indices for the tile in
the slide, reads the image region using the readbounds method of the object, rescales the image if necessary, and
saves the tile to the output directory.

Having done that thegetmasks()function firstly checks whether a Python Pickle file containing that slides
mask already exists. If not it generates the mask withtissuemaskthat generates the mask given a WSI loader.
Tissue segmentation withTIAToolboxuses patching along with masking to extract tissue information from a WSI.
The benefit of this approach is that individual patches can now be higher resolution, meaning that there is more
detail and thus information in a dataset created in this way.

![output](https://github.com/CynicalDaud/CLAM-TP53/assets/10792026/6570ebe8-cb82-46b3-bbee-796cc8a65630)
Figure 1: The two specified WSI’s (TCGA-05-4382 and TCGA-67-6216) plotted alongside their corresponding
masks.

## 2 CLAM: MIL for TP53 Prediction

To carry out this task I chose to adapt the CLAM pipeline [3]. It is a deep learning framework designed for
computational pathology. It addresses five key challenges in whole slide-level pathology, including data efficiency,
high-throughput analysis, weakly-supervised learning, multi-class classification, and interpretability of results. CLAM
uses attention-based learning to identify sub-regions of high diagnostic value and refine the feature space. It aggre-
gates patch-level features into slide-level representations for classification using an attention-based pooling function
that assigns an attention score for each patch. This attention score informs the contribution or importance of
the patch to the collective, slide-level representation for a specific class. CLAM is designed for generic multi-class
classification problems, with n parallel attention branches that calculate n unique slide-level representations. Each
representation is determined from a different set of highly-attended regions in the image viewed by the network
as strong positive evidence for one of n classes in a multi-class diagnostic task. To address the data inefficiency
in existing weakly-supervised learning algorithms for computational pathology, CLAM uses attention-based pooling
instead of max pooling and generates pseudo-labels for both highly-attended and weakly-attended patches. During
training, the network learns from an additional supervised learning task of clustering the most and least attended
patches of each class into distinct clusters.

CLAM produces highly interpretable heatmaps that allow clinicians to visualize the relative contribution and im-
portance of every tissue region to the model’s predictions without using any pixel-level annotations during training.
These heatmaps demonstrate that CLAM models can identify well-known morphological features used by patholo-
gists to make diagnostic determinations. CLAM is publicly available as an easy-to-use Python package over GitHub,
and whole slide-level attention maps can be viewed in an interactive demo.

### 2.a Architecture

The CLAMSB architecture consists of an attention network with four sequential layers. The first layer is a linear
layer with 1024 input features and 512 output features, followed by a ReLU activation function and a dropout layer
with a dropout probability of 0.25. The third layer is an attention network with a gated mechanism, which contains
another sequential module. This module has a linear layer with 512 input features and 256 output features, followed
by a hyperbolic tangent activation function and a dropout layer with a dropout probability of 0.25.

Alongside CLAM I also implemented a second model based on the one described in “Attention-based Deep Multi-
ple Instance Learning”[2]. They also propose gated attention layers in a convolutional neural network that maps
onto a learned set of features. Their model suggest training both the feature-extractor and the attention layer
simultaneously, which posed some issues when adapting it to our very large datasets (they experimented with some
low resolutions WSI’s but trained mostly on MNIST[1], a dataset of handwritten images). This implementation
was abandoned in favour of CLAM for this reason.

### 2.b Data Processing

The data processing required here is minimal. The CLAM MIL model is trained on variably-sized bags containing
feature vectors. The only additional constraint is that all feature vectors must have the same size (i.e each WSI
must undergo feature extraction following the same methodology). We have been provided a set of 100 WSI’s in
bothsvsandpklformat, where the latter is a compressed binary archive that can be loaded back into python to
render a dictionary containing patch identifiers along with their respective feature vectors. This means that we can
train CLAM directly on the features from thesepklfiles with some simple modifications to the CLAM pipeline code.

These modifications include:

- Modify theCustomMILDatasetclass: The model requires a data-loader that feeds in batches containing
    a bag (set of feature vectors), a bag label (in our case “MUT” or “WT”) and finally the coordinates of
    each patch corresponding the the provided feature vectors (by index). The default CLAM dataset has to be
    provided with acsvfile that associates a slide-ID with a label. By default, the dataset class retrieves apt
    file from a specified directory that shares a name with the slide-ID from a row in thecsv, and parses that
    information into three variables that are then returned to the DataLoader class. The getitem function
    can be modified to search forpklfiles instead.
- Change label pointers: The CLAM pipeline makes a few assumptions about the data provided that are not
    necessarily true for the data we have been given. These relate primarily to the specificity of our dataset naming conventions. Firstly, the labels in our classification problem are different to the default CLAM case. Secondly the column names in our database are also different to what CLAM expects, and while one could modify our data instead, it was decided that minimising data-preprocessing would be ideal. Finally, wheres CLAM assumes there are multiple slides from the same patient in the dataset, and therefore enables cross-patient validation for loss calculation in the backwards pass, our data does not explicit this condition. We therefore modify various functions from the __utils__ and models __modules__.

### 2.c Training Procedure

As a train/test split was already provided, the dataset is firstly divided into training and testing examples. The
CLAM pipeline has a built in function that carries outn-fold validation on a test dataset. To do this it uses a
customsplitclass that was unchanged for the purpose of validation in this coursework. 10-fold validation with a
0 .75 split was used was used here.

In order to avoid overfitting and improve the quality of learned inferences, this model was trained using the
dropoutflag that adds a dropout layer to the end of both of the attention maps used in the gated-attention
layer. This adds non-linearity, and paired with early-stopping allows the model to avoid any local minima’s in the
loss function. Each fold trains for a maximum of 10 epochs with a learning rate of 2e−4. This learning rate was
chosen because, while it is the suggested value provided in the CLAM documentation, it also seems to yield the
best results on this dataset as well.

The CLAM pipeline utilises both instance and bag level loss to improve the rate of learning of theCLAMSB model.
In this procedure we specify the use of SVM-loss on an instance level. Support vector machines are often used for
binary classification and find a decision boundary that maximises the margin between the two classes. This loss
function is also known has hinge-loss:

```
L(y , f(x)) = max(0, 1 −y(x))
```

On a bag level, cross-entropy loss is used. This measures the difference between predicted and true values by
comparing their probability distributions, penalising a model for predicted distributions that are further from the
true values:

```
L(y , p) =−[y×log(p) + (1−y)×log(1−p)]
```

The models are trained using ADAM optimisation (adaptive moment estimation) that maintains a running estimate
of the first and second moments of the loss gradient (where the first is the average of the gradient and the second
is the average of the squared gradient). It uses these bias-corrected estimates to avoid being zero-biased at the
start of training. Full summary statistics are included inresults.pdfincluded in the submission.

Having validated the models ability to learn on the data, I trained the model on the entirety of the training data.
Validation in each epoch was carried out on the test split. Trained as described, CLAM achieves a respectable
accuracy of about 64.4%. Included below are some visualisations of the summary statistics for this final model:

![ROC](https://github.com/CynicalDaud/CLAM-TP53/assets/10792026/bb1ef12d-3e5d-4e45-9f4f-eeeb61eda8ba)
Figure 2: The ROC curve and corresponding AOC value for this classifier. A perfect classifier would have an AUC-
ROC of 1, meaning that it achieves 100% true positive rate at 0% false positive rate. A classifier that performs
no better than random guessing has an AUC-ROC of 0.5, which corresponds to a diagonal line in the ROC plot. It
can be said that this classifier performs better than a random classifier and therefore has learned how to do some
form of inference.

<img width="366" alt="final-accs" src="https://github.com/CynicalDaud/CLAM-TP53/assets/10792026/45cd0c2f-f927-4e6c-ba61-0d822c933164">
Figure 3: The validation accuracy over time for the final model, trained on all training data. Two plots are provided
here, one corresponding to the 0 class (WT) and the other to the 1 class (MUT).

<img width="338" alt="final-loss" src="https://github.com/CynicalDaud/CLAM-TP53/assets/10792026/e8384e02-d3ba-468e-9f67-22896d507a94">
Figure 4: The total-loss for the model (both bag and instance level) plotted over the course of 10 epochs while
being trained on the entirety of the training data.


## 3 Heat-Mapping

The aforementioned coordinates are not needed for model training but instead are used later in the heat-mapping
process.CLAM visualizes heatmaps by computing attention scores for all patches extracted from a slide using
the attention branch that corresponds to the model’s predicted class. These scores are converted to percentile
scores and scaled to between 0 and 1.0, with 1.0 being the most attended and 0.0 being the least attended.
The normalized scores are then converted to RGB colors using a diverging colormap and displayed on top of their
respective spatial locations in the slide to visually identify and interpret regions of high and low attention. To create
more fine-grained heatmaps, slides or smaller ROIs are tiled into 256 x 256 patches using an overlap and the same
procedure is followed. The heatmaps are then overlaid over the original image with a transparency value of 0.
to visualize the underlying morphological structures. See below two such heat-maps generated on aMUTand aWT
sample.

![TCGA-05-4395_blockmap (WT)](https://github.com/CynicalDaud/CLAM-TP53/assets/10792026/21fb4e13-06ae-4281-bebe-dc22daa94147)
Figure 5: Heatmap generated from the TCGA-05-WSI (a WT sample).

![TCGA-80-5607_blockmap (MUT)](https://github.com/CynicalDaud/CLAM-TP53/assets/10792026/4236f56f-3dac-408e-849e-4309c438ba07)
Figure 6: Heatmap generated from the TCGA-80-5607 WSI (a MUT sample).

Even with an accuracy of 64%, using the above-describe training procedure to train CLAM for classification we
obtain results that clearly highlight the sections of the sample which are indicative of mutation. The growth and
expansion of presumably mutated cancerous tissue is far more noticeable in theMUTsample (as to be expected),
with darker red patches that form larger groups of visually severe tissue patches. The model has also succeeded in
identifying some outlier structures indicative of the mutation.

## References

[1] L. Deng, “The mnist database of handwritten digit images for machine learning research,”IEEE Signal
Processing Magazine, vol. 29, no. 6, pp. 141–142, 2012.

[2] M. Ilse, J. M. Tomczak, and M. Welling,Attention-based deep multiple instance learning, 2018. arXiv:1802.
04712 [cs.LG].

[3] M. Y. Lu, D. F. K. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, and F. Mahmood,Data efficient and
weakly supervised computational pathology on whole slide images, 2020. arXiv:2004.09666 [eess.IV].



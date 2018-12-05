## DPM: Distance Precision Matrix


### Installation

First, install the necessary R packages from CRAN:

```r
install.packages(c("corpcor", "devtools"))
```

Then install DPM using the `install_github` function in the
[devtools](https://github.com/hadley/devtools) package.

```r
library(devtools)
devtools::install_github("ghanbari/DPM",host="github.molgen.mpg.de/api/v3")
library(DPM)
```

### Format of input data
The (reg.)DPM function takes as input argument a matrix DataMat where each column of the matrix corresponds to a variable and each row corresponds to a sample. The column names of the DataMat specify the names of the variables. You can also specify the name of samples as rownames but this is not necessary. For example, the following code generate a fake data matrix:

```r
DataMat <- matrix(sample(1:10, 200, replace=TRUE), ncol=10, nrow=20)
colnames(DataMat) <- paste("variable", 1:10, sep="")
rownames(DataMat) <- paste("Sample", 1:20, sep="")
head(DataMat)
```
In the context of gene networks variables correspond to genes. The input is gene expression data as a matrix in which columns correspond to genes and rows correspond to samples. As an example, we have one data from DREAM4 challenge (insilico_size10_1_knockouts):

```r
library(DPM)
data(dream4)
head(dream4)
```

### Computing DPM and reg-DPM
The following example computes DPM and reg-DPM scores for dream4 data. As previously discussed, you can use your own data as a matrix containing samples in rows and variables in columns.

```r
library(DPM)
data(dream4)
dpmres <- dpm(dream4)
regdpmres <- reg.dpm(dream4)
```


### Visualization of the network

You can visualize the constructed network using [qgraph](https://cran.r-project.org/web/packages/qgraph/index.html) package. First you need to install it:

```r
install.packages("qgraph")
```

Then you can visualize the network with different options using following parameters (for more details check the documentation of [qgraph](https://cran.r-project.org/web/packages/qgraph/index.html) package):

`layout`: this argument controls the layout of the graph. "circle" places all nodes in a single circle and "spring" gives a force embedded layout. Defaults to "circular" in weighted
graphs and "spring" in unweighted graphs.

`minimum`: edges with absolute weights under this value are not shown (but not omitted). Defaults to 0.

`threshold`: a numeric value that defaults to 0. Edges with absolute weight that are not above this value are REMOVED from the network. This differs from `minimum` which simply changes the scaling of width and color so that edges with absolute weight under minimum are not
plotted/invisible.   Setting  a  threshold  influences  the  spring  layout  and  centrality  measures obtained  with  the  graph  whereas  setting  a  minimum  does  not.  


`cut`: in weighted graphs, this argument can be used to cut the scaling of edges in width and color saturation. Edges with absolute weights over this value will have the strongest color intensity
and become wider the stronger they are,  and edges with absolute weights under this value will have the smallest width and become vaguer the weaker the weight. If this is set to NULL, no cutoff is used and all edges vary in width and color.  Defaults to NULL for graphs with less then 20 nodes. For larger graphs the
cut value is automatically chosen to be equal to the maximum of the 75th quantile of absolute edge strengths or the edge strength corresponding to 2n-th edge strength (n being the number of nodes.)


```r
library(qgraph)
qgraph(dpmres, layout = "spring" , minimum=0.25, cut=0.55, filename="dpm-network", filetype="png")
qgraph(regdpmres, layout = "spring" , minimum=0.25, cut=0.55, filename="regdpm-network", filetype="png")
```

### Threshold selection
We recommend you to look at the distribution of the scores and based on that decide for a threshold. You can plot the distribution of DPM and reg-DPM scores using:

```r
library("ggplot2")
dpmdat <- data.frame(score = abs(dpmres[lower.tri(dpmres)]),name ='DPM')
regdpmdat <-data.frame(score = abs( regdpmres[lower.tri(regdpmres)]), name='reg-DPM')
dat <- rbind(dpmdat,regdpmdat)

ggplot(dat, aes(x=score,color=name))+ theme_bw()  + geom_density(size=1.5)+scale_color_manual(values=c("DPM"="SteelBlue3","reg-DPM"="dodgerblue4"))
ggsave(“ScoreDistribution.png”, plot = last_plot(), device = “png”)
```

When you decide for the threshold you can visualize the resulted network (see section [Visualization of the network](#visualization-of-the-network) ) and get all the links with a score higher than the threshold (see subsection [Get only the links with a score higher than a threshold](#get-only-the-links-with-a-score-higher-than-a-threshold) )


However, we provide a threshold selection method based on k-means clustering. You can compute the threshold for the DREAM4 data with the following command where:

* `score_matrix` is the score matrix obtained from DPM or reg.DPM.

```r
dpmtr <- kmeans_links(dpmres)$threshold
regdpmtr <- kmeans_links(regdpmres)$threshold
```

You can visualize the position of the threshold on the distribution of the scores:

```r
library("ggplot2")
dpmdat <- data.frame(score = abs(dpmres[lower.tri(dpmres)]),name ='DPM')
regdpmdat <-data.frame(score = abs( regdpmres[lower.tri(regdpmres)]), name='reg-DPM')
dat <- rbind(dpmdat,regdpmdat)

ggplot(dat, aes(x=score,color=name))+ theme_bw()  + geom_density(size=1.5)+scale_color_manual(values=c("DPM"="SteelBlue3","reg-DPM"="dodgerblue4"))+ geom_vline(xintercept=dpmtr, size=1, color="SteelBlue3") + geom_vline(xintercept=regdpmtr, size=1,color="dodgerblue4")

ggsave(“ScoreDistribution.png”, plot = last_plot(), device = “png”)
```


You can also get all the links considered as true links based on k-means clustering:

```r
dpm_links <- kmeans_links(dpmres)$Listlink
regdpm_links <- kmeans_links(regdpmres)$Listlink
```

As an alternative, you can select the top-ranked links based on (reg.)DPM score which are most likely to be true links.
See section [Get only the top ranked links](#get-only-the-top-ranked-links)

### Get the list of the links
You need to first install the necessary R packages from CRAN:

```r
install.packages("reshape2")
```

#### Get all the links

You can get the list of all links (from the highest to the lowest score) with the following command, where the resulting dpm/reddpm_list contains the ranking of links. Each row corresponds to a link. The first and second columns show the corresponding nodes and the third column indicates the score of the link.

```{r}
library(reshape2)
dpm_links <- get_link(dpmres)
dim(dpm_links)
head(dpm_links)

regdpm_links <- get_link(regdpmres)
dim(regdpm_links)
head(regdpm_links)
```

#### Get only the links with a score higher than a threshold
If you are interested in extracting the links with scores higher than a threshold use the parameter `threshold` and specify your threshold:

```r
dpm_links <- get_link(dpmres,threshold = 0.2)
dim(dpm_links)
head(dpm_links)

regdpm_links <- get_link(regdpmres,threshold = 0.2)
dim(regdpm_links)
head(regdpm_links)
```
#### Get only the top ranked links

If you are interested in extracting the top links with the highest scores, use the parameter `top_ranked` and specify the number of top-ranked links to report:

```r
dpm_links <- get_link(dpmres,top_ranked = 10)
dim(dpm_links)
head(dpm_links)

regdpm_links <- get_link(regdpmres,top_ranked = 10)
dim(regdpm_links)
head(regdpm_links)
```

### Simulation of data
#### linear data
To simulate linear data you need to first install the necessary R packages from CRAN:

```r
install.packages(c("pcalg","mvtnorm","qpgraph"))
```
Then you can simulate linear data by specifying your desired number of nodes (`nnodes`) and number of samples (`nsamp`).

```r
library(DPM)
data <- rdata(nnodes=10,nsamp=100)[["data"]]
dpmres <- dpm(data)
regdpmres <- reg.dpm(data)
```


#### non-linear data
To simulate non-linear data from the network mentioned in the paper with your desired number of samples (`nsamp`) and the standard deviation of the noise added to the generated data (`sigma`) use:

```r
library(DPM)
data <- nldata(nsamp=100,sigma=0.2)[["data"]]
dpmres <- dpm(data)
regdpmres <- reg.dpm(data)
```

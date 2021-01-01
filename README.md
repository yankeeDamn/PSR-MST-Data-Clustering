# PSR-MST Image Segmentation
A python implementation of the method described in the paper [Sequential image segmentation based on minimum spanning tree representation](https://www.sciencedirect.com/science/article/abs/pii/S0167865516301192).


## Reference:

```
@article{SAGLAM2017155,
title = "Sequential image segmentation based on minimum spanning tree representation",
journal = "Pattern Recognition Letters",
volume = "87",
pages = "155-162",
year = "2017",
issn = "0167-8655",
doi = "https://doi.org/10.1016/j.patrec.2016.06.001",
url = "http://www.sciencedirect.com/science/article/pii/S0167865516301192",
author = "Ali Saglam and Nurdan Akhan Baykan"
}
```

<img src="./images/Fig_3.jpg" height="50%" width="50%">

### Dependence:
The code depends on the following third-party libraries:
- fibheap 0.2.1

```
pip install fibheap
```

### Run demo: 
```
python Demo_segmentation.py
```

### Parameters:
*m* : The coefficient of the parameters c that calculated automatically using the differential of the PSR-MST in the [source paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865516301192). If no value is given, the default value is ``3``.

*l* : The length of the scanning frame (sub-string) that scans through the PSR-MST. If no value is given, the default value is ``"scale"`` that computed by ``int( sqrt(M * N) / 2``).

### Segmentation  fuction:

 ``labels = sequential_segmentation(img)`` -----> *m* = 3, *l* = "scale"

``labels = sequential_segmentation(img, m = 4)`` -----> *l* = "scale"

``labels = sequential_segmentation(img, m = 4, l = 200)``


<img src="./images/Fig_5.jpg" height="30%" width="30%">


## Reference:

```
@article{SAGLAM2017155,
title = "Sequential image segmentation based on minimum spanning tree representation",
journal = "Pattern Recognition Letters",
volume = "87",
pages = "155-162",
year = "2017",
issn = "0167-8655",
doi = "https://doi.org/10.1016/j.patrec.2016.06.001",
url = "http://www.sciencedirect.com/science/article/pii/S0167865516301192",
author = "Ali Saglam and Nurdan Akhan Baykan"
}
```

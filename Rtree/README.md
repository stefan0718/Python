# Python
Rtree

This is an R-tree that increase efficiency of range queries and nearest neighbor queries under huge geolocation data.

1.Documents
    In the compressed bundle here are several test files and a python file. dataSet1k.txt, dataSet10k.txt, dataSet20k.txt, dataSet50k.txt, dataSet100k.txt contain 1000, 10000, 20000, 50000, 100000 set of random 2D points respectively. rangeDataSet.txt and nearestNeighborDataset.txt respectively contains 100 set of range queries and 100 set of nearest neighbor queries. We programmatically created them for assignment testing. As you see, all the code we wrote is in the only python file: Rtree.py.

2.About Rtree.py
    Usage of all functions has been noted in this python file. 
    Before debugging, please change 2D points file, range query file and nn query file paths in line 343, 347 and 349.
    Then debug this file, you can get a lot of results. 
    Line 342 - 345 prints sequential scan time.
    Line 352 - 356 prints a list of number of points, whole time and average time of brutal force range query testing.
    Line 359 - 363 prints a list of point ids, whole time and average time of brutal force nearest neighbor query testing.
    Line 385 - 392 prints a list of number of points, whole time and average time of range query testing with r-tree.
    Line 396 - 403 prints a list of point ids, whole time and average time of range query testing with r-tree.

Please note that there is some error result that I cannot figure out when debugging nn query test. Other tests are fine though.

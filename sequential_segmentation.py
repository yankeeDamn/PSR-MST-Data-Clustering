# -*- coding: utf-8 -*-
"""
@author:
    
Saglam, Ali, & Baykan, N. A. (2017).

"Sequential image segmentation based on min- imum spanning tree representation".

Pattern Recognition Letters, 87 , 155–162.

https://doi.org/10.1016/j.patrec.2016.06.001 .

"""

import numpy as np
from fibheap import *
import math
import sys

# --------------------------------------------------------------------------------
# Segment an image:
# Returns a color image representing the segmentation.
#
# Inputs:
#           in_image: image to segment.
#           sigma: to smooth the image.
#           k: constant for threshold function.
#           min_size: minimum component size (enforced by post-processing stage).
#
# Returns:
#           num_ccs: number of connected components in the segmentation.
# --------------------------------------------------------------------------------

def sequential_segmentation(img, m = 3, l = "scale"):
    
    rows, cols, bands = img.shape
    
    ###################################################
    
    nodes, psr, mst = create_PSR_MST(img)
    
    ###################################################
    
    if l == "scale":
        l = int(math.sqrt(rows * cols) / 2)
        
    print("l : ",l)
    
    print("m : ",m)

    psrlen = len(psr)
    sumdiff = float(0)
    for i in range(1, psrlen):
        sumdiff += abs(psr[i][0] - psr[i - 1][0])
    c = float(sumdiff / (psrlen - 1))
    c *= m
        
    print("c : ",c)
    
    ###################################################
    
    labels = S_MST_segmentation(psr, mst, l, c)
    
    ###################################################
    
    
    labels = np.reshape(labels,(rows, cols))
    
    return labels
    
    
def create_PSR_MST(in_image):
    height, width, band = in_image.shape
    
    in_image = np.float64(in_image)

    
    nodesize = height * width;
    nodes = np.zeros(shape=(nodesize, 9), dtype=object);
    nodes[:,:] = -1

    # build graph
    edgessize = 2 * ((width * height * 4) - (3 * (width + height)) + 2)
    edges = np.zeros(shape=(edgessize, 3), dtype=object)
    nnum = int(0)
    enum = 0
    for y in range(height):
        for x in range(width):
            nnum = int(y * width + x)
            if x < width - 1:
                edges[enum, 0] = np.linalg.norm(in_image[y, x] - in_image[y, x + 1])
                edges[enum, 1] = nnum
                edges[enum, 2] = int(y * width + (x + 1))
                nodes[edges[enum, 1],5] = enum
                enum += 1
                edges[enum, 0] = edges[enum - 1, 0]
                edges[enum, 1] = edges[enum - 1, 2]
                edges[enum, 2] = edges[enum - 1, 1]
                nodes[edges[enum, 1],4] = enum
                enum += 1
            if y < height - 1:
                edges[enum, 0] = np.linalg.norm(in_image[y, x] - in_image[y + 1, x])
                edges[enum, 1] = nnum
                edges[enum, 2] = int((y + 1) * width + x)
                nodes[edges[enum, 1],7] = enum
                enum += 1
                edges[enum, 0] = edges[enum - 1, 0]
                edges[enum, 1] = edges[enum - 1, 2]
                edges[enum, 2] = edges[enum - 1, 1]
                nodes[edges[enum, 1],2] = enum
                enum += 1
            if (x < width - 1) and (y < height - 1):
                edges[enum, 0] = np.linalg.norm(in_image[y, x] - in_image[y + 1, x + 1])
                edges[enum, 1] = nnum
                edges[enum, 2] = int((y + 1) * width + (x + 1))
                nodes[edges[enum, 1],8] = enum
                enum += 1
                edges[enum, 0] = edges[enum - 1, 0]
                edges[enum, 1] = edges[enum - 1, 2]
                edges[enum, 2] = edges[enum - 1, 1]
                nodes[edges[enum, 1],1] = enum
                enum += 1
            if (x < width - 1) and (y < height - 1):
                edges[enum, 0] = np.linalg.norm(in_image[y, x + 1] - in_image[y + 1, x])
                edges[enum, 1] = int(y * width + (x + 1))
                edges[enum, 2] = int((y + 1) * width + x)
                nodes[edges[enum, 1],6] = enum
                enum += 1
                edges[enum, 0] = edges[enum - 1, 0]
                edges[enum, 1] = edges[enum - 1, 2]
                edges[enum, 2] = edges[enum - 1, 1]
                nodes[edges[enum, 1],3] = enum
                enum += 1

    edges = edges.tolist()
    for i in range(edgessize):
        edges[i] = tuple(edges[i])
        
    psr_mst = np.zeros(nodesize - 1, dtype=tuple)
    count = 0
    
    heap = makefheap()

    current = 0
    nodes[current,0] = 0
    
    mst = [[]] * nodesize
    
    for i in range(nodesize):
        mst[i] = []
    
    while True:   
        for i in range(1,9):
            edgenum = nodes[current,i]
            if edgenum == -1:
                continue
            if nodes[edges[edgenum][2],0] == -1:
                fheappush(heap, edges[edgenum])
                nodes[edges[edgenum][2],0] = 0
        
        psr_mst[count] = heap.extract_min().key
        
        mst[psr_mst[count][1]].append(psr_mst[count][2])
        mst[psr_mst[count][2]].append(psr_mst[count][1])
        
        current = psr_mst[count][2]
        count += 1;
        
        if (heap.num_nodes == 0):
            break
    
    psr_mst = psr_mst.tolist()
    
    return nodes, psr_mst, mst


def S_MST_segmentation(psr, mst, l, c):
    maxin1 = -1
    maxin2 = -1
    max1 = -1000
    max2 = -1000
    
    psrlen = len(psr)
    
    sys.setrecursionlimit(len(mst))
    
    labels = np.zeros(len(mst), dtype=int)

    searchlen = 2 * l + 1;
    
    if psrlen < searchlen:
        return False
    
    seg_id = 1
    
    start = l
    finish = psrlen - l
    
    for i in range(start, finish):
        
        first = i - l
        last = i - 1
        
        
        if maxin1 < first:
            max1 = psr[first][0]
            maxin1 = first
            for j in range(first + 1, last):
                if psr[j][0] > max1:
                    max1 = psr[j][0]
                    maxin1 = j
        elif psr[last][0] > max1:
                max1 = psr[last][0]
                maxin1 = last
                
        
        first = i + 1
        last = i + l
        
        
        if maxin2 < first:
            max2 = psr[first][0]
            maxin2 = first
            for j in range(first + 1, last):
                if psr[j][0] > max2:
                    max2 = psr[j][0]
                    maxin2 = j
        elif psr[last][0] > max2:
                max2 = psr[last][0]
                maxin2 = last
                
        
        if psr[i][0] > min(max1, max2) + c:  
            segsize1 = 0
            segsize2 = 0
            
            segsize1, labels = labelNodes(labels, mst, psr[i][1], psr[i][2], seg_id, 0)
            seg_id += 1
            
            mst[psr[i][1]].remove(psr[i][2])
            mst[psr[i][2]].remove(psr[i][1])

    return labels


def labelNodes(labels, mst, nodein, prein, segid, segsize):
    labels[nodein] = segid
    segsize += 1
    for i in range(len(mst[nodein])):
        if mst[nodein][i] != prein:
            segsize, labels = labelNodes(labels, mst, mst[nodein][i], nodein, segid, segsize)
    return segsize, labels
    
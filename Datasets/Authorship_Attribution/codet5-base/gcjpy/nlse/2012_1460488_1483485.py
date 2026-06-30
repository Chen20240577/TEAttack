#!/usr/bin/python

import string
import sys

f = open(sys.argv[1], 'r')

num = int(f.readline())

for i in range(num):
    s = f.readline().strip()
    t = s.translate(string.maketrans("yeqjpmslckdxvnribtahwfougz",
                                     "aozurlngeismpbtdhwyxfckjvq"))
    # print 'Case #{}:'.format(i+1), s
    print
    'Case #{}:'.format(i + 1), t

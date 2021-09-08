"""
Quick script to parse logs produced by resnet training - used to analyze GNS etc.
"""

import os
import sys
import re
logfile_path=sys.argv[1]
batch_size=int(sys.argv[2])
out_file = open('processed.csv', 'w')
with open(logfile_path, 'r') as f:
    optimizer_steps = 0 # accumulator
    for line in f:
        # pattern - Epoch (SI steps based): [89][4762/5004]	Time  0.334 ( 0.348)	Data  0.007 ( 0.009)	Loss 7.7291e-01 (8.5461e-01)	Acc@1  76.56 ( 78.98)	Acc@5  92.19 ( 92.63)
        m1 = re.search("Epoch.*?\[(?P<epoch>\d+)\]\[ *(?P<step>\d+)/(?P<total>\d+)\].*Data.*Loss.*?\((?P<avg_loss>.*?)\)", line)
        # pattern - gns=[779967.51991749]
        m2 = re.search("gns=\[(?P<gns>.*)\]", line)
        # print(line)
        if m1:
            epoch = int(m1.group('epoch')) # starts with zero
            si_steps = int(m1.group('step'))
            optimizer_steps += 10 # harcoded since we produce logs every log_interval = 10 steps
            examples = batch_size * optimizer_steps
            total = int(m1.group('total'))
            training_loss = float(m1.group('avg_loss'))
        if m2:
            gns = int(float(m2.group('gns')))
            print("{},{},{},{},{}".format(optimizer_steps, examples, epoch*total + si_steps, training_loss, gns), file=out_file)
    out_file.close()

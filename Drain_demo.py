#!/usr/bin/env python
import sys
sys.path.append('../')
import Drain

input_dir  = 'logs'  # The input directory of log file
output_dir = 'result/'  # The output directory of parsing results
log_file   = 'tfa_main.trc'  # The input log file name
log_format = '<Date> <Time> <time_zone> <Level> <Component> <log_tag> <Content>'  # HDFS log format
# Regular expression list for optional preprocessing (default: [])
pre_regex = [r'\]',
             r'\[']
regex      = [r'/[^\/]+',
              r'\d+',
              r'_[^\/]+_']
st         = 0.3  # Similarity threshold
depth      = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex, pre_rex = pre_regex)
parser.parse(log_file)
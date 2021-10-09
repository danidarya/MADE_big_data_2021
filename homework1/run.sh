#!/bin/bash
set -x

HADOOP_STREAMING_JAR=/opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar
DATA_PATH=/prices.csv
OUT_PATH=/output

hdfs dfs -rm -r $OUT_PATH
chmod a+x mapper.py reducer.py

yarn jar $HADOOP_STREAMING_JAR \
    -files mapper.py,reducer.py \
    -mapper "mapper.py" \
    -reducer "reducer.py" \
    -input $DATA_PATH \
    -output $OUT_PATH

#!/bin/bash
echo "---------------------------------------------------------"
echo -e "| 序号\t| 服务\t\t| 操作\t| 备注\t\t\t|"
echo "---------------------------------------------------------"
echo -e "| 1\t| 所有服务\t| 开启\t| 开启所有服务\t\t|"
echo -e "| 2\t| HADOOP服务\t| 开启\t| HDFS文件系统,YARN框架\t|"
echo -e "| 3\t| SPARK服务\t| 开启\t| SPARK计算引擎\t\t|"
echo -e "| 4\t| HBASE服务\t| 开启\t| HBASE数据库服务\t|"
echo "---------------------------------------------------------"
echo -e "| 5\t| 所有服务\t| 关闭\t| 关闭所有服务\t\t|"
echo -e "| 6\t| HADOOP服务\t| 关闭\t| HDFS文件系统,YARN框架\t|"
echo -e "| 7\t| SPARK服务\t| 关闭\t| SPARK计算引擎\t\t|"
echo -e "| 8\t| HBASE服务\t| 关闭\t| HBASE数据库服务\t|"
echo "---------------------------------------------------------"
read n
if [ $n -eq 1 ]; then
    echo "---------------------------------------------------------"
    echo -e "| 1\t| 所有服务\t| 开启\t| 开启所有服务\t\t|"
    echo "---------------------------------------------------------"
    cd /usr/local/hadoop/sbin
    . start-all.sh
    cd /usr/local/spark/sbin
    . start-all.sh
elif [ $n -eq 2 ]; then
    echo "---------------------------------------------------------"
    echo -e "| 2\t| HADOOP服务\t| 开启\t| HDFS文件系统,YARN框架\t|"
    echo "---------------------------------------------------------"
    cd /usr/local/hadoop/sbin
    . /start-all.sh
elif [ $n -eq 3 ]; then
    echo "---------------------------------------------------------"
    echo -e "| 3\t| SPARK服务\t| 开启\t| SPARK计算引擎\t\t|"
    echo "---------------------------------------------------------"
    cd /usr/local/spark/sbin
    . /start-all.sh
elif [ $n -eq 4 ]; then
    echo "---------------------------------------------------------"
    echo -e "| 4\t| HBASE服务\t| 开启\t| HBASE数据库服务\t|"
    echo "---------------------------------------------------------"
    cd /usr/local/hbase/bin
    . /start-hbase.sh
elif [ $n -eq 5 ]; then
    echo "---------------------------------------------------------"
    echo -e "| 5\t| 所有服务\t| 关闭\t| 关闭所有服务\t\t|"
    echo "---------------------------------------------------------"
    cd /usr/local/spark/sbin
    . stop-all.sh
    cd /usr/local/hadoop/sbin
    . stop-all.sh
elif [ $n -eq 6 ]; then
    echo "---------------------------------------------------------"
    echo -e "| 6\t| HADOOP服务\t| 关闭\t| HDFS文件系统,YARN框架\t|"
    echo "---------------------------------------------------------"
    cd /usr/local/hadoop/sbin
    . /stop-all.sh
elif [ $n -eq 7 ]; then
    echo "---------------------------------------------------------"
    echo -e "| 7\t| SPARK服务\t| 关闭\t| SPARK计算引擎\t\t|"
    echo "---------------------------------------------------------"
    cd /usr/local/spark/sbin
    . /stop-all.sh
elif [ $n -eq 8 ]; then
    echo "---------------------------------------------------------"
    echo -e "| 8\t| HBASE服务\t| 关闭\t| HBASE数据库服务\t|"
    echo "---------------------------------------------------------"
    cd /usr/local/hbase/bin
    . /stop-hbase.sh
else
    echo "错误，请重试！"
fi

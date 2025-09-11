#!/bin/bash
echo "-----------------------------------------"
echo -e "| 序号\t| 仓库\t\t| 备注\t\t|"
echo "-----------------------------------------"
echo -e "| 0\t| 全部\t\t| 所有仓库\t|"
echo -e "| 1\t| C_BOX\t\t| C仓库\t\t|"
echo -e "| 2\t| CPP_BOX\t| C++仓库\t|"
echo -e "| 3\t| JAVA_BOX\t| JAVA仓库\t|"
echo -e "| 4\t| PY_BOX\t| PYTHON仓库\t|"
echo -e "| 5\t| SH_BOX\t| SHELL仓库\t|"
echo "-----------------------------------------"
read n
if [ $n -eq 0 ]; then
	echo "-----------------------------------------"
	echo -e "| 1\t| C_BOX\t\t| C仓库\t\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/c_box
	git pull origin master:master
	echo "-----------------------------------------"
	echo -e "| 2\t| CPP_BOX\t| C++仓库\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/cpp_box
	git pull origin master:master
	echo "-----------------------------------------"
	echo -e "| 3\t| JAVA_BOX\t| JAVA仓库\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/java_box
	git pull origin master:master
	echo "-----------------------------------------"
	echo -e "| 4\t| PY_BOX\t| PYTHON仓库\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/py_box
	git pull origin master:master
	echo "-----------------------------------------"
	echo -e "| 5\t| SH_BOX\t| SHELL仓库\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/sh_box
	git pull origin master:master
elif [ $n -eq 1 ]; then
	echo "-----------------------------------------"
	echo -e "| 1\t| C_BOX\t\t| C仓库\t\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/c_box
	git pull origin master:master
elif [ $n -eq 2 ]; then
	echo "-----------------------------------------"
	echo -e "| 2\t| CPP_BOX\t| C++仓库\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/cpp_box
	git pull origin master:master
elif [ $n -eq 3 ]; then
	echo "-----------------------------------------"
	echo -e "| 3\t| JAVA_BOX\t| JAVA仓库\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/java_box
	git pull origin master:master
elif [ $n -eq 4 ]; then
	echo "-----------------------------------------"
	echo -e "| 4\t| PY_BOX\t| PYTHON仓库\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/py_box
	git pull origin master:master
elif [ $n -eq 5 ]; then
	echo "-----------------------------------------"
	echo -e "| 5\t| SH_BOX\t| SHELL仓库\t|"
	echo "-----------------------------------------"
	cd /usr/GitBox/sh_box
	git pull origin master:master
else
	echo "错误，请重试！"
fi

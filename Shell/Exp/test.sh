#!/bin/bash
echo "file name is $0"
test_name="$0"
echo "file name length is ${#test_name}"
a=$1
b=$2
echo "the first is $1"
echo "the second is $2"
sum=`expr $a + $b`
echo "sum is ${sum}"
test1="12345"
echo "test is ${test1}"
echo "test length is ${#test1}"
read test1
echo "test is ${test1}" >> test_file
echo "test length is ${#test1}" >> test
test2=(1 3 5 7 9 2 4 6 8 0)
for((i=0;i<10;i++))
do
	for((j=0;j<9-i;j++))
	do
		j1=`expr $j + 1`
		if [[ ${test2[$j]} -gt ${test2[$j1]} ]] 
		then
			temp=$test2[$j]
			echo "temp is ${temp}"
			test2[$j]=$test2[$j1]
			test2[$j1]=$temp
		fi
	done
done
for((i=1;i<=10;i++))
do 
	echo "the ${i} is ${test2[$i-1]}"
done


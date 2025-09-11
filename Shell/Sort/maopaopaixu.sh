#!/bin/bash
echo "please input number :"
read num
int_i=0
while (( ${num} > 0 ))
do 
	int_j=`expr $int_i + 1`
	echo "please input the ${int_j}"
	read test[$int_i]
	let int_i++
	let num--
done
echo "${test[*]}"

len=${#test[@]}
echo "${len}"

for((i=0;i<len;i++))
do
	for((j=0;j<len-i-1;j++))
	do
		if [ ${test[$j]} -gt ${test[$(($j+1))]}  ]
		then 
			((temp=test[j]))
			((test[j]=test[j+1]))
			((test[j+1]=temp))
		fi
	done
done
echo "${test[*]}"

#!/bin/sh

prog=$1
dirIn=$2
numIn=$3
numRange=$4
fileOut=$5

rm  $fileOut".tmp.csv"
touch  $fileOut".tmp.csv"

echo "$dirIn $fileOut orig++++++++++++++++++"
time python3 $prog Root_Words.csv Suffix_Dictionary.csv Suffix_Expansion.csv Word_Gender.csv $dirIn $numIn $numRange orig $fileOut

rm  $fileOut".tmp1.csv"
touch  $fileOut".tmp1.csv"

echo "$dirIn $fileOut diff++++++++++++++++++"
time python3 $prog Root_Words.csv Suffix_Dictionary.csv Suffix_Expansion.csv Word_Gender.csv $dirIn $numIn $numRange diff $fileOut

#echo "$dirIn $fileOut rag++++++++++++++++++"
#time python3 $prog Root_Words.csv Suffix_Dictionary.csv Suffix_Expansion.csv Word_Gender.csv $dirIn $numIn $numRange rag $fileOut

echo "$dirIn $fileOut combine++++++++++++++++++"
time python3 csv2json.py $fileOut".tmp.csv" $fileOut".tmp1.csv" $fileOut

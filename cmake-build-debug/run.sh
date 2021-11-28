#!/bin/bash

for ((i = 1; i < 5; i++ )); do
  echo "$i" attempt:
  for ((j=1; j < 17; j++)) do
    echo run app for "$i" threads...
    ./hpc_1 "$i" > attempt_"$j"_threads_"$j"
  done
done
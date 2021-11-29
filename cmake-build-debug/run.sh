#!/bin/bash

for ((i = 1; i < 6; i++ )); do
  echo "$i" attempt:
  for ((j=16; j > 0; j--)) do
    echo run app for "$j" threads...
    ./hpc_1 "$j" > attempt_"$j"_threads_"$j"
  done
done
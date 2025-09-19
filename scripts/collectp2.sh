#!/bin/bash

prefix="chtransformer_p2_586360"
output_file="${prefix}_predicts.txt"

files=()
for i in {1..13}; do
    files+=("logs/eval/runs/${prefix}_p2_${i}/predicts.txt")
done

cat "${files[@]}" > "$output_file"

echo "Created $output_file with $(wc -l < "$output_file") lines"

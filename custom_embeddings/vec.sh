#!/bin/sh
python3 context_mapping.py
echo "context mapping done"
python3 get_prob.py
echo "prob appended"
python3 prepare_data.py
echo "prepare data done"
python3 embeddings.py
echo "model training done"
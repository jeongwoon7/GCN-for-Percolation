# GCN-for-Percolation

Graph convolutional network-based unsupervised learning of percolation transition

by Moon-Hyun Cha and Jeongwoon Hwang, submitted September 2024


1. Data for ML models : configurations.csv, convifurations_test.csv
* They are generated by make_configurations.py

2. Code for machine learning : run.py, modeules.py
* In terminal, python3 run.py > stdout
* output files : df_p_graph.xlsx, df_performance.xlsx

3. Module containing classes & functions used for the ML : modules.py

4. Code for data visualization : plot_performance.py
* It plots performance graph and p-resolved performance map.
* input files : df_p_graph.xlsx, df_performance.xlsx

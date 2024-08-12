import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ratings_url = 'UNSW-NB15 - CSV Files/UNSW-NB15_1.csv'
ratings_df = pd.read_csv(ratings_url, header = None)
ratings_df.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
                 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
                 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin',
                 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
                 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
                 'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips',
                 'ct_state_ttl', 'ct_flw', 'is_ftp', 'ct_ftp', 'ct_srv_src',
                 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport',
                 'ct_dst_sport', 'ct_dst_src', 'label_10', 'label_2']
print(ratings_df)

ax = sns.scatterplot(x='proto', y='service', hue='label_2',
                     data=ratings_df)
plt.xlim(0,13)

plt.show()
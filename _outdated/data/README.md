Datos asociados al proyecto de tirosinemias.

> [!WARNING]
> This data is clinical data and thus we can't publish it. 

- `SLEIMPN_HOMA_PKU.xlsx` is the bane of my existence, the Excel full of horrors. 
- `data_handdling.py` makes a "clean" csv that can be tracked using DVC and other tools. Expect to edit it frecuently, 
  as different clinical teams use different (think _ instead of -) names for columns. It's just Excel to CSV using Pandas. 
- `multi_label.csv` drops the most personal data, but is also clinical and thus non-public. 

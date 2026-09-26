import sys, io, pandas as pd
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
for label, p in [('Train S1', 'data/train/train_source1.tsv'), ('Test S1', 'data/test/test_source1.tsv')]:
    df = pd.read_csv(p, sep='\t', usecols=['country'])
    print(label, df['country'].value_counts().to_dict())

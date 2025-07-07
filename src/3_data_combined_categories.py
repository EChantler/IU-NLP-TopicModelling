import os
import pandas as pd

# Load all targets from each domain file
raw_dir = os.path.join('data', 'raw')
files = {
    'news': 'news.csv',
    'academic': 'academic.csv',
    'social': 'social.csv',
}

targets = {}
for domain, fname in files.items():
    fpath = os.path.join(raw_dir, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        # Get unique targets for this domain
        targets[domain] = sorted(df['target'].dropna().unique())
    else:
        targets[domain] = []

# Print or save the unique targets for manual mapping
for domain, labels in targets.items():
    print(f"\nUnique targets for {domain}:")
    for label in labels:
        print(label)

# OUTPUT:
# Unique targets for news:
# alt.atheism
# comp.graphics
# comp.os.ms-windows.misc
# comp.sys.ibm.pc.hardware
# comp.sys.mac.hardware
# comp.windows.x
# misc.forsale
# rec.autos
# rec.motorcycles
# rec.sport.baseball
# rec.sport.hockey
# sci.crypt
# sci.electronics
# sci.med
# sci.space
# soc.religion.christian
# talk.politics.guns
# talk.politics.mideast
# talk.politics.misc
# talk.religion.misc

# Unique targets for academic:
# acc-phys
# adap-org
# alg-geom
# ao-sci
# astro-ph
# atom-ph
# bayes-an
# chao-dyn
# chem-ph
# cmp-lg
# comp-gas
# cond-mat
# cs
# dg-ga
# econ
# eess
# funct-an
# gr-qc
# hep-ex
# hep-lat
# hep-ph
# hep-th
# math
# math-ph
# mtrl-th
# nlin
# nucl-ex
# nucl-th
# patt-sol
# physics
# plasm-ph
# q-alg
# q-bio
# q-fin
# quant-ph
# solv-int
# stat
# supr-con

# acc-phys
# adap-org
# alg-geom
# ao-sci
# astro-ph
# atom-ph
# bayes-an
# chao-dyn
# chem-ph
# cmp-lg
# comp-gas
# cond-mat
# cs
# dg-ga
# econ
# eess
# funct-an
# gr-qc
# hep-ex
# hep-lat
# hep-ph
# hep-th
# math
# math-ph
# mtrl-th
# nlin
# nucl-ex
# nucl-th
# patt-sol
# physics
# plasm-ph
# q-alg
# q-bio
# q-fin
# quant-ph
# solv-int
# stat
# supr-con

# Unique targets for social:
# LanguageTechnology
# Python
# investing
# pytorch
# your master list of topics
topics = [
    "Computer Science & AI",
    "Electronics & Engineering",
    "Mathematics & Statistics",
    "Physical & Earth Sciences",
    "Biomedical & Health Sciences",
    "Business, Finance & Economics",
    "Politics & Government",
    "Sports & Recreation",
    "Religion & Philosophy",
    "Miscellaneous & For-Sale"
]

# Remap news_map to new categories
news_map = {
    "alt.atheism": "Religion & Philosophy",
    "comp.graphics": "Computer Science & AI",
    "comp.os.ms-windows.misc": "Computer Science & AI",
    "comp.sys.ibm.pc.hardware": "Electronics & Engineering",
    "comp.sys.mac.hardware": "Electronics & Engineering",
    "comp.windows.x": "Computer Science & AI",
    "misc.forsale": "Miscellaneous & For-Sale",
    "rec.autos": "Miscellaneous & For-Sale",
    "rec.motorcycles": "Miscellaneous & For-Sale",
    "rec.sport.baseball": "Sports & Recreation",
    "rec.sport.hockey": "Sports & Recreation",
    "sci.crypt": "Computer Science & AI",
    "sci.electronics": "Electronics & Engineering",
    "sci.med": "Biomedical & Health Sciences",
    "sci.space": "Physical & Earth Sciences",
    "soc.religion.christian": "Religion & Philosophy",
    "talk.politics.guns": "Politics & Government",
    "talk.politics.mideast": "Politics & Government",
    "talk.politics.misc": "Politics & Government",
    "talk.religion.misc": "Religion & Philosophy",
}

# Remap academic_map to new categories
academic_map = {
    "acc-phys": "Physical & Earth Sciences",
    "adap-org": "Computer Science & AI",
    "alg-geom": "Mathematics & Statistics",
    "ao-sci": "Physical & Earth Sciences",
    "astro-ph": "Physical & Earth Sciences",
    "atom-ph": "Physical & Earth Sciences",
    "bayes-an": "Mathematics & Statistics",
    "chao-dyn": "Mathematics & Statistics",
    "chem-ph": "Physical & Earth Sciences",
    "cmp-lg": "Computer Science & AI",
    "comp-gas": "Physical & Earth Sciences",
    "cond-mat": "Physical & Earth Sciences",
    "cs": "Computer Science & AI",
    "dg-ga": "Mathematics & Statistics",
    "econ": "Business, Finance & Economics",
    "eess": "Electronics & Engineering",
    "funct-an": "Mathematics & Statistics",
    "gr-qc": "Physical & Earth Sciences",
    "hep-ex": "Physical & Earth Sciences",
    "hep-lat": "Physical & Earth Sciences",
    "hep-ph": "Physical & Earth Sciences",
    "hep-th": "Physical & Earth Sciences",
    "math": "Mathematics & Statistics",
    "math-ph": "Mathematics & Statistics",
    "mtrl-th": "Electronics & Engineering",
    "nlin": "Mathematics & Statistics",
    "nucl-ex": "Physical & Earth Sciences",
    "nucl-th": "Physical & Earth Sciences",
    "patt-sol": "Mathematics & Statistics",
    "physics": "Physical & Earth Sciences",
    "plasm-ph": "Physical & Earth Sciences",
    "q-alg": "Mathematics & Statistics",
    "q-bio": "Biomedical & Health Sciences",
    "q-fin": "Business, Finance & Economics",
    "quant-ph": "Physical & Earth Sciences",
    "solv-int": "Mathematics & Statistics",
    "stat": "Mathematics & Statistics",
    "supr-con": "Electronics & Engineering",
}

# Remap social_map to new categories
social_map = {
    "LanguageTechnology": "Computer Science & AI",
    "Python": "Computer Science & AI",
    "investing": "Business, Finance & Economics",
    "pytorch": "Computer Science & AI",
}

import os
import pandas as pd

# Assert all mapping values are in the master topic list
for _map, _name in zip([news_map, academic_map, social_map], ["news_map", "academic_map", "social_map"]):
    invalid = [v for v in _map.values() if v not in topics]
    assert not invalid, f"{_name} contains invalid topics: {invalid}"

if __name__ == "__main__":
    # Load combined.csv
    raw_dir = os.path.join( 'data', 'raw')
    combined_path = os.path.join(raw_dir, 'combined.csv')
    df = pd.read_csv(combined_path)

    # Apply mapping based on domain
    def map_target(row):
        if row['domain'] == 'news':
            return news_map.get(row['target'], None)
        elif row['domain'] == 'academic':
            return academic_map.get(row['target'], None)
        elif row['domain'] == 'social':
            return social_map.get(row['target'], None)
        else:
            return None

    df['target'] = df.apply(map_target, axis=1)

    # Save to combined_mapped.csv
    mapped_path = os.path.join(raw_dir, 'combined_mapped.csv')
    df.to_csv(mapped_path, index=False)
    print(f"Mapped combined dataset saved to {mapped_path}")
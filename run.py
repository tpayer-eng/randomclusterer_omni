import argparse, json, os
import numpy as np, pandas as pd, anndata as ad

p = argparse.ArgumentParser()
p.add_argument("--output_dir", "-o", required=True)
p.add_argument("--name", "-n", required=True)
p.add_argument("--data.h5ad", dest="h5ad", required=True)
p.add_argument("--data.clusters_truth_num", dest="k_path", required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--d_cluster", type=int, default=0)
a, _ = p.parse_known_args()

cells = ad.read_h5ad(a.h5ad).obs_names
k = int(open(a.k_path).read()) + a.d_cluster
labels = np.random.default_rng(a.seed).integers(1, k + 1, size=len(cells))

out = lambda ext: os.path.join(a.output_dir, f"{a.name}.{ext}")
pd.DataFrame({"cell_id": cells, "louvain": labels, "leiden": labels}) \
  .to_csv(out("clusters.tsv"), sep="\t", index=False)


json.dump({}, open(out("timings.json"), "w"))
json.dump({}, open(out("clustering_info.json"), "w"))
open(out("hvgs.tsv"), "w").write("dummy\n")
pd.DataFrame({"PC1": np.zeros(len(cells))}, index=cells).to_csv(out("pca.tsv"), sep="\t", index_label="cell_id")

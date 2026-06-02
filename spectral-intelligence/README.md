# Spectral Intelligence

Spectral graph theory analysis for Meilisearch HNSW vector indexes.

## What It Does

You've got 2M vectors sitting in an HNSW index. You configured 8 filters. But what's *actually* in there?

```
$ spectral-intelligence analyze --index vectors.json --clusters 0

Loaded adjacency: 2,000,000 vectors
Detected/requested clusters: 47
Fiedler value range: [-0.0082, 0.0091]
Cheeger constant (conductance estimate): 0.0312

Cluster breakdown:
  Cluster   0: 340,221 vectors ( 17.0%)  ← tech documentation
  Cluster   1: 298,405 vectors ( 14.9%)
  Cluster   2: 201,337 vectors ( 10.1%)
  ...
  Cluster  46:    892 vectors (  0.0%)

No labels provided — skipping quality score.
```

That top cluster with 340K vectors? They all look like tech documentation. Did you mean to index your entire wiki twice? Probably.

## How It Works

Spectral graph theory treats your HNSW graph as a mathematical object and asks: *what does the shape of this graph tell us about the data?*

**The Laplacian.** We compute `L = D - A` (degree matrix minus adjacency matrix). This encodes the graph's connectivity structure.

**Fiedler vector.** The eigenvector of the second-smallest eigenvalue. Nodes with similar Fiedler values are well-connected — they belong to the same region of the graph.

**Cheeger constant.** A measure of graph conductance — how hard is it to cut the graph into pieces? Low values mean bottlenecks. If your Cheeger constant is near zero, some clusters are barely connected to the rest.

**Spectral clustering.** We take the top eigenvectors and run k-means in that space. This finds clusters that respect the actual graph topology, not just raw distances.

**Drift detection.** Save a spectral snapshot. Next time you update the index, compare. If the eigenvalue signature shifted significantly, your embedding space changed — maybe you swapped models, maybe the data distribution shifted.

## Installation

```bash
cd spectral-intelligence
cargo build --release
```

## Usage

### Analyze an index

Export your HNSW adjacency as JSON (array of neighbor arrays) or use the Meilisearch dump format:

```bash
spectral-intelligence analyze \
  --index /path/to/adjacency.json \
  --output snapshot.json
```

### With ground-truth labels

Got labeled data? Check if your embeddings actually cluster by label:

```bash
spectral-intelligence analyze \
  --index adjacency.json \
  --labels labels.json \
  --output snapshot.json
```

Labels file formats:

```json
// Array of integer labels (one per vector)
[0, 0, 1, 1, 2, 0, ...]
```

```json
// Object mapping vector index to label string
{"0": "documentation", "1": "documentation", "2": "code", ...}
```

The quality score is the Adjusted Rand Index (ARI): 1.0 = perfect match, 0.0 = random, negative = worse than random.

### Drift detection across index updates

```bash
# After first index build
spectral-intelligence analyze --index v1.json --output v1-snapshot.json

# After update
spectral-intelligence analyze \
  --index v2.json \
  --baseline v1-snapshot.json \
  --output v2-snapshot.json

# Or compare directly
spectral-intelligence drift \
  --before v1-snapshot.json \
  --after v2-snapshot.json \
  --threshold 0.15
```

### Generate test data

```bash
spectral-intelligence generate --n 1000 --clusters 5 --output test-adj.json
```

## The Math (For the Curious)

1. **Adjacency matrix A**: `A[i][j] = 1` if vectors i,j are neighbors in HNSW
2. **Degree matrix D**: diagonal, `D[i][i]` = number of neighbors of i
3. **Graph Laplacian L**: `L = D - A`
4. **Eigen-decomposition**: `L·v = λ·v`, sort eigenvalues ascending
5. **Fiedler value**: `λ₁` (second smallest) — near zero means nearly disconnected
6. **Eigengap heuristic**: largest gap between consecutive eigenvalues suggests natural cluster count
7. **Cheeger constant**: estimated by sweeping the Fiedler vector to find the minimum-conductance cut

## When This Is Useful

- **Index health checks**: "Is my vector index well-structured or a mess?"
- **Filter design**: "I have 47 natural clusters but only 8 filters — am I missing important segments?"
- **Model validation**: "I switched from text-embedding-ada-002 to a new model. Did my cluster structure change?"
- **Data quality**: "Why is retrieval bad for these queries? Oh, half my index is in one cluster."
- **Monitoring**: "Track spectral drift over time to catch data distribution shifts"

## Dependencies

- `nalgebra` — linear algebra (eigen decomposition)
- `rayon` — parallel k-means
- `serde_json` — I/O

## License

MIT

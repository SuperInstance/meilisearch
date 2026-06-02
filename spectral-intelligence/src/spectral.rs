//! Core spectral graph theory operations for HNSW index analysis.

use anyhow::{Context, Result};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A spectral snapshot for drift comparison across index updates.
#[derive(Debug, Serialize, Deserialize)]
pub struct SpectralSnapshot {
    pub vector_count: usize,
    pub cluster_count: usize,
    pub eigenvalues: Vec<f64>,
    pub cheeger_constant: f64,
    pub cluster_sizes: Vec<usize>,
    pub quality_score: Option<f64>,
    pub drift_score: Option<f64>,
}

/// Adjacency list representation: `adj[node_id] = [neighbor_ids]`.
pub type AdjacencyList = Vec<Vec<usize>>;

/// Load an adjacency list from a JSON file.
///
/// Expects either:
/// - A JSON array of arrays: `[[1,2],[0,2],[0,1],...]`
/// - A JSON object with a `"neighbors"` key containing the above
/// - A newline-delimited text file where each line is `node_id neighbor1 neighbor2 ...`
pub fn load_adjacency(path: &Path) -> Result<AdjacencyList> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read adjacency file: {}", path.display()))?;

    // Try JSON first
    if let Ok(parsed) = serde_json::from_str::<AdjacencyList>(&content) {
        return Ok(parsed);
    }

    // Try JSON object with "neighbors" key
    if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&content) {
        if let Some(neighbors) = obj.get("neighbors") {
            if let Ok(adj) = serde_json::from_value::<AdjacencyList>(neighbors.clone()) {
                return Ok(adj);
            }
        }
    }

    // Fall back to text format
    let mut adj: AdjacencyList = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let neighbors: Vec<usize> = parts[1..]
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        // Ensure the adjacency list is large enough
        let node_id: usize = parts[0].parse().unwrap_or(adj.len());
        while adj.len() <= node_id {
            adj.push(Vec::new());
        }
        adj[node_id] = neighbors;
    }

    if adj.is_empty() {
        anyhow::bail!("Could not parse adjacency from {}", path.display());
    }

    Ok(adj)
}

/// Load ground-truth labels from a JSON file.
///
/// Expects `{"0": "label_a", "1": "label_b", ...}` or `[0, 1, 0, 2, ...]`.
pub fn load_labels(path: &Path) -> Result<HashMap<usize, String>> {
    let content = std::fs::read_to_string(path)?;
    let mut labels = HashMap::new();

    // Try array of integer labels
    if let Ok(arr) = serde_json::from_str::<Vec<usize>>(&content) {
        for (i, label) in arr.iter().enumerate() {
            labels.insert(i, label.to_string());
        }
        return Ok(labels);
    }

    // Try object mapping
    if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content) {
        for (k, v) in map {
            if let Ok(id) = k.parse::<usize>() {
                labels.insert(id, v);
            }
        }
        return Ok(labels);
    }

    anyhow::bail!("Could not parse labels from {}", path.display());
}

/// Build the sparse adjacency matrix (as dense for eigen computation) and degree matrix.
///
/// Returns `(A, D)` where `A[i][j] = 1` if edge exists, and `D` is diagonal with degrees.
pub fn build_matrices(adj: &AdjacencyList, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut a = DMatrix::<f64>::zeros(n, n);
    let mut degrees = vec![0usize; n];

    for (i, neighbors) in adj.iter().enumerate() {
        for &j in neighbors {
            if j < n {
                a[(i, j)] = 1.0;
                a[(j, i)] = 1.0;
                degrees[i] += 1;
            }
        }
    }

    // Make symmetric (in case of directed edges in input)
    for i in 0..n {
        for j in (i + 1)..n {
            if a[(i, j)] != a[(j, i)] {
                let val = a[(i, j)].max(a[(j, i)]);
                a[(i, j)] = val;
                a[(j, i)] = val;
            }
        }
    }

    // Recompute degrees from symmetric matrix
    for i in 0..n {
        degrees[i] = adj[i].len();
    }

    let mut d = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = degrees[i] as f64;
    }

    (a, d)
}

/// Compute the graph Laplacian: `L = D - A`.
pub fn laplacian(d: &DMatrix<f64>, a: &DMatrix<f64>) -> DMatrix<f64> {
    d - a
}

/// Compute the bottom `k` eigenpairs of the Laplacian using power iteration + deflation.
///
/// For large matrices, this uses an iterative approach. Returns eigenvalues sorted ascending
/// and their corresponding eigenvectors as columns of a matrix.
pub fn bottom_eigenpairs(l: &DMatrix<f64>, k: usize) -> Result<(DVector<f64>, DMatrix<f64>)> {
    let n = l.nrows();
    let actual_k = k.min(n);

    // Use the symmetric eigen decomposition from nalgebra if matrix is small enough
    if n <= 512 {
        return small_eigen_decomp(l, actual_k);
    }

    // For larger matrices, use iterative method
    iterative_eigenpairs(l, actual_k)
}

/// Full eigen decomposition for small matrices (≤512).
fn small_eigen_decomp(l: &DMatrix<f64>, k: usize) -> Result<(DVector<f64>, DMatrix<f64>)> {
    let n = l.nrows();
    let decomposed = l.clone().symmetric_eigen();
    let eigenvalues = decomposed.eigenvalues;
    let eigenvectors = decomposed.eigenvectors;

    // Sort by eigenvalue (ascending)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    let mut sorted_values = DVector::zeros(k);
    let mut sorted_vectors = DMatrix::zeros(n, k);

    for (col, &idx) in indices.iter().take(k).enumerate() {
        sorted_values[col] = eigenvalues[idx];
        sorted_vectors.set_column(col, &eigenvectors.column(idx));
    }

    Ok((sorted_values, sorted_vectors))
}

/// Iterative eigenpair computation for larger matrices using inverse power iteration.
fn iterative_eigenpairs(l: &DMatrix<f64>, k: usize) -> Result<(DVector<f64>, DMatrix<f64>)> {
    let n = l.nrows();

    // Add small regularization to make L invertible
    let l_reg = l + DMatrix::from_diagonal_element(n, n, 1e-10);

    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = DMatrix::zeros(n, k);
    let mut deflated = l_reg.clone();

    for i in 0..k {
        let (val, vec) = power_iteration(&deflated, 200, 1e-8)?;
        eigenvalues.push(val);

        // Deflate
        let outer = &vec * &vec.transpose();
        deflated -= &(outer * val);

        eigenvectors.set_column(i, &vec);
    }

    // Subtract regularization from eigenvalues
    let eigenvalues: DVector<f64> = DVector::from_vec(eigenvalues.into_iter().map(|v| v - 1e-10).collect());

    Ok((eigenvalues, eigenvectors))
}

/// Power iteration to find the smallest eigenvalue/eigenvector.
fn power_iteration(m: &DMatrix<f64>, max_iter: usize, tol: f64) -> Result<(f64, DVector<f64>)> {
    let n = m.nrows();
    let mut v = DVector::from_fn(n, |_, _| {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen::<f64>() - 0.5
    });
    let _ = v.normalize_mut();

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        let mv = m * &v;
        let new_eigenvalue = v.dot(&mv);
        let norm = mv.norm();
        if norm < 1e-15 {
            break;
        }
        v = mv / norm;

        if (new_eigenvalue - eigenvalue).abs() < tol {
            eigenvalue = new_eigenvalue;
            break;
        }
        eigenvalue = new_eigenvalue;
    }

    Ok((eigenvalue, v))
}

/// Use the eigengap heuristic to determine the natural number of clusters.
///
/// Looks for the largest relative gap between consecutive eigenvalues.
pub fn eigengap_heuristic(eigenvalues: &DVector<f64>) -> usize {
    let n = eigenvalues.len().min(50); // Don't look past 50 eigenvalues
    if n <= 2 {
        return 2;
    }

    let mut best_k = 2;
    let mut best_gap = 0.0f64;

    for i in 1..(n - 1) {
        let gap = eigenvalues[i + 1] - eigenvalues[i];
        let scale = eigenvalues[i].abs().max(1e-10);
        let rel_gap = gap / scale;
        if rel_gap > best_gap {
            best_gap = rel_gap;
            best_k = i + 1;
        }
    }

    best_k.max(2).min(n)
}

/// Estimate the Cheeger constant (graph conductance) using the Fiedler vector.
///
/// Sorts nodes by Fiedler vector values and finds the cut with minimum conductance.
pub fn cheeger_constant(adj: &AdjacencyList, fiedler: &DVector<f64>) -> f64 {
    let n = adj.len();
    if n <= 1 {
        return 0.0;
    }

    // Sort node indices by Fiedler value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| fiedler[a].partial_cmp(&fiedler[b]).unwrap());

    let total_edges: usize = adj.iter().map(|n| n.len()).sum::<usize>() / 2;
    if total_edges == 0 {
        return 0.0;
    }

    let mut min_conductance = 1.0f64;
    let mut cut_edges = 0usize;
    let mut volume_s = 0usize;

    // Track which side of the cut each node is on
    let mut in_s = vec![false; n];

    for (step, &node) in indices.iter().enumerate() {
        in_s[node] = true;
        volume_s += adj[node].len();

        // Count edges crossing the cut
        for &neighbor in &adj[node] {
            if !in_s[neighbor] {
                cut_edges += 1;
            }
        }

        let volume_complement = total_edges * 2 - volume_s;
        let min_volume = volume_s.min(volume_complement) as f64;

        if min_volume > 0.0 && step > 0 && step < n - 1 {
            let conductance = cut_edges as f64 / min_volume;
            min_conductance = min_conductance.min(conductance);
        }
    }

    min_conductance
}

/// Perform spectral clustering using k-means on the top-k eigenvectors.
pub fn spectral_clustering(eigenvectors: &DMatrix<f64>, k: usize) -> Vec<usize> {
    let n = eigenvectors.nrows();
    let actual_k = k.min(n);

    // Extract the k eigenvectors (skip the first which is constant)
    let start = if actual_k < eigenvectors.ncols() { 1 } else { 0 };
    let cols = eigenvectors.columns(start, actual_k.min(eigenvectors.ncols() - start));

    // Normalize rows
    let data: Vec<DVector<f64>> = (0..n)
        .map(|i| {
            let row = cols.row(i).clone_owned().transpose();
            let norm = row.norm();
            if norm > 1e-10 { row / norm } else { row }
        })
        .collect();

    // k-means++ initialization
    let mut centroids = kmeans_plus_plus_init(&data, actual_k);

    // k-means iterations
    let mut assignments = vec![0; n];
    for _ in 0..50 {
        // Assign
        let new_assignments: Vec<usize> = data
            .par_iter()
            .map(|point| {
                let mut best = 0;
                let mut best_dist = f64::INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let dist = (point - centroid).norm_squared();
                    if dist < best_dist {
                        best_dist = dist;
                        best = c;
                    }
                }
                best
            })
            .collect();

        // Check convergence
        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;

        // Update centroids
        let mut counts = vec![0usize; actual_k];
        let mut sums = vec![DVector::zeros(data[0].len()); actual_k];

        for (i, &cluster) in assignments.iter().enumerate() {
            sums[cluster] += &data[i];
            counts[cluster] += 1;
        }

        for c in 0..actual_k {
            if counts[c] > 0 {
                centroids[c] = &sums[c] / counts[c] as f64;
            }
        }
    }

    assignments
}

/// K-means++ initialization.
fn kmeans_plus_plus_init(data: &[DVector<f64>], k: usize) -> Vec<DVector<f64>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = data.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let mut centroids = Vec::with_capacity(k);

    // Pick first centroid randomly
    let first = rng.gen_range(0..n);
    centroids.push(data[first].clone());

    for _ in 1..k {
        let distances: Vec<f64> = data
            .iter()
            .map(|point| {
                centroids
                    .iter()
                    .map(|c| (point - c).norm_squared())
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let total: f64 = distances.iter().sum();
        if total < 1e-15 {
            // All points are the same, pick random
            centroids.push(data[rng.gen_range(0..n)].clone());
            continue;
        }

        let mut r = rng.gen::<f64>() * total;
        let mut chosen = 0;
        for (i, &d) in distances.iter().enumerate() {
            r -= d;
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    centroids
}

/// Compute cluster sizes from assignments.
pub fn cluster_sizes(assignments: &[usize], k: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; k];
    for &c in assignments {
        if c < k {
            sizes[c] += 1;
        }
    }
    sizes
}

/// Compute the Adjusted Rand Index to measure embedding quality against ground-truth labels.
///
/// ARI = 1.0 means perfect match, 0.0 means random, <0 means worse than random.
pub fn embedding_quality_score(
    assignments: &[usize],
    labels: &HashMap<usize, String>,
    _k: usize,
) -> f64 {
    let n = assignments.len();
    if n == 0 {
        return 0.0;
    }

    // Build contingency table
    let mut label_to_int: HashMap<String, usize> = HashMap::new();
    let mut next_label_id = 0;

    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut row_sums: HashMap<usize, usize> = HashMap::new();
    let mut col_sums: HashMap<usize, usize> = HashMap::new();

    for (i, &cluster) in assignments.iter().enumerate() {
        if let Some(label) = labels.get(&i) {
            let label_id = *label_to_int.entry(label.clone()).or_insert_with(|| {
                let id = next_label_id;
                next_label_id += 1;
                id
            });

            *contingency.entry((cluster, label_id)).or_insert(0) += 1;
            *row_sums.entry(cluster).or_insert(0) += 1;
            *col_sums.entry(label_id).or_insert(0) += 1;
        }
    }

    let matched = contingency.values().sum::<usize>() as f64;
    if matched == 0.0 {
        return 0.0;
    }

    // ARI computation
    let sum_comb_c = contingency
        .values()
        .map(|&v| comb2(v))
        .sum::<f64>();
    let sum_comb_a = row_sums.values().map(|&v| comb2(v)).sum::<f64>();
    let sum_comb_b = col_sums.values().map(|&v| comb2(v)).sum::<f64>();

    let n_comb = comb2(matched as usize);
    let expected = sum_comb_a * sum_comb_b / n_comb;
    let max_index = 0.5 * (sum_comb_a + sum_comb_b);

    if max_index == expected {
        return 1.0;
    }

    (sum_comb_c - expected) / (max_index - expected)
}

/// Binomial coefficient C(n, 2).
fn comb2(n: usize) -> f64 {
    (n as f64) * (n as f64 - 1.0) / 2.0
}

/// Compute drift score between a baseline snapshot and current state.
///
/// Combines eigenvalue shift and cluster distribution shift.
pub fn compute_drift(
    baseline: &SpectralSnapshot,
    current_eigenvalues: &DVector<f64>,
    current_cluster_sizes: &[usize],
) -> f64 {
    // Eigenvalue shift: cosine distance between eigenvalue vectors
    let baseline_vec: DVector<f64> = DVector::from_vec(
        baseline.eigenvalues[..baseline.eigenvalues.len().min(current_eigenvalues.len())].to_vec(),
    );
    let current_vec = current_eigenvalues.rows(0, baseline_vec.len()).clone_owned();

    let eigen_drift = cosine_distance(&baseline_vec, &current_vec);

    // Cluster distribution shift: Jensen-Shannon-like divergence
    let n_current: f64 = current_cluster_sizes.iter().sum::<usize>() as f64;
    let n_baseline: f64 = baseline.cluster_sizes.iter().sum::<usize>() as f64;

    let max_clusters = baseline.cluster_sizes.len().max(current_cluster_sizes.len());
    let mut p = vec![0.0f64; max_clusters];
    let mut q = vec![0.0f64; max_clusters];

    for (i, &s) in baseline.cluster_sizes.iter().enumerate() {
        p[i] = s as f64 / n_baseline;
    }
    for (i, &s) in current_cluster_sizes.iter().enumerate() {
        q[i] = s as f64 / n_current;
    }

    let cluster_drift = total_variation_distance(&p, &q);

    // Weighted combination
    0.6 * eigen_drift + 0.4 * cluster_drift
}

/// Cosine distance between two vectors (1 - cosine similarity).
fn cosine_distance(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    let norm_a = a.norm();
    let norm_b = b.norm();
    if norm_a < 1e-15 || norm_b < 1e-15 {
        return 1.0;
    }
    1.0 - a.dot(b) / (norm_a * norm_b)
}

/// Total variation distance between two probability distributions.
fn total_variation_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).abs())
        .sum::<f64>()
        / 2.0
}

/// Load a spectral snapshot from a JSON file.
pub fn load_snapshot(path: &Path) -> Result<SpectralSnapshot> {
    let content = std::fs::read_to_string(path)?;
    let snapshot = serde_json::from_str(&content)?;
    Ok(snapshot)
}

/// Generate a synthetic clustered adjacency list for testing.
pub fn generate_synthetic(n: usize, clusters: usize) -> AdjacencyList {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let cluster_size = n / clusters;

    let mut adj = vec![Vec::new(); n];
    let mut cluster_assignments = Vec::with_capacity(n);

    for i in 0..n {
        cluster_assignments.push(i / cluster_size);
    }
    // Handle remainder
    for i in (clusters * cluster_size)..n {
        cluster_assignments[i] = clusters - 1;
    }

    // Intra-cluster edges (dense)
    for c in 0..clusters {
        let members: Vec<usize> = (0..n).filter(|&i| cluster_assignments[i] == c).collect();
        for (_idx_a, &a) in members.iter().enumerate() {
            let k = (3 + rng.gen_range(0..5)).min(members.len() - 1);
            for _ in 0..k {
                let b = members[rng.gen_range(0..members.len())];
                if a != b && !adj[a].contains(&b) {
                    adj[a].push(b);
                    adj[b].push(a);
                }
            }
        }
    }

    // Inter-cluster edges (sparse bridges)
    for _ in 0..(n / 20) {
        let a = rng.gen_range(0..n);
        let b = rng.gen_range(0..n);
        if a != b && cluster_assignments[a] != cluster_assignments[b] && !adj[a].contains(&b) {
            adj[a].push(b);
            adj[b].push(a);
        }
    }

    adj
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use tempfile::NamedTempFile;

    #[test]
    fn test_laplacian_basic() {
        // Triangle graph: fully connected 3 nodes
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let (a, d) = build_matrices(&adj, 3);
        let l = laplacian(&d, &a);

        // L for triangle: each row sums to 0
        assert_relative_eq!(l[(0, 0)], 2.0);
        assert_relative_eq!(l[(0, 1)], -1.0);
        assert_relative_eq!(l[(0, 2)], -1.0);

        // Row sum should be 0
        for i in 0..3 {
            let row_sum: f64 = (0..3).map(|j| l[(i, j)]).sum();
            assert_relative_eq!(row_sum, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fiedler_value_disconnected() {
        // Two disconnected triangles
        let adj = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![4, 5],
            vec![3, 5],
            vec![3, 4],
        ];
        let (a, d) = build_matrices(&adj, 6);
        let l = laplacian(&d, &a);
        let (eigenvalues, _) = bottom_eigenpairs(&l, 3).unwrap();

        // Second eigenvalue should be 0 for disconnected graph
        assert_relative_eq!(eigenvalues[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(eigenvalues[1], 0.0, epsilon = 1e-6);
        // Third eigenvalue should be > 0
        assert!(eigenvalues[2] > 0.0);
    }

    #[test]
    fn test_eigengap_heuristic() {
        // Eigenvalues with clear gap after 3rd
        let eigenvalues = DVector::from_vec(vec![0.0, 0.01, 0.02, 2.5, 3.0, 3.5]);
        let k = eigengap_heuristic(&eigenvalues);
        assert_eq!(k, 3);
    }

    #[test]
    fn test_cheeger_constant_connected() {
        // Complete graph K4: Cheeger constant should be high
        let adj = vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ];
        let (a, d) = build_matrices(&adj, 4);
        let l = laplacian(&d, &a);
        let (_, eigenvectors) = bottom_eigenpairs(&l, 2).unwrap();
        let fiedler = eigenvectors.column(1).clone_owned();

        let h = cheeger_constant(&adj, &fiedler);
        // K4 has Cheeger constant 3/4 = 0.75
        assert!(h > 0.3, "Cheeger constant for K4 should be > 0.3, got {}", h);
    }

    #[test]
    fn test_spectral_clustering_two_clusters() {
        // Two well-separated cliques with a single bridge
        let adj = vec![
            // Cluster 1: nodes 0-3
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2, 4], // bridge to node 4
            // Cluster 2: nodes 4-7
            vec![3, 5, 6, 7],
            vec![4, 6, 7],
            vec![4, 5, 7],
            vec![4, 5, 6],
        ];
        let (a, d) = build_matrices(&adj, 8);
        let l = laplacian(&d, &a);
        let (_, eigenvectors) = bottom_eigenpairs(&l, 3).unwrap();

        let assignments = spectral_clustering(&eigenvectors, 2);

        // Nodes 0-3 should be in one cluster, 4-7 in another
        let c0 = assignments[0];
        for i in 1..4 {
            assert_eq!(assignments[i], c0, "Node {} should be in same cluster as 0", i);
        }
        let c4 = assignments[4];
        assert_ne!(c0, c4, "Clusters should be different");
        for i in 5..8 {
            assert_eq!(assignments[i], c4, "Node {} should be in same cluster as 4", i);
        }
    }

    #[test]
    fn test_embedding_quality_perfect() {
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let labels: HashMap<usize, String> = [
            (0, "a".into()),
            (1, "a".into()),
            (2, "a".into()),
            (3, "b".into()),
            (4, "b".into()),
            (5, "b".into()),
        ]
        .into_iter()
        .collect();

        let score = embedding_quality_score(&assignments, &labels, 2);
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_embedding_quality_random() {
        let assignments = vec![0, 1, 0, 1, 0, 1];
        let labels: HashMap<usize, String> = [
            (0, "a".into()),
            (1, "a".into()),
            (2, "a".into()),
            (3, "b".into()),
            (4, "b".into()),
            (5, "b".into()),
        ]
        .into_iter()
        .collect();

        let score = embedding_quality_score(&assignments, &labels, 2);
        // Should be low (close to 0 or negative)
        assert!(score < 0.3, "Random assignments should score low, got {}", score);
    }

    #[test]
    fn test_generate_and_load_adjacency() {
        let adj = generate_synthetic(100, 4);
        assert_eq!(adj.len(), 100);

        // Serialize and reload
        let json = serde_json::to_string(&adj).unwrap();
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), &json).unwrap();

        let loaded = load_adjacency(file.path()).unwrap();
        assert_eq!(loaded.len(), 100);
        assert_eq!(loaded[0], adj[0]);
    }

    #[test]
    fn test_drift_detection() {
        let baseline = SpectralSnapshot {
            vector_count: 1000,
            cluster_count: 3,
            eigenvalues: vec![0.0, 0.1, 0.2, 1.5, 2.0],
            cheeger_constant: 0.3,
            cluster_sizes: vec![300, 400, 300],
            quality_score: None,
            drift_score: None,
        };

        // Same distribution — low drift
        let current_eigen = DVector::from_vec(vec![0.0, 0.11, 0.21, 1.52, 2.01]);
        let current_sizes = vec![305, 395, 300];
        let drift = compute_drift(&baseline, &current_eigen, &current_sizes);
        assert!(drift < 0.1, "Similar distributions should have low drift, got {}", drift);

        // Very different distribution — high drift
        let current_eigen = DVector::from_vec(vec![0.0, 0.5, 1.2, 3.0, 5.0]);
        let current_sizes = vec![800, 100, 100];
        let drift = compute_drift(&baseline, &current_eigen, &current_sizes);
        assert!(drift > 0.1, "Different distributions should have high drift, got {}", drift);
    }

    #[test]
    fn test_load_labels_array() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "[0, 0, 1, 1, 2]").unwrap();
        let labels = load_labels(file.path()).unwrap();
        assert_eq!(labels[&0], "0");
        assert_eq!(labels[&4], "2");
    }

    #[test]
    fn test_load_labels_object() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), r#"{"0":"docs","1":"code","2":"docs"}"#).unwrap();
        let labels = load_labels(file.path()).unwrap();
        assert_eq!(labels[&0], "docs");
        assert_eq!(labels[&1], "code");
    }

    #[test]
    fn test_cluster_sizes() {
        let assignments = vec![0, 1, 0, 2, 1, 0];
        let sizes = cluster_sizes(&assignments, 3);
        assert_eq!(sizes, vec![3, 2, 1]);
    }

    #[test]
    fn test_cosine_distance() {
        let a = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let b = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        assert_relative_eq!(cosine_distance(&a, &b), 0.0, epsilon = 1e-10);

        let c = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        assert_relative_eq!(cosine_distance(&a, &c), 1.0, epsilon = 1e-10);
    }
}

//! Spectral Search Intelligence — spectral graph analysis for Meilisearch HNSW indexes.
//!
//! This module provides tools to analyze the structure of HNSW vector indexes using
//! spectral graph theory. It can detect embedding quality issues, identify cluster
//! structure, estimate graph conductance, and track drift across index updates.

mod spectral;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "spectral-intelligence", version, about = "Spectral graph analysis for Meilisearch HNSW vector indexes")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze an HNSW index and print spectral diagnostics
    Analyze {
        /// Path to the Meilisearch vector index file (.mdb or exported adjacency)
        #[arg(short, long)]
        index: PathBuf,

        /// Optional path to a JSON file with ground-truth labels for quality scoring
        #[arg(short, long)]
        labels: Option<PathBuf>,

        /// Number of clusters to detect (0 = auto via eigengap)
        #[arg(short, long, default_value = "0")]
        clusters: usize,

        /// Previous spectral snapshot for drift detection
        #[arg(long)]
        baseline: Option<PathBuf>,

        /// Output path for spectral snapshot JSON
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Compare two spectral snapshots and report drift
    Drift {
        /// Baseline spectral snapshot
        #[arg(long)]
        before: PathBuf,

        /// Current spectral snapshot
        #[arg(long)]
        after: PathBuf,

        /// Drift sensitivity threshold (0.0–1.0)
        #[arg(short, default_value = "0.15")]
        threshold: f64,
    },

    /// Generate a synthetic HNSW-like adjacency for testing
    Generate {
        /// Number of vectors
        #[arg(short, long, default_value = "1000")]
        n: usize,

        /// Number of clusters in synthetic data
        #[arg(short, long, default_value = "5")]
        clusters: usize,

        /// Output path for adjacency list
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze {
            index,
            labels,
            clusters,
            baseline,
            output,
        } => {
            let adjacency = spectral::load_adjacency(&index)?;
            let n = adjacency.len();
            println!("Loaded adjacency: {} vectors", n);

            // Build adjacency matrix
            let (adj_matrix, degree_matrix) = spectral::build_matrices(&adjacency, n);

            // Laplacian
            let laplacian = spectral::laplacian(&degree_matrix, &adj_matrix);

            // Eigen decomposition (partial — bottom eigenvalues)
            let eigen_count = (clusters + 2).min(n).max(10);
            let (eigenvalues, eigenvectors) = spectral::bottom_eigenpairs(&laplacian, eigen_count)?;

            // Determine cluster count
            let k = if clusters > 0 {
                clusters
            } else {
                spectral::eigengap_heuristic(&eigenvalues)
            };
            println!("Detected/requested clusters: {}", k);

            // Fiedler vector (eigenvector of 2nd smallest eigenvalue)
            let fiedler = eigenvectors.column(1).clone_owned();
            println!(
                "Fiedler value range: [{:.4}, {:.4}]",
                fiedler.iter().cloned().fold(f64::INFINITY, f64::min),
                fiedler.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );

            // Cheeger constant
            let cheeger = spectral::cheeger_constant(&adjacency, &fiedler);
            println!("Cheeger constant (conductance estimate): {:.6}", cheeger);

            // Spectral clustering
            let assignments = spectral::spectral_clustering(&eigenvectors, k);
            let cluster_sizes = spectral::cluster_sizes(&assignments, k);

            // Print cluster summary
            let mut sorted_clusters: Vec<_> = cluster_sizes.iter().enumerate().collect();
            sorted_clusters.sort_by(|a, b| b.1.cmp(a.1));
            println!("\nCluster breakdown:");
            for (id, size) in &sorted_clusters {
                let pct = (**size as f64 / n as f64) * 100.0;
                println!("  Cluster {:>3}: {:>6} vectors ({:>5.1}%)", id, size, pct);
            }

            // Embedding quality score
            let quality = if let Some(labels_path) = labels {
                let label_map = spectral::load_labels(&labels_path)?;
                let score = spectral::embedding_quality_score(&assignments, &label_map, k);
                println!("\nEmbedding quality (ARI): {:.4}", score);
                Some(score)
            } else {
                println!("\nNo labels provided — skipping quality score.");
                None
            };

            // Drift detection
            let drift = if let Some(baseline_path) = baseline {
                let prev = spectral::load_snapshot(&baseline_path)?;
                let drift_score = spectral::compute_drift(&prev, &eigenvalues, &cluster_sizes);
                println!(
                    "\nDrift from baseline: {:.4} {}",
                    drift_score,
                    if drift_score > 0.15 {
                        "⚠ SIGNIFICANT"
                    } else {
                        "✓ stable"
                    }
                );
                Some(drift_score)
            } else {
                None
            };

            // Save snapshot
            if let Some(out_path) = output {
                let snapshot = spectral::SpectralSnapshot {
                    vector_count: n,
                    cluster_count: k,
                    eigenvalues: eigenvalues.iter().cloned().take(k + 2).collect(),
                    cheeger_constant: cheeger,
                    cluster_sizes: cluster_sizes.clone(),
                    quality_score: quality,
                    drift_score: drift,
                };
                let json = serde_json::to_string_pretty(&snapshot)?;
                std::fs::write(&out_path, json)?;
                println!("\nSnapshot saved to {}", out_path.display());
            }
        }

        Commands::Drift {
            before,
            after,
            threshold,
        } => {
            let snap_before = spectral::load_snapshot(&before)?;
            let snap_after = spectral::load_snapshot(&after)?;

            let drift = spectral::compute_drift(
                &snap_before,
                &nalgebra::DVector::from_vec(snap_after.eigenvalues.clone()),
                &snap_after.cluster_sizes,
            );
            println!("Drift score: {:.4}", drift);
            println!(
                "Threshold:   {:.4} ({})",
                threshold,
                if drift > threshold {
                    "EXCEEDED ⚠"
                } else {
                    "within bounds ✓"
                }
            );
        }

        Commands::Generate {
            n,
            clusters,
            output,
        } => {
            let adj = spectral::generate_synthetic(n, clusters);
            let json = serde_json::to_string(&adj)?;
            std::fs::write(&output, json)?;
            println!(
                "Generated synthetic adjacency: {} vectors, {} clusters → {}",
                n,
                clusters,
                output.display()
            );
        }
    }

    Ok(())
}

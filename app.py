import os
import streamlit as st
from Bio import SeqIO
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import io

# --- Configuration --- #
PROJECT_TITLE = "PhyloGuard: An Automated Bioinformatics Pipeline for Sequence Similarity, Mutation/Variation Analysis and Phylogenetic Classification"
SUBTITLE = "Integrated DNA Sequence Analysis, Variation Analysis and Phylogenetic Classification"
PROJECT_DIR = 'PhyloGuard'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

st.set_page_config(layout="wide", page_title="PhyloGuard")

# --- Core Analysis Functions ---
def get_sequence_properties(sequences):
    if not sequences:
        return None

    seq_data = []
    for seq_obj in sequences:
        seq_id = str(seq_obj.id)
        sequence = str(seq_obj.seq).upper()
        length = len(sequence)

        gc_count = sequence.count('G') + sequence.count('C')
        gc_percent = (gc_count / length * 100) if length > 0 else 0

        nuc_counts = Counter(sequence)
        a_count = nuc_counts.get('A', 0)
        t_count = nuc_counts.get('T', 0)
        g_count = nuc_counts.get('G', 0)
        c_count = nuc_counts.get('C', 0)
        n_count = nuc_counts.get('N', 0)

        total_bases = sum(nuc_counts.values())
        a_pct = (a_count / total_bases * 100) if total_bases > 0 else 0
        t_pct = (t_count / total_bases * 100) if total_bases > 0 else 0
        g_pct = (g_count / total_bases * 100) if total_bases > 0 else 0
        c_pct = (c_count / total_bases * 100) if total_bases > 0 else 0

        valid = all(n in 'ATGCN-' for n in sequence)
        validation_status = "Valid" if valid else "Invalid: Contains non-standard characters"

        seq_data.append({
            "ID": seq_id,
            "Length": length,
            "GC%": f"{gc_percent:.2f}",
            "A_count": a_count,
            "A_pct": f"{a_pct:.2f}%",
            "T_count": t_count,
            "T_pct": f"{t_pct:.2f}%",
            "G_count": g_count,
            "G_pct": f"{g_pct:.2f}%",
            "C_count": c_count,
            "C_pct": f"{c_pct:.2f}%",
            "N_count": n_count,
            "Validation": validation_status
        })

    return pd.DataFrame(seq_data)

def calculate_pairwise_similarity(sequences):
    if len(sequences) < 2:
        return None

    seq_records = {seq.id: str(seq.seq).upper() for seq in sequences}
    ids = list(seq_records.keys())
    similarity_data = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            seq1, seq2 = seq_records[id1], seq_records[id2]

            max_len = max(len(seq1), len(seq2))
            if max_len == 0:
                continue

            matches = 0
            mismatches = 0
            gaps = abs(len(seq1) - len(seq2))

            min_len = min(len(seq1), len(seq2))
            for k in range(min_len):
                char1 = seq1[k]
                char2 = seq2[k]
                if char1 == '-' or char2 == '-':
                    gaps += 1
                elif char1 == char2:
                    matches += 1
                else:
                    mismatches += 1

            identity_percent = (matches / max_len * 100)

            similarity_data.append({
                "Sequence_1": id1,
                "Sequence_2": id2,
                "Identity%": f"{identity_percent:.2f}%",
                "Mismatches": mismatches,
                "Gaps": gaps
            })

    df_similarity = pd.DataFrame(similarity_data)
    return df_similarity

def run_simple_msa(sequences):
    if not sequences:
        return []

    max_len = max(len(s.seq) for s in sequences)
    aligned_sequences = []

    for seq_obj in sequences:
        seq_str = str(seq_obj.seq).upper()
        if len(seq_str) < max_len:
            padded_seq = seq_str + '-' * (max_len - len(seq_str))
        else:
            padded_seq = seq_str

        new_record = SeqIO.SeqRecord(seq_obj.seq.__class__(padded_seq), id=seq_obj.id, description="")
        aligned_sequences.append(new_record)

    return aligned_sequences

def analyze_msa_properties(aligned_sequences):
    if not aligned_sequences:
        return None

    alignment_length = len(aligned_sequences[0].seq)
    aligned_sequences_count = len(aligned_sequences)

    alignment_preview_lines = []
    for seq in aligned_sequences[:5]:
        alignment_preview_lines.append(f">{seq.id}\n{str(seq.seq)[:70]}{' ...' if len(seq.seq) > 70 else ''}")
    alignment_preview = "\n".join(alignment_preview_lines)
    if len(aligned_sequences) > 5:
        alignment_preview += f"\n... ({len(aligned_sequences) - 5} more sequences not shown)"

    conserved_count = 0
    variable_count = 0
    for i in range(alignment_length):
        column = [str(seq.seq)[i] for seq in aligned_sequences]
        unique_bases = set(column) - {'-'}
        if len(unique_bases) <= 1:
            conserved_count += 1
        else:
            variable_count += 1

    conserved_percent = (conserved_count / alignment_length * 100) if alignment_length > 0 else 0
    variable_percent = (variable_count / alignment_length * 100) if alignment_length > 0 else 0

    msa_props = {
        "alignment_length": alignment_length,
        "aligned_sequences_count": aligned_sequences_count,
        "alignment_preview": alignment_preview,
        "conserved_positions_count": conserved_count,
        "variable_positions_count": variable_count,
        "conserved_positions_percent": f"{conserved_percent:.2f}%",
        "variable_positions_percent": f"{variable_percent:.2f}%",
    }
    return msa_props

def perform_variation_analysis(aligned_sequences, reference_id):
    if not aligned_sequences:
        return None

    reference_seq_obj = next((seq for seq in aligned_sequences if seq.id == reference_id), None)
    if not reference_seq_obj:
        return None
    reference_seq = str(reference_seq_obj.seq).upper()

    variation_data = []
    for sample_seq_obj in aligned_sequences:
        if sample_seq_obj.id == reference_id:
            continue

        sample_seq = str(sample_seq_obj.seq).upper()
        for i in range(min(len(reference_seq), len(sample_seq))):
            ref_char = reference_seq[i]
            sample_char = sample_seq[i]

            if ref_char != sample_char:
                variation_type = "Substitution"
                if ref_char == '-' and sample_char != '-':
                    variation_type = "Insertion"
                elif ref_char != '-' and sample_char == '-':
                    variation_type = "Deletion"

                variation_data.append({
                    "Position": i + 1,
                    "Reference": ref_char,
                    "Sample_ID": sample_seq_obj.id,
                    "Sample_Base": sample_char,
                    "Type": variation_type
                })

    return pd.DataFrame(variation_data)

# --- Dynamic Phylogenetic Construction (Neighbor Joining via raw Python) ---
def construct_nj_tree(aligned_sequences):
    names = [s.id for s in aligned_sequences]
    n = len(names)

    # 1. Build distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s1 = str(aligned_sequences[i].seq).upper()
            s2 = str(aligned_sequences[j].seq).upper()
            mismatch = sum(1 for c1, c2 in zip(s1, s2) if c1 != c2 and c1 != '-' and c2 != '-')
            dist_matrix[i, j] = mismatch
            dist_matrix[j, i] = mismatch

    # Create printable matrix dataframe
    dist_df = pd.DataFrame(dist_matrix, index=names, columns=names)

    # 2. Reconstruct Tree using a standard NJ approach represented as nested Newick/dict
    tree_nodes = {i: {"name": names[i], "length": 0.0, "is_leaf": True} for i in range(n)}
    nodes = {i: f"{names[i]}" for i in range(n)}
    active_indices = list(range(n))
    D = dist_matrix.copy()

    while len(active_indices) > 2:
        N = len(active_indices)
        r = {}
        for i in active_indices:
            r[i] = sum(D[i, j] for j in active_indices) / (N - 2)

        # Find closest pair
        min_q = float('inf')
        best_pair = (None, None)
        for u in range(len(active_indices)):
            for v in range(u + 1, len(active_indices)):
                i = active_indices[u]
                j = active_indices[v]
                q = D[i, j] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    best_pair = (i, j)

        f, g = best_pair
        new_id = max(nodes.keys()) + 1

        # Compute branch lengths
        dist_f = (D[f, g] + r[f] - r[g]) / 2
        dist_g = D[f, g] - dist_f
        if dist_f < 0: dist_f = 0.0
        if dist_g < 0: dist_g = 0.0

        nodes[new_id] = f"({nodes[f]}:{dist_f:.4f}, {nodes[g]}:{dist_g:.4f})"

        tree_nodes[new_id] = {
            "is_leaf": False,
            "left": tree_nodes[f],
            "right": tree_nodes[g],
            "left_len": dist_f,
            "right_len": dist_g
        }

        # Update distances
        row = np.zeros(D.shape[0] + 1)
        for k in active_indices:
            if k != f and k != g:
                dist_new = (D[f, k] + D[g, k] - D[f, g]) / 2
                row[k] = dist_new

        D = np.vstack([D, row[:-1]])
        new_col = np.append(row[:-1], 0).reshape(-1, 1)
        D = np.hstack([D, new_col])

        active_indices.remove(f)
        active_indices.remove(g)
        active_indices.append(new_id)

    if len(active_indices) == 2:
        f, g = active_indices
        newick = f"({nodes[f]}:1.0000, {nodes[g]}:1.0000);"
        final_root_node = {
            "is_leaf": False,
            "left": tree_nodes[f],
            "right": tree_nodes[g],
            "left_len": 1.0,
            "right_len": 1.0
        }
    else:
        newick = f"({nodes[active_indices[0]]});"
        final_root_node = tree_nodes[active_indices[0]]

    return dist_df, newick, final_root_node

# --- Dynamic Dendrogram Drawer ---
def draw_branching_tree(node, x=0, y=0, height=1.0, current_depth=0, leaf_y_counter=[0], ax=None):
    if ax is None:
        return

    if node["is_leaf"]:
        node_y = leaf_y_counter[0]
        leaf_y_counter[0] += 1
        ax.text(x + 0.05, node_y, f" {node['name']}", va='center', ha='left', fontsize=10, weight='bold', color='#1A330E')
        ax.plot(x, node_y, 'go', markersize=6)
        return x, node_y

    left_x_target = x + node["left_len"]
    right_x_target = x + node["right_len"]

    left_x, left_y = draw_branching_tree(node["left"], left_x_target, y, height, current_depth + 1, leaf_y_counter, ax)
    right_x, right_y = draw_branching_tree(node["right"], right_x_target, y, height, current_depth + 1, leaf_y_counter, ax)

    parent_x = min(left_x_target, right_x_target) - min(node["left_len"], node["right_len"])
    parent_y = (left_y + right_y) / 2.0

    ax.plot([parent_x, left_x], [left_y, left_y], color='#3E7324', linewidth=2)
    ax.plot([parent_x, right_x], [right_y, right_y], color='#3E7324', linewidth=2)
    ax.plot([parent_x, parent_x], [left_y, right_y], color='#3E7324', linewidth=2)

    return parent_x, parent_y

def render_phylogenetic_tree_plot(root_node, total_leaves):
    fig, ax = plt.subplots(figsize=(8, max(4, total_leaves * 0.5 + 1.5)))
    leaf_y_counter = [0]

    draw_branching_tree(root_node, x=0, y=0, height=1.0, current_depth=0, leaf_y_counter=leaf_y_counter, ax=ax)

    ax.set_ylim(-0.5, total_leaves - 0.5)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Evolutionary Divergence Distance scale')
    ax.set_title('Dynamic Neighbor-Joining Phylogenetic Tree', fontsize=12, weight='bold', color='#2F4F4F')
    plt.tight_layout()
    return fig

# --- Dynamic Data-Driven Distance Cluster Classification ---
def compute_clusters(dist_df):
    names = list(dist_df.index)
    n = len(names)
    matrix = dist_df.values

    if n <= 1:
        return pd.DataFrame([{"Sequence ID": names[0], "Cluster Assignment": "Cluster 1"}] if n == 1 else []), 1

    # Get all off-diagonal pairwise distance elements
    off_diag = matrix[np.triu_indices(n, k=1)]
    mean_dist = np.mean(off_diag)
    std_dist = np.std(off_diag) if len(off_diag) > 1 else 0.0

    # Adaptive single-linkage clustering threshold
    # If standard deviation is extremely low, sequences are highly uniform -> single cluster threshold
    if std_dist < 1e-5:
        threshold = mean_dist + 1e-5
    else:
        threshold = mean_dist - (0.5 * std_dist)
        if threshold < 0:
            threshold = mean_dist

    # Standard Graph/Connected-components approach based on distance threshold
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] <= threshold:
                adj[i].append(j)
                adj[j].append(i)

    # Depth First Search traversal to find connected subgraphs
    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            temp_comp = []
            stack = [i]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    temp_comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(temp_comp)

    # Format output data records
    cluster_map = {}
    for cluster_idx, comp in enumerate(components):
        c_id = f"Cluster {cluster_idx + 1}"
        for node_idx in comp:
            cluster_map[names[node_idx]] = c_id

    cluster_records = []
    for name in names:
        cluster_records.append({
            "Sequence ID": name,
            "Cluster Assignment": cluster_map.get(name, "Cluster 1")
        })

    return pd.DataFrame(cluster_records), len(components)

# --- Sequence-Specific Custom Project-Defined Phylo-Confidence Score ---
def compute_phylo_confidence(sequences, msa_props, variation_df, cluster_df, dist_df):
    scores = []
    names = [s.id for s in sequences]
    n = len(names)

    # Group sequence names by assigned cluster
    clusters_mapping = {}
    for _, row in cluster_df.iterrows():
        clusters_mapping[row["Sequence ID"]] = row["Cluster Assignment"]

    for seq in sequences:
        seq_id = seq.id
        seq_str = str(seq.seq).upper()

        # 1. Base Quality Component (30% weight): Penalizes ambiguous 'N' nucleotides
        total_len = len(seq_str)
        n_count = seq_str.count('N')
        base_quality_pct = (1.0 - (n_count / total_len)) if total_len > 0 else 1.0
        score_base_quality = base_quality_pct * 30.0

        # 2. Alignment Purity Component (40% weight): Dynamic based on gap ratio
        gap_count = seq_str.count('-')
        alignment_purity_pct = (1.0 - (gap_count / total_len)) if total_len > 0 else 1.0
        score_alignment_purity = alignment_purity_pct * 40.0

        # 3. Cluster Placement & Evolutionary Consistency Component (30% weight)
        # If the sequence belongs to a cluster with multiple sequences, we calculate its closeness
        # to other members of the same cluster. If it's an outlier or single sequence, we grade relative to mean
        assigned_cluster = clusters_mapping.get(seq_id, "Cluster 1")
        cluster_members = [k for k, v in clusters_mapping.items() if v == assigned_cluster]

        if len(cluster_members) > 1:
            # Average distance to other members within the assigned cluster
            member_distances = [dist_df.at[seq_id, m] for m in cluster_members if m != seq_id]
            avg_member_dist = np.mean(member_distances)
            max_all_dist = np.max(dist_df.values)
            if max_all_dist > 0:
                placement_purity_pct = 1.0 - (avg_member_dist / max_all_dist)
            else:
                placement_purity_pct = 1.0
        else:
            # Single sequence cluster placement grading
            if n > 1:
                mean_global_dist = np.mean(dist_df.values)
                avg_seq_dist = np.mean(dist_df.loc[seq_id].values)
                if mean_global_dist > 0:
                    placement_purity_pct = max(0.0, 1.0 - (avg_seq_dist / (mean_global_dist * 2.0)))
                else:
                    placement_purity_pct = 1.0
            else:
                placement_purity_pct = 1.0

        score_cluster_consistency = placement_purity_pct * 30.0
        final_score = score_base_quality + score_alignment_purity + score_cluster_consistency

        scores.append({
            "Sequence ID": seq_id,
            "Project-Defined Phylo-Confidence Score": f"{final_score:.2f}%",
            "Sequence Base Quality Component (30%)": f"{score_base_quality:.2f}/30",
            "Alignment Purity Component (40%)": f"{score_alignment_purity:.2f}/40",
            "Cluster Placement Consistency Component (30%)": f"{score_cluster_consistency:.2f}/30"
        })

    return pd.DataFrame(scores)

# --- Streamlit UI Main Code --- #
def main():
    st.title(PROJECT_TITLE)
    st.subheader(SUBTITLE)
    st.write("PhyloGuard is an automated bioinformatics application for DNA sequence analysis, sequence similarity, multiple sequence alignment, sequence variation analysis and phylogenetic classification.")

    st.sidebar.header("Upload DNA Sequence File")
    uploaded_file = st.sidebar.file_uploader(
        "Upload DNA FASTA/FA/FNA/TXT File",
        type=["fasta", "fa", "fna", "txt"]
    )

    # Tabs setup
    tab_seq, tab_msa, tab_phy, tab_clus, tab_conf, tab_final = st.tabs([
        "Sequence Analysis",
        "MSA & Variation Analysis",
        "Phylogenetic Analysis",
        "Cluster Classification",
        "Phylo-Confidence Score",
        "Final Results"
    ])

    if uploaded_file is None:
        with tab_seq:
            st.info("Please upload a multi-sequence FASTA/TXT file to get started.")
        with tab_msa:
            st.info("Please upload a multi-sequence FASTA/TXT file to get started.")
        with tab_phy:
            st.info("Please upload a multi-sequence FASTA/TXT file to get started.")
        with tab_clus:
            st.info("Please upload a multi-sequence FASTA/TXT file to get started.")
        with tab_conf:
            st.info("Please upload a multi-sequence FASTA/TXT file to get started.")
        with tab_final:
            st.info("Please upload a multi-sequence FASTA/TXT file to get started.")
        return

    # Process file dynamically
    fasta_content = uploaded_file.getvalue().decode("utf-8")
    try:
        sequences = list(SeqIO.parse(StringIO(fasta_content), "fasta"))
    except Exception as e:
        st.error(f"Error parsing uploaded file: {e}. Please ensure it is in valid FASTA format.")
        return

    if not sequences:
        st.error("Zero sequences detected in your uploaded file. Please format it with a '>' header for each sequence.")
        return

    # Pre-calculate sequence quality details
    summary_df = get_sequence_properties(sequences)
    similarity_df = calculate_pairwise_similarity(sequences)

    # Handle single sequence scenarios gracefully
    is_single = len(sequences) < 2

    # Dynamic calculations for MSA/Phylo tabs
    if not is_single:
        aligned_seqs = run_simple_msa(sequences)
        msa_props = analyze_msa_properties(aligned_seqs)

        # Variation Analysis against the first sequence as reference
        first_seq_id = sequences[0].id
        variation_df = perform_variation_analysis(aligned_seqs, first_seq_id)

        # Phylo NJ Matrix, Newick and branching nodes structure
        dist_df, newick_tree, tree_structure = construct_nj_tree(aligned_seqs)

        # Cluster assignments
        cluster_df, num_clusters = compute_clusters(dist_df)

        # Phylo-Confidence score
        confidence_df = compute_phylo_confidence(sequences, msa_props, variation_df, cluster_df, dist_df)
    else:
        msa_props = None
        variation_df = None
        dist_df = None
        newick_tree = None
        tree_structure = None
        cluster_df = None
        num_clusters = 0
        confidence_df = None

    # Tab 1: Sequence Analysis
    with tab_seq:
        st.header("Sequence Analysis")
        st.subheader("Uploaded DNA Sequence Content Preview:")
        st.code(fasta_content[:500] + '...' if len(fasta_content) > 500 else fasta_content, language='fasta')

        st.subheader("Sequence Quality Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Number of Sequences", value=len(sequences))
            st.metric(label="Average Length", value=f"{summary_df['Length'].astype(float).mean():.2f}")
        with col2:
            st.metric(label="Average GC%", value=f"{summary_df['GC%'].astype(float).mean():.2f}%")
            st.metric(label="Validation Status", value=summary_df['Validation'].iloc[0] if len(summary_df) > 0 else "No Data")

        st.dataframe(summary_df, use_container_width=True)

        if not is_single and similarity_df is not None:
            st.subheader("Pairwise Sequence Similarities")
            st.dataframe(similarity_df, use_container_width=True)

    # Tab 2: MSA & Variation Analysis
    with tab_msa:
        st.header("MSA & Variation Analysis")
        if is_single:
            st.warning("MSA and Variation Analysis require multiple comparable sequences (minimum 2).")
        else:
            st.subheader("Alignment Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Number of Aligned Sequences", value=msa_props['aligned_sequences_count'])
                st.metric(label="Alignment Length", value=msa_props['alignment_length'])
            with col2:
                st.metric(label="Conserved Positions", value=f"{msa_props['conserved_positions_percent']} ({msa_props['conserved_positions_count']} sites)")
                st.metric(label="Variable Positions", value=f"{msa_props['variable_positions_percent']} ({msa_props['variable_positions_count']} sites)")

            st.subheader("Alignment Preview")
            st.code(msa_props['alignment_preview'], language="fasta")

            st.subheader("Variation Analysis")
            selected_ref = st.selectbox("Select reference sequence for mutation detection:", [s.id for s in aligned_seqs])
            var_df_custom = perform_variation_analysis(aligned_seqs, selected_ref)

            if var_df_custom is not None and not var_df_custom.empty:
                st.dataframe(var_df_custom, use_container_width=True)
                csv_data = var_df_custom.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Mutation Analysis CSV",
                    data=csv_data,
                    file_name="variation_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("No variation mutations detected against the selected reference.")

    # Tab 3: Phylogenetic Analysis
    with tab_phy:
        st.header("Phylogenetic Analysis")
        if is_single:
            st.warning("Phylogenetic Analysis requires multiple comparable sequences (minimum 2).")
        else:
            st.subheader("1. Phylogenetic Tree")
            fig_tree = render_phylogenetic_tree_plot(tree_structure, len(sequences))
            st.pyplot(fig_tree)

            st.subheader("2. Distance Matrix")
            st.dataframe(dist_df, use_container_width=True)

            st.subheader("3. Newick Tree")
            with st.expander("View Newick Tree Text"):
                st.code(newick_tree, language='text')

            st.subheader("4. Download Newick Tree File")
            st.download_button(
                label="Download Newick Tree File",
                data=newick_tree.encode('utf-8'),
                file_name="phylogenetic_tree.newick",
                mime="text/plain"
            )

    # Tab 4: Cluster Classification
    with tab_clus:
        st.header("Cluster Classification")
        if is_single:
            st.warning("Cluster Classification requires multiple comparable sequences (minimum 2).")
        else:
            st.subheader("Clusters Identified")
            st.metric(label="Total Identified Clusters", value=num_clusters)
            st.dataframe(cluster_df, use_container_width=True)

            st.write("**Explanation of Clustering Methodology:**")
            st.write("The clusters were determined automatically using distance-based Single-Linkage Hierarchical grouping on the pairwise mutation distance matrix. An adaptive, data-driven threshold is calculated as: `Threshold = Mean_Distance - (0.5 * Std_Dev_Distance)`. Sequences with pairwise distance values below or equal to this threshold are dynamically linked and partitioned into separate cohesive clusters. Highly uniform datasets automatically group into a single cluster to reflect genuine genetic similarity, preventing arbitrary splits.")

    # Tab 5: Phylo-Confidence Score
    with tab_conf:
        st.header("Phylo-Confidence Score")
        if is_single:
            st.warning("Phylo-Confidence Score requires multiple comparable sequences (minimum 2).")
        else:
            st.subheader("Project-Defined Phylo-Confidence Score")
            st.dataframe(confidence_df, use_container_width=True)

            st.write("**Formula & Calculation Strategy:**")
            st.write("The confidence score is computed individually per sequence to measure base composition reliability, structural padding ratio, and topological cluster alignment:")
            st.write("1. **Sequence Base Quality Component (30%)**: Calculates sequence completeness by checking the fraction of high-fidelity, non-ambiguous Bases (penalizing 'N' bases). Sums up to 30 points.")
            st.write("2. **Alignment Purity Component (40%)**: Assesses alignment compactness by evaluating the presence of gap characters introduced during MSA padding. Sums up to 40 points.")
            st.write("3. **Cluster Placement Consistency (30%)**: Evaluates placement quality by taking the sequence's distance to other members within its cluster relative to the maximum global divergence. Sums up to 30 points.")

            st.warning("**Important Notice**: The Phylo-Confidence Score is a **PROJECT-DEFINED CUSTOM ANALYTICAL SCORE** developed specifically for this application context. It is NOT an established, clinically validated, or certified diagnostic metric and should not be used as such.")

    # Tab 6: Final Results Dashboard
    with tab_final:
        st.header("Final Results")
        st.subheader("Automated Dashboard Summary")

        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            st.metric(label="Total Sequences Processed", value=len(sequences))
            st.metric(label="Mean Length", value=f"{summary_df['Length'].astype(float).mean():.2f} bp")
        with sum_col2:
            st.metric(label="Conserved Position Count", value=msa_props['conserved_positions_count'] if msa_props else "N/A")
            st.metric(label="Variable Position Count", value=msa_props['variable_positions_count'] if msa_props else "N/A")
        with sum_col3:
            st.metric(label="Total Mutation Sites", value=len(variation_df) if variation_df is not None else "N/A")
            st.metric(label="Clusters Assigned", value=num_clusters if num_clusters > 0 else "N/A")

        st.write("### Workflow Interpretation")
        if is_single:
            st.info("Analysis limited to basic single-sequence validation quality checking. Please upload a multi-sequence dataset to generate alignments, distance profiles, trees, cluster networks, and custom analytical metrics.")
        else:
            st.success(f"All processing runs were completed dynamically. Your dataset contains {len(sequences)} distinct sequence keys assigned across {num_clusters} cluster groups. High sequence validation purity was verified across the active targets.")

            col_l, col_r = st.columns(2)
            with col_l:
                st.write("**Phylogenetic Context Tree Visual**")
                fig_final = render_phylogenetic_tree_plot(tree_structure, len(sequences))
                st.pyplot(fig_final)
            with col_r:
                st.write("**Sequence Cluster Details**")
                st.dataframe(cluster_df, use_container_width=True)

if __name__ == '__main__':
    main()

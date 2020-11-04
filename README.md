# Computational Biology
This is a publicly available repo that has implemented the algorithms available in the book Bioinformatics 
Algorithms: An active Learning Approach using the Python language.

You can find the following chapters implemented in the relevant files.

* [Chapter 1](./dna_replication.py): Where in the Genome Does DNA Replication Begin?
    * most_frequent_kmer - Find the most common k-length sequence
    * skew_loc - Finds the pivoting point of the G-C graph
    * motif_enumeration - Find a list of k_mers that appears in a set of sequence with at most d mismatches
    * clump_finding - Find l,t clumps in a DNA sequences
    
* [Chapter 2](./dna_patterns.py): Which DNA Patterns Play the Role of Molecular Clocks?
    * generate_motif_profile - Generates a 4xk profile matrix representing the percentage of nucleotide in each column
    * median_string - Find the k_mer that has the lowest distance score with respect to a list of strands
    * greedy_motif_search - Finds a k_mer in a greedy approach which has the highest chance of appearing
    * gibbs_sampler - Uses a Monte-Carlo simulation to find the best set of k_mers
    
* [Chapter 3](./genome_assembly.py): How Do We Assemble Genomes?
    * de_bruijn_graph - Creates a De-Bruijn graph from a list of k_mers
    * generate_eulerian_walk - Finds an eulerian walk/cycle in a Graph G.
    * read_pair_string_construction - Finds the sequence that matches a list of paired reads
    * maximal_non_branching_paths - Finds all path whose internal nodes are 1-in-1-out nodes except the first and last nodes 
    
* [Chapter 4](./sequencing_antibiotics.py): How Do We Sequence Antibiotics?
    * peptide_encoding_substring - Finds substrings in a DNA that transcript in to RNA an then translates into the peptide
    * cyclopeptide_sequencing - Finds all circular peptides that produce an equivalent theoretical mass spectrum
    * leaderboard_cyclopeptide_sequencing - Uses a scoreboard to find the most probable peptide that would produce a mass spectrum similar to the given spectrum
    * convolution_cyclopeptide_sequencing - Identify candidate amino acids based on the convolution spectrum and use a scoreboard
     
* [Chapter 5](./compare_sequences.py):  How Do We Compare Sequences?
    * find_longest_common_sequence - Finds the alignment of the two strings that maximizes the score respective to a score matrix
    * find_local_alignment - Finds a local alignment of the two strings maximizes the score
    * edit_distance - Calculates the minimum number of edits needed to make two stings equivalent
    * fitting_alignment - Finds a fitting alignment of two string maximizes the score
    * affine_gap_penalty - Finds a global alignment subjected to a gap opening and extension penalty
    * linear_space_alignment - Uses linear memory to find a global alignment

* [Chapter 6](./fragile_genome.py): Are There Fragile Regions in the Human Genome?
    * greedy_sort - Solves the reversal problem by aligning each sorting the permutation in ascending order.
    * coloured_edges - Returns edges joining synteny blocks in a chromosome
    * graph_to_cycles - Generates the circular chromosomes that are responsible for the coloured edges.
    * two_break_on_genome - Performs a 2-break on the genome
    * two_break_sorting - Generates the shortest sequence of genomes transforming one genome into the other.

* [Chapter 7](./evolutionary_patterns.py): Which Animal Gave Us SARS?
    * calculate_leaf_distance - Uses the Floyd-Warshall algorithm to calculate distances between all leaves
    * additive_phylogeny - Creates a phylogeny tree using a bottom up approach based on leaf distances
    * upgma - Creates a phylogeny tree using Unweighted Pair Group Method with Arithmetic Mean
    * neighbour_join - Transforms the distance matrix to the neighbour-joining-matrix and use it for tree creation
    * create_character_sequence - Finds a labeling of all nodes of the tree by strings of length m that minimizes the tree’s parsimony score
    * move_sub_tree - Generate all possible rooted trees with only O(num_nodes) time complexity
    * nearest_neighbour_interchange - Tries to find a solution for the large parsimony problem using the nearest neighbour interchange as a heuristic
    
* [Chapter 8](./clustering_algorithms.py): How Did Yeast Become a Wine Maker? 
    * farthest_first_travel - Select k centers that are the furthest from each other
    * k_means_initializer - Chooses each point at random in such a way that distant points are more likely to be chosen than nearby points
    * k_means_clustering - Clusters the dataset using k-means clustering into k clusters
    * soft_k_means - Cluster by assign each data point a responsibility for each cluster
    * randomized_haplotype_search - Finds a set of snips S' in S that maximizes Diff(S’, T)
    * perfect_phylogeny - Creates a phylogeny tree based on snips
    
* [Chapter 9](./pattern_matching.py): How Do We Locate Disease-Causing Mutations?
* [Chapter 10](./hidden_markov_models.py): Why Have Biologists Still Not Developed An HIV Vaccine? 
* [Chapter 11](./peptide_sequence.py): Was T. Rex Just A Big Chicken?
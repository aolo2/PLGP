#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#define MASTER 0
#define REF 0

typedef unsigned vertex_id_t;
typedef unsigned long long edge_id_t;

typedef struct vec {
    int *data;
    int size;
    int cap;
} vec;

typedef struct vecf {
    float *data;
    int size;
    int cap;
} vecf;

typedef struct graph_t {
    vertex_id_t n;
    edge_id_t m;
    vertex_id_t offset;
    edge_id_t *starts;
    vertex_id_t *ends;
    int *sigma;
    vec *predecessors;
    int master_size;
} graph_t;

typedef struct csr2 {
    vec starts;
    vec counts;
    vec dist;
    vec sigma;
    vec ends;
} csr2;

static void
vec_push(vec *v, int item)
{
    if (v->size == v->cap) {
        if (v->cap == 0) {
            v->cap = 4;
        } else {
            v->cap *= 2;
        }
        
        v->data = realloc(v->data, v->cap * sizeof(int));
    }
    
    v->data[v->size++] = item;
}

static inline void
vec_clear(vec *v)
{
    v->size = 0;
}

static inline void
vecf_clear(vecf *v)
{
    v->size = 0;
}

static void
vecf_push(vecf *v, float item)
{
    if (v->size == v->cap) {
        if (v->cap == 0) {
            v->cap = 4;
        } else {
            v->cap *= 2;
        }
        
        v->data = realloc(v->data, v->cap * sizeof(float));
    }
    
    v->data[v->size++] = item;
}

#if REF
#include "brandes_cpu.c"
#endif


static graph_t
read_graph(char *filename)
{
    graph_t result = { 0 };
    
    unsigned char align;
    
    FILE *f = fopen(filename, "rb");
    assert(f);
    
    assert(fread(&result.n, sizeof(vertex_id_t), 1, f) == 1);
    assert(fread(&result.m, sizeof(edge_id_t), 1, f) == 1);
    assert(fread(&align, sizeof(unsigned char), 1, f) == 1);
    
    result.starts = (edge_id_t *) malloc((result.n + 1) * sizeof(edge_id_t));
    result.ends = (vertex_id_t *) malloc(result.m * sizeof(vertex_id_t));
    
    assert(result.starts);
    assert(result.ends);
    
    assert(fread(result.starts, sizeof(edge_id_t) * (result.n + 1), 1, f) == 1);
    assert(fread(result.ends, sizeof(vertex_id_t) * result.m, 1, f) == 1);
    
    fclose(f);
    
    return(result);
}

static void
write_result(char *filename, double *bc, int degree)
{
    FILE *f = fopen(filename, "wb");
    assert(f);
    assert(fwrite(bc, degree * sizeof(double), 1, f) == 1);
    fclose(f);
}

static inline bool
owned_by(int vertex, int offset, int count)
{
    bool result = (offset <= vertex && vertex < offset + count);
    return(result);
}

static void
master_partition_graph_1d(MPI_Comm comm, int size, graph_t graph)
{
    int nverts = graph.n;
    int batch_size = nverts / size;
    int master_verts_count = graph.n;
    
    MPI_Request *throwaway = malloc(size * sizeof(MPI_Request));
    assert(throwaway);
    
    for (int p = 0; p < size; ++p) {
        int from = p * batch_size;
        int to = (p + 1) * batch_size;
        
        if (p == size - 1) {
            to = nverts;
        }
        
        int edges_from = graph.starts[from];
        int edges_to = graph.starts[to];
        
        int verts_count = to - from;
        int edges_count = edges_to - edges_from;
        
        MPI_Isend(&from, 1, MPI_INT, p, 0, comm, throwaway + p);
        MPI_Isend(&verts_count, 1, MPI_INT, p, 0, comm, throwaway + p);
        MPI_Isend(&edges_count, 1, MPI_INT, p, 0, comm, throwaway + p);
        MPI_Isend(&master_verts_count, 1, MPI_INT, p, 0, comm, throwaway + p);
        
        MPI_Isend(graph.starts + from, verts_count + 1, MPI_LONG, p, 0, comm, throwaway + p);
        MPI_Isend(graph.ends + edges_from, edges_count, MPI_INT, p, 0, comm, throwaway + p);
    }
}

static graph_t
all_receive_partitions_1d(MPI_Comm comm)
{
    int nverts = 0;
    int nedges = 0;
    int offset = 0;
    int master_size = 0;
    
    MPI_Barrier(comm);
    
    MPI_Recv(&offset, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, 0);
    MPI_Recv(&nverts, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, 0);
    MPI_Recv(&nedges, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, 0);
    MPI_Recv(&master_size, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, 0);
    
    graph_t result = { 0 };
    
    result.n = nverts;
    result.m = nedges;
    result.offset = offset;
    result.master_size = master_size;
    
    result.starts = (edge_id_t *) malloc((result.n + 1) * sizeof(edge_id_t));
    result.ends = (vertex_id_t *) malloc(result.m * sizeof(vertex_id_t));
    result.sigma = (int *) malloc(result.n * sizeof(int));
    result.predecessors = (vec *) calloc(1, result.n * sizeof(vec));
    
    assert(result.starts);
    assert(result.ends);
    assert(result.sigma);
    assert(result.predecessors);
    
    MPI_Recv(result.starts, result.n + 1, MPI_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, 0);
    MPI_Recv(result.ends, result.m, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, 0);
    
    return(result);
}

static int
all_should_terminate(MPI_Comm comm, int rank, int size, vec FS)
{
    int result;
    int fs_size = FS.size;
    int *fs_sizes = malloc(size * sizeof(int));
    assert(fs_sizes);
    
    MPI_Gather(&fs_size, 1, MPI_INT, fs_sizes, 1, MPI_INT, MASTER, comm);
    
    if (rank == MASTER) {
        result = 1;
        for (int v = 0; v < size; ++v) {
            if (fs_sizes[v] > 0) {
                result = 0;
                break;
            }
        }
    }
    
    MPI_Bcast(&result, 1, MPI_INT, MASTER, comm);
    
    return(result);
}

static void
all_collect_local_neighbours_csr(graph_t graph, int *all_dist, vec FS, csr2 *frontier)
{
    int offset = graph.offset;
    
    vec starts = { 0 };
    vec counts = { 0 };
    vec dist = { 0 };
    vec sigma = { 0 }; 
    vec ends = { 0 };
    
    /* NOTE(aolo2): collect neighbours */
    for (int i = 0; i < FS.size; ++i) {
        int vertex = FS.data[i];
        
        int from = graph.starts[vertex - offset];
        int to = graph.starts[vertex + 1 - offset];
        
        vec_push(&starts, vertex);
        vec_push(&counts, to - from);
        vec_push(&dist, all_dist[vertex - offset]);
        vec_push(&sigma, graph.sigma[vertex - offset]);
        
        for (unsigned e = from - graph.starts[0]; e < to - graph.starts[0]; ++e) {
            int end = graph.ends[e];
            vec_push(&ends, end);
        }
    }
    
    frontier->starts = starts;
    frontier->counts = counts;
    frontier->dist = dist;
    frontier->sigma = sigma;
    frontier->ends = ends;
}

static void
all_send_owned_csr(MPI_Comm comm, csr2 frontier,
                   graph_t graph, int offset, int *all_dist, int other, int other_offset, int other_count)
{
    vec starts_other = { 0 };
    vec counts_other = { 0 };
    vec ends_other = { 0 };
    vec dist = { 0 };
    vec sigma = { 0 };
    
    int *end = frontier.ends.data;
    
    for (int v = 0; v < frontier.starts.size; ++v) {
        int start = frontier.starts.data[v];
        int neighbours_owned = 0;
        
        for (int i = 0; i < frontier.counts.data[v]; ++i) {
            int e = *end++;
            if (owned_by(e, other_offset, other_count)) {
                vec_push(&ends_other, e);
                ++neighbours_owned;
            }
        }
        
        if (neighbours_owned > 0) {
            vec_push(&starts_other, start);
            vec_push(&counts_other, neighbours_owned);
            vec_push(&dist, all_dist[start - offset]);
            vec_push(&sigma, graph.sigma[start - offset]);
        }
    }
    
    int nverts = starts_other.size;
    
    MPI_Request throwaway;
    
    MPI_Isend(&starts_other.size, 1, MPI_INT, other, 0, comm, &throwaway);
    MPI_Isend(&ends_other.size, 1, MPI_INT, other, 0, comm, &throwaway);
    
    MPI_Isend(starts_other.data, nverts, MPI_INT, other, 0, comm, &throwaway);
    MPI_Isend(counts_other.data, nverts, MPI_INT, other, 0, comm, &throwaway);
    MPI_Isend(dist.data, nverts, MPI_INT, other, 0, comm, &throwaway);
    MPI_Isend(sigma.data, nverts, MPI_INT, other, 0, comm, &throwaway);
    
    MPI_Isend(ends_other.data, ends_other.size, MPI_INT, other, 0, comm, &throwaway);
}

static void
all_receive_owned_csr(MPI_Comm comm, csr2 *frontier, int other)
{
    int nstarts = 0;
    int nends = 0;
    
    int *starts = NULL;
    int *counts = NULL;
    int *dist = NULL;
    int *sigma = NULL;
    int *ends = NULL;
    
    MPI_Recv(&nstarts, 1, MPI_INT, other, MPI_ANY_TAG, comm, 0);
    MPI_Recv(&nends, 1, MPI_INT, other, MPI_ANY_TAG, comm, 0);
    
    starts = malloc(nstarts * sizeof(int));
    counts = malloc(nstarts * sizeof(int));
    dist = malloc(nstarts * sizeof(int));
    sigma = malloc(nstarts * sizeof(int));
    ends = malloc(nends * sizeof(int));
    
    MPI_Recv(starts, nstarts, MPI_INT, other, 0, comm, 0);
    MPI_Recv(counts, nstarts, MPI_INT, other, 0, comm, 0);
    MPI_Recv(dist, nstarts, MPI_INT, other, 0, comm, 0);
    MPI_Recv(sigma, nstarts, MPI_INT, other, 0, comm, 0);
    
    MPI_Recv(ends, nends, MPI_INT, other, 0, comm, 0);
    
    for (int i = 0; i < nstarts; ++i) {
        vec_push(&frontier->starts, starts[i]);
        vec_push(&frontier->counts, counts[i]);
        vec_push(&frontier->dist, dist[i]);
        vec_push(&frontier->sigma, sigma[i]);
    }
    
    for (int i = 0; i < nends; ++i) {
        vec_push(&frontier->ends, ends[i]);
    }
}


static void
all_exchange_offsets(MPI_Comm comm, graph_t graph, int *offsets, int *counts)
{
    MPI_Allgather(&graph.offset, 1, MPI_INT, offsets, 1, MPI_INT, comm);
    MPI_Allgather(&graph.n, 1, MPI_INT, counts, 1, MPI_INT, comm);
}

static int
all_max_level(MPI_Comm comm, int level)
{
    int max_level;
    MPI_Allreduce(&level, &max_level, 1, MPI_INT, MPI_MAX, comm);
    return(max_level);
}

static void
all_send_predecessors(MPI_Comm comm, vec external_predecessors, vec external_sigma, vecf external_delta, int other, int other_offset, int other_count)
{
    vec preds_owned_by_other = { 0 };
    vec sigma_owned_by_other = { 0 };
    vecf delta_owned_by_other = { 0 };
    
    for (int i = 0; i < external_predecessors.size; ++i) {
        int p = external_predecessors.data[i];
        if (owned_by(p, other_offset, other_count)) {
            vec_push(&preds_owned_by_other, p);
            vec_push(&sigma_owned_by_other, external_sigma.data[i]);
            vecf_push(&delta_owned_by_other, external_delta.data[i]);
        }
    }
    
    
    int count = preds_owned_by_other.size;
    
    MPI_Request throwaway;
    
    MPI_Isend(&count, 1, MPI_INT, other, 0, comm, &throwaway);
    MPI_Isend(preds_owned_by_other.data, count, MPI_INT, other, 0, comm, &throwaway);
    MPI_Isend(sigma_owned_by_other.data, count, MPI_INT, other, 0, comm, &throwaway);
    MPI_Isend(delta_owned_by_other.data, count, MPI_FLOAT, other, 0, comm, &throwaway);
}

static void
all_receive_predecessors(MPI_Comm comm, vec *external_predecessors, vec *external_sigma, vecf *external_delta, int other)
{
    int count;
    int *other_preds = NULL;
    int *other_sigma = NULL;
    float *other_delta = NULL;
    
    MPI_Recv(&count, 1, MPI_INT, other, MPI_ANY_TAG, comm, 0);
    
    other_preds = malloc(count * sizeof(int));
    other_sigma = malloc(count * sizeof(int));
    other_delta = malloc(count * sizeof(float));
    
    assert(other_preds);
    assert(other_sigma);
    assert(other_delta);
    
    MPI_Recv(other_preds, count, MPI_INT, other, MPI_ANY_TAG, comm, 0);
    MPI_Recv(other_sigma, count, MPI_INT, other, MPI_ANY_TAG, comm, 0);
    MPI_Recv(other_delta, count, MPI_FLOAT, other, MPI_ANY_TAG, comm, 0);
    
    for (int i = 0; i < count; ++i) {
        vec_push(external_predecessors, other_preds[i]);
        vec_push(external_sigma, other_sigma[i]);
        vecf_push(external_delta, other_delta[i]);
    }
    
    free(other_preds);
    free(other_sigma);
    free(other_delta);
}

static void
all_brandes(MPI_Comm comm, int rank, int size, graph_t graph, int start, int *offsets, int *counts, int *dist, double *bc)
{
    int level = 0;
    int offset = graph.offset;
    int count = graph.n;
    vec FS = { 0 };
    csr2 frontier = { 0 };
    double *delta = malloc(count * sizeof(double));
    
    for (int v = 0; v < count; ++v) {
        dist[v] = -1;
        delta[v] = 0;
        graph.sigma[v] = 0;
        vec_clear(graph.predecessors + v);
    }
    
    if (owned_by(start, offset, count)) {
        dist[start - offset] = 0;
        graph.sigma[start - offset] = 1;
    }
    
    for (;;) {
        vec_clear(&FS);
        for (int v = 0; v < count; ++v) {
            if (dist[v] == level) {
                vec_push(&FS, v + offset);
            }
        }
        
        if (all_should_terminate(comm, rank, size, FS)) {
            break;
        }
        
        all_collect_local_neighbours_csr(graph, dist, FS, &frontier);
        
        for (int p = 0; p < size; ++p) {
            if (p != rank) {
                all_send_owned_csr(comm, frontier, graph, offset, dist, p, offsets[p], counts[p]);
            }
        }
        
        for (int p = 0; p < size; ++p) {
            if (p != rank) {
                all_receive_owned_csr(comm, &frontier, p);
            }
        }
        
        int *end = frontier.ends.data;
        
        for (int v = 0; v < frontier.starts.size; ++v) {
            int start = frontier.starts.data[v];
            for (int i = 0; i < frontier.counts.data[v]; ++i) {
                int e = *end++;
                if (owned_by(e, offset, count)) {
                    int w = e - offset;
                    
                    // w found for the first time [comment from brandes01]
                    if (dist[w] < 0) {
                        dist[w] = frontier.dist.data[v] + 1;
                    }
                    
                    // shortest path to w via v? [comment from brandes01]
                    if (dist[w] == frontier.dist.data[v] + 1) {
                        graph.sigma[w] += frontier.sigma.data[v];
                        vec_push(graph.predecessors + w, start);
                    }
                }
            }
        }
        
        ++level;
    }
    
    int global_level = all_max_level(comm, level);
    
    while (global_level >= 0) {
        vec  external_predecessors = { 0 };
        vec  external_sigma = { 0 };
        vecf external_delta = { 0 };
        
        /* NOTE(aolo2): get predecessors of w that belong to other procceses, for each of them
also get delta[w] and sigma[w] */
        for (int w = 0; w < count; ++w) {
            if (dist[w] == global_level) {
                
                //printf("%d[%d] ", w + offset, dist[w]);
                
                vec preds = graph.predecessors[w];
                for (int i = 0; i < preds.size; ++i) {
                    int v = preds.data[i];
                    //printf("%d = predecessor of %d[%d]\n", v, w + offset, dist[w]);
                    if (owned_by(v, offset, count)) {
                        delta[v - offset] += (double) graph.sigma[v - offset] / graph.sigma[w] * (1.0 + delta[w]);
                    } else {
                        vec_push(&external_predecessors, v);
                        vec_push(&external_sigma, graph.sigma[w]);
                        vecf_push(&external_delta, delta[w]);
                    }
                }
            }
        }
        
        for (int p = 0; p < size; ++p) {
            if (p != rank) {
                all_send_predecessors(comm, external_predecessors, external_sigma, external_delta, 
                                      p, offsets[p], counts[p]);
            }
        }
        
        vec_clear(&external_predecessors);
        vec_clear(&external_sigma);
        vecf_clear(&external_delta);
        
        for (int p = 0; p < size; ++p) {
            if (p != rank) {
                all_receive_predecessors(comm, &external_predecessors, &external_sigma, &external_delta, 
                                         p);
            }
        }
        
        for (int i = 0; i < external_predecessors.size; ++i) {
            int v = external_predecessors.data[i];
            int sigma_w = external_sigma.data[i];
            float delta_w = external_delta.data[i];
            delta[v - offset] += (double) graph.sigma[v - offset] / sigma_w * (1.0 + delta_w);
        }
        
        for (int w = 0; w < count; ++w) {
            if (dist[w] == global_level) {
                if (w != start - offset) {
                    bc[w] += delta[w];
                }
            }
        }
        
        --global_level;
        
        MPI_Barrier(comm);
    }
}


static void
all_send_bc(MPI_Comm comm, int count, double *bc)
{
    MPI_Request throwaway;
    MPI_Isend(&count, 1, MPI_INT, MASTER, 0, comm, &throwaway);
    MPI_Isend(bc, count, MPI_DOUBLE, MASTER, 0, comm, &throwaway);
}

static void
master_gather_bc(MPI_Comm comm, int size, double *bc)
{
    int accum = 0;
    int count;
    
    for (int i = 0; i < size; ++i) {
        MPI_Recv(&count, 1, MPI_INT, i, MPI_ANY_TAG, comm, 0);
        MPI_Recv(bc + accum, count, MPI_DOUBLE, i, MPI_ANY_TAG, comm, 0);
        accum += count;
    }
}

int
main(int argc, char **argv)
{
#if REF
    
    graph_t graph = read_graph(argv[1]);
    double *betweenness_centrality = (double *) calloc(1, graph.n * sizeof(double));
    
    for (unsigned v = 0; v < graph.n; ++v) {
        brandes_cpu(betweenness_centrality, graph.starts, graph.ends, 
                    graph.n, v);
    }
    
    printf("BC: ");
    for (unsigned v = 0; v < graph.n; ++v) {
        printf("%.2ff ", betweenness_centrality[v]);
    }
    printf("\n");
    
#else
    
    int rank, size;
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    if (argc != 3) {
        if (rank == MASTER) {
            printf("[ERROR] Usage: %s in out\n", argv[0]);
        }
        return(MPI_Finalize());
    }
    
    graph_t master_graph = { 0 };
    graph_t graph = { 0 };
    
    int *offsets = malloc(size * sizeof(int));
    int *counts = malloc(size * sizeof(int));
    int *dist = NULL;
    int *master_dist = NULL;
    double *master_bc = NULL;
    double *bc = NULL;
    
    assert(offsets);
    assert(counts);
    
    if (rank == MASTER) {
        master_graph = read_graph(argv[1]);
        master_partition_graph_1d(comm, size, master_graph);
    }
    
    graph = all_receive_partitions_1d(comm);
    
    dist = malloc(graph.n * sizeof(int));
    bc = calloc(1, graph.n * sizeof(double));
    
    assert(dist);
    assert(bc);
    
    all_exchange_offsets(comm, graph, offsets, counts);
    
    for (int v = 0; v < graph.master_size; ++v) {
        all_brandes(comm, rank, size, graph, v, offsets, counts, dist, bc);
        MPI_Barrier(comm);
    }
    
    all_send_bc(comm, graph.n, bc);
    
    if (rank == MASTER) {
        master_dist = malloc(master_graph.n * sizeof(int));
        master_bc = malloc(master_graph.n * sizeof(double));
        
        assert(master_dist);
        assert(master_bc);
        
        master_gather_bc(comm, size, master_bc);
        
        printf("BC: ");
        for (unsigned v = 0; v < master_graph.n; ++v) {
            printf("%.2ff ", master_bc[v]);
        }
        printf("\n");
    }
    
    int rc = MPI_Finalize();
    
    return(rc);
#endif
}
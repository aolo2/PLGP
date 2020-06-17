#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#define MASTER 0

typedef unsigned vertex_id_t;
typedef unsigned long long edge_id_t;

typedef struct vec {
    int *data;
    int size;
    int cap;
} vec;

typedef struct graph_t {
    vertex_id_t n;
    edge_id_t m;
    vertex_id_t offset;
    edge_id_t *starts;
    vertex_id_t *ends;
} graph_t;

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

static void
vec_resize(vec *v, int size)
{
    if (size > v->cap) {
        v->cap = size;
        v->data = realloc(v->data, v->cap * sizeof(int));
    }
}

static inline void
vec_clear(vec *v)
{
    v->size = 0;
}

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
        
        MPI_Isend(graph.starts + from, verts_count + 1, MPI_LONG, p, 0, comm, throwaway + p);
        MPI_Isend(graph.ends + edges_from, edges_count, MPI_INT, p, 0, comm, throwaway + p);
    }
}

static graph_t
all_receive_partitions_1d(MPI_Comm comm, int rank, int size)
{
    int nverts = 0;
    int nedges = 0;
    int offset = 0;
    
    MPI_Barrier(comm);
    
    MPI_Recv(&offset, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, 0);
    MPI_Recv(&nverts, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, 0);
    MPI_Recv(&nedges, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, 0);
    
    graph_t result = { 0 };
    
    result.n = nverts;
    result.m = nedges;
    result.offset = offset;
    
    result.starts = (edge_id_t *) malloc((result.n + 1) * sizeof(edge_id_t));
    result.ends = (vertex_id_t *) malloc(result.m * sizeof(vertex_id_t));
    
    assert(result.starts);
    assert(result.ends);
    
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
all_collect_local_neighbours(graph_t graph, vec FS, vec *NS, int rank)
{
    int offset = graph.offset;
    
    vec_clear(NS);
    
    /* NOTE(aolo2): collect neighbours */
    for (int i = 0; i < FS.size; ++i) {
        int vertex = FS.data[i];
        
        int from = graph.starts[vertex - offset];
        int to = graph.starts[vertex + 1 - offset];
        
        for (unsigned e = from - graph.starts[0]; e < to - graph.starts[0]; ++e) {
            int end = graph.ends[e];
            vec_push(NS, end);
        }
    }
}

static void
all_send_owned(MPI_Comm comm, int rank, int size, graph_t graph, vec NS,
               int other, int other_offset, int other_count)
{
    vec NS_owned_by_other = { 0 };
    
    for (int i = 0; i < NS.size; ++i) {
        int vertex = NS.data[i];
        if (owned_by(vertex, other_offset, other_count)) {
            vec_push(&NS_owned_by_other, vertex);
        }
    }
    
    MPI_Request throwaway;
    MPI_Isend(&NS_owned_by_other.size, 1, MPI_INT, other, 0, comm, &throwaway);
    MPI_Isend(NS_owned_by_other.data, NS_owned_by_other.size, MPI_INT, other, 0, comm, &throwaway);
}

static void
all_receive_owned(MPI_Comm comm, int rank, int size, graph_t graph, vec *NS, int other)
{
    int other_count;
    int *other_data;
    
    MPI_Recv(&other_count, 1, MPI_INT, other, MPI_ANY_TAG, comm, 0);
    
    other_data = malloc(other_count * sizeof(int));
    assert(other_data);
    
    MPI_Recv(other_data, other_count, MPI_INT, other, MPI_ANY_TAG, comm, 0);
    
    for (int i = 0; i < other_count; ++i) {
        vec_push(NS, other_data[i]);
    }
}

static void
all_exchange_offsets(MPI_Comm comm, int rank, int size, graph_t graph, int *offsets, int *counts)
{
    MPI_Allgather(&graph.offset, 1, MPI_INT, offsets, 1, MPI_INT, comm);
    MPI_Allgather(&graph.n, 1, MPI_INT, counts, 1, MPI_INT, comm);
}

static void
all_bfs(MPI_Comm comm, int rank, int size, graph_t graph, int start, int *offsets, int *counts, int *dist)
{
    int level = 0;
    int offset = graph.offset;
    int count = graph.n;
    vec FS = { 0 };
    vec NS = { 0 };
    
    for (unsigned v = 0; v < graph.n; ++v) {
        dist[v] = -1;
    }
    
    if (owned_by(start, offset, count)) {
        dist[start - offset] = 0;
    }
    
    for (;;) {
        vec_clear(&FS);
        for (unsigned v = 0; v < graph.n; ++v) {
            if (dist[v] == level) {
                vec_push(&FS, v + offset);
            }
        }
        
        if (all_should_terminate(comm, rank, size, FS)) {
            break;
        }
        
        all_collect_local_neighbours(graph, FS, &NS, rank);
        
        for (int p = 0; p < size; ++p) {
            if (p != rank) {
                all_send_owned(comm, rank, size, graph, NS, p, offsets[p], counts[p]);
            }
        }
        
        for (int p = 0; p < size; ++p) {
            if (p != rank) {
                all_receive_owned(comm, rank, size, graph, &NS, p);
            }
        }
        
        /* NOTE(aolo2): local neighbours could've included other processes' vertices */
        for (int i = 0; i < NS.size; ++i) {
            int vertex = NS.data[i];
            if (owned_by(vertex, offset, count)) {
                if (dist[vertex - offset] == -1) {
                    dist[vertex - offset] = level + 1;
                }
            }
        }
        
        ++level;
    }
}

static void
all_send_dist(MPI_Comm comm, int count, int *dist)
{
    MPI_Request throwaway;
    MPI_Isend(&count, 1, MPI_INT, MASTER, 0, comm, &throwaway);
    MPI_Isend(dist, count, MPI_INT, MASTER, 0, comm, &throwaway);
}

static void
master_gather_dist(MPI_Comm comm, int size, int *dist)
{
    int accum = 0;
    int count;
    
    for (int i = 0; i < size; ++i) {
        MPI_Recv(&count, 1, MPI_INT, i, MPI_ANY_TAG, comm, 0);
        MPI_Recv(dist + accum, count, MPI_INT, i, MPI_ANY_TAG, comm, 0);
        accum += count;
    }
}

static void
master_print_graph(graph_t graph)
{
    printf("|V| = %d. |E| = %lld\n", graph.n, graph.m);
    
    for (unsigned v = 0; v < graph.n; ++v) {
        printf("v_%d: ", v);
        int from = graph.starts[v];
        int to = graph.starts[v + 1];
        for (int e = from; e < to; ++e) {
            printf("%d ", graph.ends[e]);
        }
        printf("\n");
    }
}

int
main(int argc, char **argv)
{
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
    int start = 0;
    
    assert(offsets);
    assert(counts);
    
    if (rank == MASTER) {
        master_graph = read_graph(argv[1]);
        //master_print_graph(graph);
        master_partition_graph_1d(comm, size, master_graph);
    }
    
    graph = all_receive_partitions_1d(comm, rank, size);
    
    dist = malloc(graph.n * sizeof(int));
    assert(dist);
    
    all_exchange_offsets(comm, rank, size, graph, offsets, counts);
    all_bfs(comm, rank, size, graph, start, offsets, counts, dist);
    all_send_dist(comm, graph.n, dist);
    
    if (rank == MASTER) {
        master_dist = malloc(master_graph.n * sizeof(int));
        assert(master_dist);
        
        master_gather_dist(comm, size, master_dist);
        
        printf("DIST: ");
        for (unsigned v = 0; v < master_graph.n; ++v) {
            printf("%d ", master_dist[v]);
        }
        printf("\n");
    }
    
    int rc = MPI_Finalize();
    
    
    return(rc);
}
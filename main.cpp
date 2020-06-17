#define CUDA_OK(ret) assert((ret) == cudaSuccess)

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

typedef unsigned vertex_id_t;
typedef unsigned long long edge_id_t;

struct vec {
    int *data;
    int size;
    int cap;
};

struct graph_t {
    vertex_id_t n;
    edge_id_t m;
    edge_id_t *starts;
    vertex_id_t *ends;
};

__global__ void
init(int *dist, int *sigma, bool *global_work_done,
     unsigned int *device_preds_from, int *stack_head, int start, int N)
{
    int vertex = blockDim.x * blockIdx.x + threadIdx.x;
    if (vertex < N) {
        dist[vertex] = -1;
        sigma[vertex] = 0;
        device_preds_from[vertex] = 0;
    }
    
    if (vertex == start) {
        dist[vertex] = 0;
        sigma[vertex] = 1;
    }
    
    if (vertex == 0) {
        global_work_done[0] = false;
        stack_head[0] = 0;
        device_preds_from[N] = 0;
    }
}

/* NOTE(aolo2): функции collect_preds_inblock и collect_preds_interblock преобразуют массив,
*  содержащий _количество_ предикатов для каждой вершины в массив _свдигов_ относительно начала.
*  То есть, каждый элемент массива в итоге содержит сумму всех предыдущих количеств предикатов.
*  Функция xxx_inblock суммирует сдвиги внутри одного блока (CUDA потоков). xxx_interblock 
*  распространяет сдвиги между соседники блоками. */
__global__ void
collect_preds_inblock(unsigned int *device_preds_from, int N)
{
    // make thread 0 of each block collect preds from
    int offset = blockDim.x * blockIdx.x;
    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x && offset + i < N; ++i) {
            device_preds_from[offset + i] += device_preds_from[offset + i - 1];
        }
    }
}

__global__ void
collect_preds_interblock(unsigned int *device_preds_from, int *total_count, int N)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int b = 1; b < gridDim.x; ++b) {
            int offset = b * blockDim.x;
            int prev_last = (b - 1) * blockDim.x + blockDim.x - 1;
            int prev_last_value = device_preds_from[prev_last];
            
            for (int i = 0; i < blockDim.x && offset + i < N; ++i) {
                device_preds_from[offset + i] += prev_last_value;
            }
        }
        
        device_preds_from[N] += device_preds_from[N - 1];
        total_count[0] = device_preds_from[N];
    }
}

/* NOTE(aolo2): простой BFS на CUDA. Работает так: на каждой итерации _все_ вершины
*  одновременно проверяются: не были ли они только что посещены. Если были, то они посещяют 
*  всех своих непосещенных соседей. Попутно сохраняются все нужные для алгоритма Брандеса
*  величины */
__global__ void
bfs(edge_id_t *starts, vertex_id_t *ends, int *dist,
    int *sigma, vertex_id_t *preds_from, int *preds,
    int *stack, int *stack_head,
    int N, int start, int step,
    volatile bool *global_work_done)
{
    /* NOTE(aolo2): флаг остановки внутри этого блока */
    volatile __shared__ bool work_done;
    
    /* NOTE(aolo2): индекс вершины. Количество запущенных блоков и потоков гарантирует, что
    *  величина vertex покрывает _все_ вершины графа (с небольшим запасом) */
    int vertex = blockDim.x * blockIdx.x + threadIdx.x;
    
    /* NOTE(aolo2): глобальный флаг сбрасывается только одним потоком. Локальный флаг 
    *  сбрасывается в каждом блоке первый потоком этого блока */
    if (vertex == 0) {
        global_work_done[0] = false;
    }
    
    if (threadIdx.x == 0) {
        work_done = false;
    }
    
    /* NOTE(aolo2): все потоки блока должны увидеть, что work_done = false, прежде, чем приступать к работе */
    __syncthreads();
    
    /* NOTE(aolo2): отсеиваем несколько лишних потоков, которые могли возникнуть в последнем блоке */
    if (vertex < N) {
        /* NOTE(aolo2): если мы были посещены на предыдущей итерации (то есть только что) */
        if (dist[vertex] == step) {
            int from = starts[vertex];
            int to = starts[vertex + 1];
            
            /* NOTE(aolo2): стек заполняется только при первом цикле запусков (эта работа не зависит от 
            *  массива предикатов) */
            if (preds == NULL) {
                int stack_write_to = atomicAdd(stack_head, 1);
                stack[stack_write_to] = vertex;
            }
            
            /* NOTE(aolo2): обходим всех соседей как это делается в формате CSR. Если какие-то 
            *  соседи не были посещены, то устанавливаем глобальный флаг */
            for (int i = from; i < to; ++i) {
                int w = ends[i];
                
                if (dist[w] == -1) {
                    dist[w] = step + 1;
                    work_done = true;
                }
                
                if (dist[w] == step + 1) {
                    if (preds == NULL) {
                        /* NOTE(aolo2): несколько потоков могут достигнуть этой строки одновременно, используем atomic */
                        atomicAdd(sigma + w, sigma[vertex]);
                    }
                    
                    if (preds == NULL) {
                        /* NOTE(aolo2): несколько потоком могут достигнуть этой строки одновременно, используем atomic */
                        atomicAdd(preds_from + w + 1, 1);
                    } else {
                        /* NOTE(aolo2): каждый поток должен писать в свою уникальную позицию, используем atomic */
                        int write_at = atomicAdd(preds_from + w, 1);
                        preds[write_at] = vertex;
                    }
                    
                    work_done = true;
                }
            }
        }
    }
    
    /* NOTE(aolo2): первый поток каждого блока проверяет флаг (локальный). Если флаг установлен - устанавливаем глобальный флаг*/
    if (threadIdx.x == 0 && work_done) {
        global_work_done[0] = true;
    }
}

static graph_t
read_graph(char *filename)
{
    graph_t result = {};
    
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

static void
brandes(double *bc, edge_id_t *device_starts, vertex_id_t *device_ends,
        int *device_dist, int *device_sigma, unsigned int *device_preds_from,
        int *device_stack,
        double *host_delta, int *host_preds_from, int *host_sigma,
        int *host_dist,int N, int vertex)
{
    
    bool *device_work_done = 0;
    int *device_preds = 0;
    int *device_stack_head = 0;
    
    /* NOTE(aolo2): оптимальный размер блока установлен экспериментальным путем (так советуют
    *  делать в документации). Количество блоков рассчитано так, чтобы потоки покрывали все
    *  вершины графа (с запасом, за счет округления вверх). */
    int thread_count = 32;
    int block_count = (N + thread_count - 1) / thread_count; /* "+ thread_count - 1" - вот тут округление вверх */
    bool host_work_done = false;
    
    /* NOTE(aolo2): глобальный (с точки зрения GPU) флаг, который означает что BFS закончен */
    CUDA_OK(cudaMalloc(&device_work_done, sizeof(bool)));
    
    /* NOTE(aolo2): глобальный (с точки зрения GPU) указатель на вершину стека */
    CUDA_OK(cudaMalloc(&device_stack_head, sizeof(int)));
    
    /* NOTE(aolo2): инициализация массивов вынесена в CUDA-ядро, с тем чтобы не копировать данные туда-сюда */
    init<<<block_count, thread_count>>>(device_dist, device_sigma, device_work_done,
                                        device_preds_from, device_stack_head,
                                        vertex, N);
    
    /* NOTE(aolo2): в алгоритме Брандеса используются растущие списки предикатов. Так как внутри CUDA-ядра
    *  выделять и перевыделять память под растущие списки проблематично, используется два прохода алгоритма.
    *  Первый проход только считает количество предикатов, которые будут сохранены. Эта информация используется
    *  для выделения памяти и расчета сдвигов в массиве предикатов (каждый поток будет знать по какому сдвигу 
    *  ему писать). Второй проход же уже сохраняет по этим сдвигам сами предикаты */
    for (int bfs_step = 0; ; ++bfs_step) {
        bfs<<<block_count, thread_count>>>(device_starts, device_ends,
                                           device_dist, device_sigma,
                                           device_preds_from, NULL,
                                           device_stack, device_stack_head,
                                           N, vertex, bfs_step,
                                           device_work_done);
        
        /* NOTE(aolo2): проверим, не установлен ли глобальный флаг остановки алгоритма */
        CUDA_OK(cudaMemcpy(&host_work_done, device_work_done, sizeof(bool), cudaMemcpyDeviceToHost));
        
        if (!host_work_done) {
            break;
        }
    }
    
    int *device_total_count = 0;
    int host_total_count = 0;
    
    CUDA_OK(cudaMalloc(&device_total_count, sizeof(int)));
    
    /* NOTE(aolo2): здесь складываются и продвигаются посчитанные сдвиги предикатов */
    collect_preds_inblock<<<block_count, thread_count>>>(device_preds_from, N);
    collect_preds_interblock<<<block_count, thread_count>>>(device_preds_from, device_total_count, N);
    
    CUDA_OK(cudaMemcpy(host_preds_from, device_preds_from, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&host_total_count, device_total_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMalloc(&device_preds, host_total_count * sizeof(unsigned int)));
    
    /* NOTE(aolo2): второй запуск алгоритма, здесь уже сохраняются настоящие предикаты */
    for (int bfs_step = 0; ; ++bfs_step) {
        bfs<<<block_count, thread_count>>>(device_starts, device_ends,
                                           device_dist, device_sigma,
                                           device_preds_from, device_preds,
                                           NULL, NULL,
                                           N, vertex, bfs_step,
                                           device_work_done);
        
        /* NOTE(aolo2): аналогично, проверяем флаг остановки */
        CUDA_OK(cudaMemcpy(&host_work_done, device_work_done, sizeof(bool), cudaMemcpyDeviceToHost));
        
        if (!host_work_done) {
            break;
        }
    }
    
    
    int *host_preds = (int *) malloc(host_total_count * sizeof(int));
    int host_stack_head = 0;
    
    CUDA_OK(cudaMemcpy(host_preds, device_preds, host_total_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&host_stack_head, device_stack_head, sizeof(int), cudaMemcpyDeviceToHost));
    
    /* NOTE(aolo2): копируем стек, dist и sigma на CPU. Часть алгоритма с проходом по стеку и суммированием
    *  производится в последовательном режиме */
    int *host_stack = (int *) malloc(host_stack_head * sizeof(int));
    
    CUDA_OK(cudaMemcpy(host_stack, device_stack, host_stack_head * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(host_dist, device_dist, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(host_sigma, device_sigma, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    memset(host_delta, 0x00, N * sizeof(double));
    
    /* NOTE(aolo2): совпадает с псевдокодом алгоритма Брандеса */
    while (host_stack_head > 0) {
        int w = host_stack[host_stack_head - 1];
        --host_stack_head;
        
        int from = host_preds_from[w];
        int to = host_preds_from[w + 1];
        
        for (int i = from; i < to; ++i) {
            int v = host_preds[i];
            host_delta[v] += (double) host_sigma[v] / host_sigma[w] * (1.0 + host_delta[w]);
        }
        
        if (w != vertex) {
            bc[w] += 0.5f * host_delta[w];
        }
    }
    
    CUDA_OK(cudaFree(device_work_done));
    CUDA_OK(cudaFree(device_stack_head));
    CUDA_OK(cudaFree(device_total_count));
    CUDA_OK(cudaFree(device_preds));
    
    free(host_stack);
    free(host_preds);
}

int
main(int argc, char **argv)
{
	if (argc != 3) {
		printf("[ERROR] Usage: %s in out\n", argv[0]);
		return(1);
	}
    
    /* NOTE(aolo2): читать граф будем из argv[1], писать ответ - в argv[2] */
    char *filename = argv[1];
    char *output_filename = argv[2];
    
    /* NOTE(aolo2): читаем граф в соответствии с форматом, который используется в валидирующем коде */
    graph_t g = read_graph(filename);
    
    edge_id_t *starts = g.starts;
    vertex_id_t *edges = g.ends;
    int N = g.n;
    int nends = starts[N];
    
    /* NOTE(aolo2): выделяем память под используемые в алгоритме Брандеса величины. Здесь выделяется
    *  память только на хосте (CPU). Величины, используемые на девайсе (GPU), выделяются через cudaMalloc */
    double *host_delta = (double *) malloc(N * sizeof(double));
    int *host_preds_from = (int *) malloc((N + 1) * sizeof(int));
    int *host_sigma = (int *) malloc(N * sizeof(int));
    int *host_dist = (int *) malloc(N * sizeof(int));
    
    edge_id_t *device_starts = 0;
    vertex_id_t *device_ends = 0;
    int *device_dist = 0;
    int *device_sigma = 0;
    unsigned int *device_preds_from = 0;
    int *device_stack = 0;
    
    /* NOTE(aolo2): выделяем память под граф в формате CSR на GPU, копируем туда данные с CPU */
    CUDA_OK(cudaMalloc(&device_starts, (N + 1) * sizeof(edge_id_t)));
    CUDA_OK(cudaMalloc(&device_ends, nends * sizeof(vertex_id_t)));
    CUDA_OK(cudaMemcpy(device_starts, starts, (N + 1) * sizeof(edge_id_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(device_ends, edges, nends * sizeof(vertex_id_t), cudaMemcpyHostToDevice));
    
    /* NOTE(aolo2): выделяем память под величины, используемые в алгоритме Брандеса */
    CUDA_OK(cudaMalloc(&device_dist, N * sizeof(int)));
    CUDA_OK(cudaMalloc(&device_sigma, N * sizeof(int)));
    CUDA_OK(cudaMalloc(&device_preds_from, (N + 1) * sizeof(unsigned int)));
    CUDA_OK(cudaMalloc(&device_stack, N * sizeof(int)));
    
    /* NOTE(aolo2): в этот массив будут записываться результаты */
    double *betweenness_centrality = (double *) calloc(1, N * sizeof(double));
    
    for (int v = 0; v < N; ++v) {
        brandes(betweenness_centrality, device_starts, device_ends,
                device_dist, device_sigma, device_preds_from,
                device_stack,
                host_delta, host_preds_from, host_sigma, host_dist, N, v);
    }
    
    /* NOTE(aolo2): сохраняем результат в соответствии с форматом */
    write_result(output_filename, betweenness_centrality, N);
    
    CUDA_OK(cudaFree(device_starts));
    CUDA_OK(cudaFree(device_ends));
    CUDA_OK(cudaFree(device_dist));
    CUDA_OK(cudaFree(device_sigma));
    CUDA_OK(cudaFree(device_preds_from));
    CUDA_OK(cudaFree(device_stack));
    
    free(host_delta);
    free(host_preds_from);
    free(host_sigma);
    free(host_dist);
    
    return(0);
}
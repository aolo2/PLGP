static void
brandes_cpu(double *bc, edge_id_t *starts, vertex_id_t *ends, int N, int vertex)
{
    int *sigma = (int *) malloc(N * sizeof(int));
    int *dist = (int *) malloc(N * sizeof(int));
    double *delta = (double *) malloc(N * sizeof(double));
    
    for (int i = 0; i < N; ++i) {
        sigma[i] = 0;
        dist[i] = -1;
        delta[i] = 0;
    }
    
    int *stack = (int *) malloc(N * sizeof(int)); // stack will contain every vertex once
    int stack_head = 0;
    
    int *queue = (int *) malloc(N * sizeof(int));
    int queue_head = 0;
    int queue_tail = 0;
    int queue_size = 0;
    
    // empty lists
    struct vec *predecessors = (struct vec *) calloc(1, N * sizeof(struct vec));
    
    sigma[vertex] = 1;
    dist[vertex] = 0;
    
    queue[queue_head] = vertex;
    ++queue_head;
    ++queue_size;
    
    while (queue_size > 0) {
        int v = queue[queue_tail];
        ++queue_tail;
        --queue_size;
        
        stack[stack_head] = v;
        ++stack_head;
        
        //printf("dist=%d vertex=%d\n", dist[v], v);
        
        edge_id_t w_from = starts[v];
        edge_id_t w_to = starts[v + 1];
        
        for (edge_id_t i = w_from; i < w_to; ++i) {
            vertex_id_t w = ends[i];
            
            // w found for the first time [comment from brandes01]
            if (dist[w] < 0) {
                queue[queue_head] = w;
                ++queue_head;
                ++queue_size;
                dist[w] = dist[v] + 1;
            }
            
            // shortest path to w via v? [comment from brandes01]
            if (dist[w] == dist[v] + 1) {
                sigma[w] += sigma[v];
                vec_push(predecessors + w, v);
            }
        }
    }
    
    // S returns vertices in order of non-increasing distance from s [comment from brandes01]
    while (stack_head > 0) {
        int w = stack[stack_head - 1];
        --stack_head;
        
        
        for (int i = 0; i < predecessors[w].size; ++i) {
            int v = predecessors[w].data[i];
            //printf("%d = predecessor of %d[%d]\n", v, w, dist[w]);
            delta[v] += (double) sigma[v] / sigma[w] * (1.0 + delta[w]);
            //printf("%d %d %d %.2f. delta[%d] now is %.2f\n", v, sigma[v], sigma[w], delta[w], v, delta[v]);
        }
        
        if (w != vertex) {
            bc[w] += delta[w];
        }
    }
    
#if 0
    int offset = 0;
    for (int v = 0; v < N; ++v) {
        printf("preds %d offset = %d\n", v, offset);
        offset += predecessors[v].size;
    }
    
    for (int v = 0; v < N; ++v) {
        printf("preds %d:", v);
        for (int i = 0; i < predecessors[v].size; ++i) {
            printf(" %d", predecessors[v].data[i]);
        }
        printf("\n");
    }
    
    printf("sigma:");
    for (int i = 0; i < N; ++i) {
        printf(" %d", sigma[i]);
    }
    printf("\n");
    
    printf("dist:");
    for (int i = 0; i < N; ++i) {
        printf(" %d", dist[i]);
    }
    printf("\n");
#endif
    
#if 1
    printf("START=%d. delta:", vertex);
    for (int i = 0; i < N; ++i) {
        printf(" %.2f", delta[i]);
    }
    printf("\n");
#endif
}

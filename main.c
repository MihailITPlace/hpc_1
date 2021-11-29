#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct {
    int m, n;
    double **v;
} mat_t, *mat;

mat matrix_new(int m, int n) {
    mat x = malloc(sizeof(mat_t));
    x->v = malloc(sizeof(double *) * m);
    x->v[0] = calloc(sizeof(double), m * n);

    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < m; i++)
        x->v[i] = x->v[0] + n * i;
    x->m = m;
    x->n = n;
    return x;
}

void matrix_delete(mat m) {
    free(m->v[0]);
    free(m->v);
    free(m);
}

void matrix_transpose(mat m) {
    int i, j;
#pragma omp parallel for shared(m) private(i, j)
    for (i = 0; i < m->m; i++) {
        for (j = 0; j < i; j++) {
            double t = m->v[i][j];
            m->v[i][j] = m->v[j][i];
            m->v[j][i] = t;
        }
    }
}

mat matrix_copy(mat a) {
    mat x = matrix_new(a->m, a->n);
    for (int i = 0; i < a->m; i++)
        for (int j = 0; j < a->n; j++)
            x->v[i][j] = a->v[i][j];
    return x;
}


mat matrix_from_file(char* filename, int matrix_size) {
    mat x = matrix_new(matrix_size, matrix_size);
    FILE* f = fopen(filename, "r+");
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            fscanf(f, "%lf", &(x->v[i][j]));
        }
    }
    fclose(f);

    return x;
}

mat matrix_mul(mat x, mat y) {
    if (x->n != y->m) return 0;
    mat r = matrix_new(x->m, y->n);
    int i, j, k;
#pragma omp parallel for shared(x, y, r) private(i, j, k)
    for (i = 0; i < x->m; i++) {
        for (j = 0; j < y->n; j++)
            for (k = 0; k < x->n; k++)
                r->v[i][j] += x->v[i][k] * y->v[k][j];
    }
    return r;
}

mat matrix_minor(mat x, int d) {
    mat m = matrix_new(x->m, x->n);
    for (int i = 0; i < d; i++)
        m->v[i][i] = 1;

    int i, j;
#pragma omp parallel for shared(x) private(i, j)
    for (i = d; i < x->m; i++) {
        for (j = d; j < x->n; j++)
            m->v[i][j] = x->v[i][j];
    }
    return m;
}

// c = a + b * s
double *vmadd(double a[], double b[], double s, double c[], int n) {
    int i;
#pragma omp parallel for shared(a, b, c) private(i)
    for (i = 0; i < n; i++)
        c[i] = a[i] + s * b[i];
    return c;
}

// m = I - v v^T
mat vmul(double v[], int n) {
    mat x = matrix_new(n, n);
    int i, j;
#pragma omp parallel for shared(v) private(i, j)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            x->v[i][j] = -2 * v[i] * v[j];
    }

    int k;
#pragma omp parallel for shared(v) private(k)
    for (k = 0; k < n; k++)
        x->v[k][k] += 1;

    return x;
}

// ||x||
double vnorm(double x[], int n) {
    double sum = 0;
#pragma omp parallel for reduction (+:sum)
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

// y = x / d
double *vdiv(double x[], double d, double y[], int n) {
    int i;
#pragma omp parallel for shared(x, y) private(i)
    for (i = 0; i < n; i++)
        y[i] = x[i] / d;
    return y;
}

// take c-th column of m, put in v
double *mcol(mat m, double *v, int c) {
    for (int i = 0; i < m->m; i++)
        v[i] = m->v[i][c];
    return v;
}

void show_eigenvalues(mat input_matrix) {
    for (int i = 0; i < input_matrix->m; i++) {
        printf(" %8.3f\n", input_matrix->v[i][i]);
    }
}

void householder(mat m, mat *R, mat *Q) {
    mat q[m->m];
    mat z = m, z1;
    for (int k = 0; k < m->n && k < m->m - 1; k++) {
        double e[m->m], x[m->m], a;
        z1 = matrix_minor(z, k);
        if (z != m) matrix_delete(z);
        z = z1;

        mcol(z, x, k);
        a = vnorm(x, m->m);
        if (m->v[k][k] > 0) a = -a;

        for (int i = 0; i < m->m; i++)
            e[i] = (i == k) ? 1 : 0;

        vmadd(x, e, a, e, m->m);
        vdiv(e, vnorm(e, m->m), e, m->m);
        q[k] = vmul(e, m->m);
        z1 = matrix_mul(q[k], z);
        if (z != m) matrix_delete(z);
        z = z1;
    }
    matrix_delete(z);
    *Q = q[0];
    *R = matrix_mul(q[0], m);

    for (int i = 1; i < m->n && i < m->m - 1; i++) {
        z1 = matrix_mul(q[i], *Q);
        if (i > 1) matrix_delete(*Q);
        *Q = z1;
        matrix_delete(q[i]);
    }
    matrix_delete(q[0]);
    z = matrix_mul(*Q, m);
    matrix_delete(*R);
    *R = z;
    matrix_transpose(*Q);
}

mat find_eigenvalues(mat input_matrix, int iterations_count) {
    mat R, Q;
    mat A = matrix_copy(input_matrix);
    for (int i = 0; i < iterations_count; i++) {
        if (i % 50 == 0)
            printf("iteration %d\n", i);
        householder(A, &R, &Q);
        matrix_delete(A);
        A = matrix_mul(R, Q);
    }

    matrix_delete(R);
    matrix_delete(Q);
    return A;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("./main <количество потоков>");
        return 1;
    }

    char *end;
    int count_of_threads = (int) strtol(argv[1], &end, 10);
    printf("Threads: %d\n", count_of_threads);

    omp_set_nested(1);
    omp_set_num_threads(count_of_threads);

    double start_time = omp_get_wtime();
    mat input_matrix = matrix_from_file("../inputs/matrix-3000", 3000);
    mat E = find_eigenvalues(input_matrix, 1);
    double elapsed_time = omp_get_wtime() - start_time;

    show_eigenvalues(E);
    matrix_delete(input_matrix);
    matrix_delete(E);

    printf("Elapsed time in seconds %.6f\n", elapsed_time);
    return 0;
}
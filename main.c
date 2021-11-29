#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

void show_matrix(double **m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.3f ", m[i][j]);
        }
        printf("\n");
    }
}

void qr_decomposition(int n, double **r, double **q, double **a) {
    int k, i, j;
    double mysum;

    for (k = 0; k < n; k++){
        r[k][k] = 0;
        mysum = 0;

        for (i = 0; i < n; i++)
            mysum += a[i][k] * a[i][k];

        r[k][k] = mysum;
        r[k][k] = sqrt(r[k][k]);

#pragma omp parallel for shared(q, a, r)
        for (i = 0; i < n; i++)
            q[i][k] = a[i][k] / r[k][k];

#pragma omp parallel for shared(q, a, r) private(j, i) reduction(+:mysum)
        for(j = k + 1; j < n; j++) {
            mysum = 0;

            for(i = 0; i < n; i++)
                mysum += q[i][k] * a[i][j];
            r[k][j] = mysum;

            for (i = 0; i < n; i++)
                a[i][j] = a[i][j] - r[k][j] * q[i][k];
        }
    }
}

void matrix_mul(double  **a, double **b, double **c, int matrix_size) {
    int i, j, k;

#pragma omp parallel for shared(c) private(i, j)
    for (i = 0; i < matrix_size; i++)
        for (j = 0; j < matrix_size; j++)
            c[i][j] = 0.0;

#pragma omp parallel for shared(a, b, c) private(i, j, k)
    for (i = 0; i < matrix_size; i++) {
        for (j = 0; j < matrix_size; j++)
            for (k = 0; k < matrix_size; k++)
                c[i][j] += a[i][k] * b[k][j];
    }
}

double** matrix_from_file(char* filename, int matrix_size) {
    double **x = malloc(sizeof(double *) * matrix_size);

    FILE* f = fopen(filename, "r+");
    for (int i = 0; i < matrix_size; i++) {
        x[i] = malloc(sizeof(double) * matrix_size);
        for (int j = 0; j < matrix_size; j++) {
            fscanf(f, "%lf", &(x[i][j]));
        }
    }
    fclose(f);

    return x;
}

double** new_matrix(int matrix_size) {
    double **x = malloc(sizeof(double *) * matrix_size);
    for (int i = 0; i < matrix_size; i++) {
        x[i] = malloc(sizeof(double) * matrix_size);
    }

    return x;
}

void remove_matrix(double **x, int matrix_size) {
    for (int i = 0; i < matrix_size; i++) {
        free(x[i]);
    }
    free(x);
}

double* find_eigenvalues(double **a, int iterations_count, int matrix_size) {
    double **r, **q;

    q = new_matrix(matrix_size);
    r = new_matrix(matrix_size);

    for (int i = 0; i < iterations_count; i++) {
        if (i % 10 == 0) {
            printf("Iterations %d\n", i);
        }
        qr_decomposition(matrix_size, r, q, a);
        matrix_mul(r, q, a, matrix_size);
    }

    remove_matrix(q, matrix_size);
    remove_matrix(r, matrix_size);

    double *eigenvalues = malloc(sizeof(double) * matrix_size);
    for (int i = 0; i < matrix_size; i++) {
        eigenvalues[i] = a[i][i];
    }

    return eigenvalues;
}

void show_eigenvalues(double* input_matrix, int size) {
    for (int i = 0; i < size; i++) {
        printf(" %8.3f\n", input_matrix[i]);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("./main <количество потоков> <размер матрицы>");
        return 1;
    }

    char *end;
    int count_of_threads = (int) strtol(argv[1], &end, 10);
    int n = (int) strtol(argv[2], &end, 10);

    char input_name[255];
    sprintf(input_name, "../inputs/matrix-%d", n);
    double **a = matrix_from_file(input_name, n);

    omp_set_nested(1);
    omp_set_num_threads(count_of_threads);
    double start_time = omp_get_wtime();
    double *eigenvalues = find_eigenvalues(a, 50, n);
    double elapsed_time = omp_get_wtime() - start_time;

    show_eigenvalues(eigenvalues, n);
    printf("Elapsed time in seconds %.6f\n", elapsed_time);

    remove_matrix(a, n);
    return 0;
}



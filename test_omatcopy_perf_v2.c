#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// 定义数据类型
typedef float FLOAT;
typedef long BLASLONG;

// 标量版本 (原始实现)
int omatcopy_rn_scalar(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
    BLASLONG i, j;
    FLOAT *aptr, *bptr;

    if (rows <= 0) return(0);
    if (cols <= 0) return(0);

    aptr = a;
    bptr = b;

    if (alpha == 0.0) {
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                bptr[j] = 0.0;
            }
            bptr += ldb;
        }
        return(0);
    }

    if (alpha == 1.0) {
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                bptr[j] = aptr[j];
            }
            aptr += lda;
            bptr += ldb;
        }
        return(0);
    }

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            bptr[j] = alpha * aptr[j];
        }
        aptr += lda;
        bptr += ldb;
    }

    return(0);
}

// 改进的RVV版本
#ifdef __riscv_vector
#include <riscv_vector.h>

int omatcopy_rn_rvv(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
    BLASLONG i, j;
    FLOAT *aptr, *bptr;
    size_t vl;

    vfloat32m8_t va;
    if (rows <= 0) return(0);
    if (cols <= 0) return(0);

    aptr = a;
    bptr = b;

    if (alpha == 0.0) {
        vl = __riscv_vsetvlmax_e32m8();
        va = __riscv_vfmv_v_f_f32m8(0, vl);
        for (i = 0; i < rows; i++) {
            j = 0;
            while (j < cols) {
                vl = __riscv_vsetvl_e32m8(cols - j);
                __riscv_vse32_v_f32m8(bptr + j, va, vl);
                j += vl;
            }
            aptr += lda;
            bptr += ldb;
        }
        return(0);
    }

    if (alpha == 1.0) {
        for (i = 0; i < rows; i++) {
            j = 0;
            while (j < cols) {
                vl = __riscv_vsetvl_e32m8(cols - j);
                va = __riscv_vle32_v_f32m8(aptr + j, vl);
                __riscv_vse32_v_f32m8(bptr + j, va, vl);
                j += vl;
            }
            aptr += lda;
            bptr += ldb;
        }
        return(0);
    }

    for (i = 0; i < rows; i++) {
        j = 0;
        while (j < cols) {
            vl = __riscv_vsetvl_e32m8(cols - j);
            va = __riscv_vle32_v_f32m8(aptr + j, vl);
            va = __riscv_vfmul_vf_f32m8(va, alpha, vl);
            __riscv_vse32_v_f32m8(bptr + j, va, vl);
            j += vl;
        }
        aptr += lda;
        bptr += ldb;
    }

    return(0);
}
#else
// 如果没有RVV支持，使用标量版本
int omatcopy_rn_rvv(BLASLONG rows, BLASLONG cols, FLOAT alpha, FLOAT *a, BLASLONG lda, FLOAT *b, BLASLONG ldb)
{
    return omatcopy_rn_scalar(rows, cols, alpha, a, lda, b, ldb);
}
#endif

// 获取时间
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 初始化矩阵
void init_matrix(FLOAT *matrix, BLASLONG rows, BLASLONG cols, BLASLONG lda) {
    BLASLONG i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            matrix[i * lda + j] = (FLOAT)(i * cols + j) / 1000.0f;
        }
    }
}

// 验证结果
int verify_result(FLOAT *result1, FLOAT *result2, BLASLONG rows, BLASLONG cols, BLASLONG ldb) {
    BLASLONG i, j;
    FLOAT diff, max_diff = 0.0f;
    
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            diff = result1[i * ldb + j] - result2[i * ldb + j];
            if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
        }
    }
    
    return (max_diff < 1e-5f) ? 1 : 0;
}

int main(int argc, char *argv[]) {
    BLASLONG rows, cols, iterations;
    FLOAT alpha = 2.5f;
    FLOAT *a, *b_scalar, *b_rvv;
    BLASLONG lda, ldb;
    double start_time, end_time, scalar_time, rvv_time;
    double gflops_scalar, gflops_rvv;
    int i;
    
    // 解析命令行参数
    if (argc != 4) {
        printf("Usage: %s <rows> <cols> <iterations>\n", argv[0]);
        printf("Example: %s 3000 4000 100\n", argv[0]);
        return 1;
    }
    
    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    iterations = atoi(argv[3]);
    
    printf("=== OMATCOPY Performance Test (Improved RVV) ===\n");
    printf("Matrix Size: %ld x %ld\n", rows, cols);
    printf("Iterations: %ld\n", iterations);
    printf("Alpha: %.1f\n", alpha);
    printf("Data Type: %s\n", sizeof(FLOAT) == 4 ? "float" : "double");
#ifdef __riscv_vector
    printf("RVV Support: Yes\n");
#else
    printf("RVV Support: No (using scalar fallback)\n");
#endif
    printf("\n");
    
    // 分配内存
    lda = cols;
    ldb = cols;
    a = (FLOAT*)malloc(rows * lda * sizeof(FLOAT));
    b_scalar = (FLOAT*)malloc(rows * ldb * sizeof(FLOAT));
    b_rvv = (FLOAT*)malloc(rows * ldb * sizeof(FLOAT));
    
    if (!a || !b_scalar || !b_rvv) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    // 初始化输入矩阵
    init_matrix(a, rows, cols, lda);
    
    // 预热
    memset(b_scalar, 0, rows * ldb * sizeof(FLOAT));
    memset(b_rvv, 0, rows * ldb * sizeof(FLOAT));
    
    // 测试标量版本
    printf("Testing Scalar Version...\n");
    start_time = get_time();
    for (i = 0; i < iterations; i++) {
        omatcopy_rn_scalar(rows, cols, alpha, a, lda, b_scalar, ldb);
    }
    end_time = get_time();
    scalar_time = end_time - start_time;
    
    // 测试RVV版本
    printf("Testing Improved RVV Version...\n");
    start_time = get_time();
    for (i = 0; i < iterations; i++) {
        omatcopy_rn_rvv(rows, cols, alpha, a, lda, b_rvv, ldb);
    }
    end_time = get_time();
    rvv_time = end_time - start_time;
    
    // 验证结果
    if (!verify_result(b_scalar, b_rvv, rows, cols, ldb)) {
        printf("ERROR: Results don't match!\n");
        return 1;
    }
    
    // 计算性能指标
    double ops_per_iteration = 2.0 * rows * cols; // 读取+写入
    gflops_scalar = (ops_per_iteration * iterations) / (scalar_time * 1e9);
    gflops_rvv = (ops_per_iteration * iterations) / (rvv_time * 1e9);
    
    double improvement = ((scalar_time - rvv_time) / scalar_time) * 100.0;
    
    // 输出结果
    printf("\n=== Performance Results ===\n");
    printf("Version\t\tTotal Time(s)\tAvg Time(s)\tGFLOPS\t\tImprovement\n");
    printf("Scalar\t\t%.2f\t\t%.3f\t\t%.3f\t\t-\n", 
           scalar_time, scalar_time/iterations, gflops_scalar);
    printf("RVV\t\t%.2f\t\t%.3f\t\t%.3f\t\t+%.1f%%\n", 
           rvv_time, rvv_time/iterations, gflops_rvv, improvement);
    
    printf("\n=== Key Improvements ===\n");
    printf("Execution Time Reduction: From %.3fs to %.3fs, %.1f%% improvement\n", 
           scalar_time/iterations, rvv_time/iterations, improvement);
    printf("GFLOPS Enhancement: From %.3f to %.3f, %.1f%% increase\n", 
           gflops_scalar, gflops_rvv, ((gflops_rvv - gflops_scalar) / gflops_scalar) * 100.0);
    
    // 清理内存
    free(a);
    free(b_scalar);
    free(b_rvv);
    
    return 0;
}

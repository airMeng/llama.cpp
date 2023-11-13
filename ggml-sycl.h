#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_sycl_init(void);

void ggml_sycl_mul(const struct ggml_tensor *src0,
                   const struct ggml_tensor *src1, struct ggml_tensor *dst);
bool ggml_sycl_can_mul_mat(const struct ggml_tensor *src0,
                           const struct ggml_tensor *src1,
                           struct ggml_tensor *dst);
size_t ggml_sycl_mul_mat_get_wsize(const struct ggml_tensor *src0,
                                   const struct ggml_tensor *src1,
                                   struct ggml_tensor *dst);
void ggml_sycl_mul_mat(const struct ggml_tensor *src0,
                       const struct ggml_tensor *src1, struct ggml_tensor *dst,
                       void *wdata, size_t wsize);

void *ggml_sycl_host_malloc(size_t size);
void ggml_sycl_host_free(void *ptr);

void ggml_sycl_free_data(const struct ggml_tensor *tensor);

void ggml_sycl_transform_tensor(void *data, struct ggml_tensor *tensor);

#ifdef __cplusplus
}
#endif

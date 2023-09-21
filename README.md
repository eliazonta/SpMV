# SpMV 
![Sequential build](https://github.com/eliazonta/SpMV/actions/workflows/sequential.yml/badge.svg)


![CUDA build](https://github.com/eliazonta/SpMV/actions/workflows/parallel.yml/badge.svg)

Sequential and parallel (GPU based) implementation of a Sparse Matrix Vector Multiplication using (Compressed Sparse Row (CSR) matrix format.

## Sequential execution
```shell
make sequential
```
```shell
./bin/SpMV-SEQ <iterations> <print mode [1 default, 2 view data info]> <file path>
```

## Parallel execution
```shell
make parallel
```
```shell
./bin/SpMV-CUDA <threads num> <iterations> <print mode [1 default, 2 view data info]> <file path>
```

### TODO LIST
- [X] ⚠️ **IMPORTANT** more tests on my GPU (working on my Mac M1 rn🥲)
- [ ] more performances evaluation 
- [ ] CUDA workflow
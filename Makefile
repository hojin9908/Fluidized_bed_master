make :
	nvcc -w -use_fast_math -arch=sm_60 -O3 -expt-relaxed-constexpr SOPHIA_gpu.cu -o SOPHIA_gpu -I./cub-1.8.0/ -lpthread
	nvcc -arch=sm_60 -O3 output-merge.cu -o output-merge
clean :
	rm -rf *.*~ SOPHIA_gpu output-merge

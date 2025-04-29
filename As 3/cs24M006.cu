#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
#define MOD 1000000007


__global__ void adjustWeights(int* weight, char* types, int E) {
  int idx =blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>=E) return;
  if(types[idx]=='g') weight[idx] *=2;
  else if(types[idx]=='t') weight[idx] *=5;
  else if(types[idx]=='d') weight[idx] *=3;
}


__global__ void initParentRank(int* parent, int* rank, int V) {
  int idx =blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>=V) return;
  parent[idx] =idx;
  rank[idx] =0;
}


__global__ void initMinEdge(int2* minEdge, int V) {
  int idx =blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>=V) return;
  minEdge[idx] =make_int2(INT_MAX, -1);
}


__device__ int find(int* parent, int idx) {
  while(parent[idx]!=idx){
    parent[idx] =parent[parent[idx]];
    idx =parent[idx];
  }
  return idx;
}


__global__ void findMinEdges(int* src, int* dest, int* weight, int2* minEdge, int* parent, int V, int E) {
  int idx =blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>=E) return;

  int sr =src[idx],de =dest[idx],w =weight[idx];
  int Sr =find(parent, sr), De =find(parent, de);
  if(Sr==De) return;

  unsigned long long int* ptr =(unsigned long long int*)&minEdge[Sr];
  unsigned long long int currV =*ptr;
  int2 edg =minEdge[Sr];
  while(edg.x>w){
    int2 temp ={w, idx};
    unsigned long long int newVal =*((unsigned long long int*)&temp);
    unsigned long long int preV =atomicCAS(ptr, currV, newVal);
    if(preV==currV) break;
    currV =preV;
    edg =*((int2*)&currV);
  }

  ptr =(unsigned long long int*)&minEdge[De];
  currV =*ptr;
  edg =minEdge[De];
  while(edg.x>w){
    int2 temp ={w, idx};
    unsigned long long int newVal =*((unsigned long long int*)&temp);
    unsigned long long int preV =atomicCAS(ptr, currV, newVal);
    if(preV==currV) break;
    currV =preV;
    edg =*((int2*)&currV);
  }
}


__global__ void mergeComponents(int* src, int* dest, int* weight, int2* minEdge, int* parent, int* rank, long long* mst_weight, int* numComp, int V) {
  int idx =blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>=V || parent[idx]!=idx) return;

  int2 edge =minEdge[idx];
  if(edge.y==-1) return;

  int edgeIdx =edge.y;
  int sr =src[edgeIdx],de =dest[edgeIdx],w =weight[edgeIdx];
  int Sr =find(parent, sr),De =find(parent, de);
  if(Sr==De) return;

  int minN =min(Sr, De),maxN =max(Sr, De);
  if(atomicCAS(&parent[maxN], maxN, minN)==maxN){
    atomicAdd((unsigned long long int*)mst_weight, (unsigned long long int)w);
    atomicSub(numComp, 1);
    if(rank[minN]==rank[maxN])
      atomicAdd(&rank[minN], 1);
  }
}


int main() {
  int V, E;
  cin>>V>>E;

  char type[10];
  int *h_src =(int*)malloc(E * sizeof(int));
  int *h_dest =(int*)malloc(E * sizeof(int));
  int *h_weight =(int*)malloc(E * sizeof(int));
  char *h_types =(char*)malloc(E * sizeof(char));

  for (int i=0;i<E;i++) {
    cin>>h_src[i]>>h_dest[i]>>h_weight[i]>>type;
    h_types[i]=type[0];
  }


  char *d_types;
  int2* d_minEdge;
  long long* d_mst_weight;
  long long mst_weight = 0;
  int threads = 512, blocksV, blocksE;
  blocksV = (V + threads - 1) / threads;
  blocksE = (E + threads - 1) / threads;
  int *d_src, *d_dest, *d_weight, *d_parent, *d_rank, *d_numComp, h_numComp=V;


  cudaMalloc(&d_src, E*sizeof(int));
  cudaMalloc(&d_dest, E*sizeof(int));
  cudaMalloc(&d_rank, V*sizeof(int));
  cudaMalloc(&d_numComp, sizeof(int));
  cudaMalloc(&d_weight, E*sizeof(int));
  cudaMalloc(&d_types, E*sizeof(char));
  cudaMalloc(&d_parent, V*sizeof(int));
  cudaMalloc(&d_minEdge, V*sizeof(int2));
  cudaMalloc(&d_mst_weight, sizeof(long long));
  cudaMemcpy(d_src, h_src, E*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dest, h_dest, E*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight, E*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_types, h_types, E*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_numComp, &h_numComp, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mst_weight, &mst_weight, sizeof(long long), cudaMemcpyHostToDevice);


  auto start = std::chrono::high_resolution_clock::now();


  adjustWeights<<<blocksE, threads>>>(d_weight, d_types, E);
  initParentRank<<<blocksV, threads>>>(d_parent, d_rank, V);
  cudaDeviceSynchronize();
  cudaFree(d_types);

  while (h_numComp>1) {
    initMinEdge<<<blocksV, threads>>>(d_minEdge, V);
    cudaDeviceSynchronize();

    findMinEdges<<<blocksE, threads>>>(d_src, d_dest, d_weight, d_minEdge, d_parent, V, E);
    cudaDeviceSynchronize();

    mergeComponents<<<blocksV, threads>>>(d_src, d_dest, d_weight, d_minEdge, d_parent, d_rank, d_mst_weight, d_numComp, V);

    cudaMemcpy(&h_numComp, d_numComp, sizeof(int), cudaMemcpyDeviceToHost);
  }

  cudaMemcpy(&mst_weight, d_mst_weight, sizeof(long long), cudaMemcpyDeviceToHost);
  cout<<mst_weight%MOD;


  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  //cout<<endl<<elapsed.count()<<" s\n";

  cudaFree(d_src);
  cudaFree(d_dest);
  cudaFree(d_rank);
  cudaFree(d_weight);
  cudaFree(d_parent);
  cudaFree(d_minEdge);
  cudaFree(d_numComp);
  cudaFree(d_mst_weight);

  return 0;
}
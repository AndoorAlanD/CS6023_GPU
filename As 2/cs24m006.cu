#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h> // I have changed the header since it was showing compilation error on Colab.

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void dkernel(long int *mat, long int *fil, long int *ans, int h, int w, int c, int r, int s, int k)
{
    extern __shared__ long int shared_fil[];
    int id=(blockIdx.x*blockDim.x)+threadIdx.x;

    int fil_size= k * c * r * s;
    for (int ii=threadIdx.x;ii<fil_size;ii+=blockDim.x) {
        shared_fil[ii] = fil[ii];
    }
    __syncthreads();

    if(id<(h*w)){
        int row=id/w;
        int col=id%w;
        for(int k1=0;k1<k;k1++){
            int sum=0;
            for(int c1=0;c1<c;c1++){
                for(int r1=0;r1<r;r1++){
                    int i=row-(r/2)+r1;
                    for(int s1=0;s1<s;s1++){
                        int j=col-(s/2)+s1;
                        if(i>=0 && i<h && j>=0 && j<w){
                            sum+=mat[(i*w +j)+(h*w*c1)]*shared_fil[(r1*s+s1)+(c1*r*s)+(k1*c*r*s)];
                        }
                    }
                }
            }
            ans[(row*w +col)+ (k1*h*w)]=sum;
        }
    }
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch
    /****************************************************Start Here***********************************************************/
    long int *mat,*fil,*ans;
    cudaMalloc(&mat, h*w*c*sizeof(long int));
    cudaMalloc(&fil, k*cf*r*s*sizeof(long int));
    cudaMalloc(&ans, k*h*w*sizeof(long int));

    cudaMemcpy(mat,h_mat,h*w*c*sizeof(long int),cudaMemcpyHostToDevice);
    cudaMemcpy(fil,h_filter,k*cf*r*s*sizeof(long int),cudaMemcpyHostToDevice);

    long int x=ceil((h*w*0.1)/102.4);
    dkernel<<<x,1024,(k*c*r*s*sizeof(long int))>>>(mat,fil,ans,h,w,c,r,s,k);

    cudaMemcpy(h_ans,ans,k*h*w*sizeof(long int),cudaMemcpyDeviceToHost);
    cudaFree(mat);
    cudaFree(fil);
    cudaFree(ans);
    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}

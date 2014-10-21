#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

bool verbose=false;
int benchmark=1;
#define LRATE 0.01
#define LPARAM 0.02
#define NUM_FACTORS 32

const float lparam=LPARAM;
const float lrate=LRATE;

double get_walltime(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec/1000000.0);
}

float norm(std::vector<float> &vec);
float norm2(std::vector<float> &vec);
void file_to_map(std::ifstream &f, char delim, std::map<int,std::vector<std::pair<int,int> > > &map, int numlines);
void print_map( std::map<int,std::vector<std::pair<int,int> > > &map);
int get_max_user(std::map<int,std::vector<std::pair<int,int> > > &map);
int get_max_movie(std::map<int,std::vector<std::pair<int,int> > > &map);
int get_num_ratings(std::map<int,std::vector<std::pair<int,int> > > &map);
float predict_rating(std::vector<float> &userFactors, std::vector<float> &movieFactors);
void sgdFactors(std::vector<std::vector<float> > &userFactors, 
        std::vector<std::vector<float> > &movieFactors, std::map<int,std::vector<std::pair<int,int> > > &map);
void sgdFactorsCuda(std::vector<std::vector<float> > &userFactors, 
        std::vector<std::vector<float> > &movieFactors, std::map<int,std::vector<std::pair<int,int> > > &map);
//--------------------------------------------------------------------------CUDA
//__global__ void descendGradient(float *ratings, float *p_u, float *q_i);
__global__ void descendGradient(int num_ratings, float *ratings, float *p_u, float *q_i, float *min);
__device__ void dotNUMFACTORS(int user, int movie, float *p_u, float *q_i, float *res);
__device__ void norm2FACTOR(int vec_indx, float *vec, float *res);

template<class T>
void dump_linear(int nl, std::vector<T> &vec);
std::vector<float> getUserVector(int user, std::map<int,std::vector<std::pair<int,int> > > &map);

int main(int argc,char *argv[]){ 
    std::string input_file_name;
    char delim=' ';
    int numlines=-1;

    //argvars
    if(argc>1){
        int c=1;
        while(c<argc){
            std::string arg=argv[c];
            
            if(arg=="-i" || arg=="--input-file"){
                if(++c<argc){
                    input_file_name=argv[c++];
                    continue;
                }
            }
            if(arg=="-n" || arg=="--size"){
                if(++c<argc){
                    numlines=atoi(argv[c++]);
                    continue;
                }
            }
            if(arg=="-d" || arg=="--delimiter"){
                if(++c<argc){
                    delim=*argv[c++];
                    continue;
                }
            }
            if(arg=="-v" || arg=="--verbose"){
                verbose=true;
                c++;
                continue;
            }
            
            if(arg=="-k" || arg=="--benchmark"){
                benchmark=5;
                c++;
                continue;
            }
        }
    }

    std::ifstream in_file(input_file_name.c_str());
    std::map<int,std::vector<std::pair<int,int> > > ratings;

    file_to_map(in_file, delim, ratings, numlines);
    in_file.close();
    
    std::vector<float> factors(NUM_FACTORS,0.1);
    int max_user=get_max_user(ratings);
    int max_movie=get_max_movie(ratings);
    
    std::cout<<max_user<<std::endl;
    std::cout<<max_movie<<std::endl;
    std::vector<std::vector<float> > userFactors(max_user+1, factors);
    std::vector<std::vector<float> > movieFactors(max_movie+1, factors);
     
    //print_map(ratings);


    double runtime_sum=0;
    for(int b=0; b<benchmark; ++b){
        double timer=get_walltime();
        
        //sgdFactors(userFactors, movieFactors, ratings);
        sgdFactorsCuda(userFactors, movieFactors, ratings);

        timer=get_walltime()-timer;
        runtime_sum+=timer;
    }

    runtime_sum/=benchmark;
    std::cout<<"calculation time averaged over "<<benchmark<< "run(s) is "<<runtime_sum<<" seconds"<<std::endl;

}

//dangerous method with expected input sizes
//todo:fix!
void file_to_map(std::ifstream &f, char delim, std::map<int,std::vector<std::pair<int,int> > > &map, int numlines){
    if(verbose){
        std::cout<<"reading file..."<<std::endl;
    }
    std::string line;
    if(f.is_open()){
        int linenum=0;
        std::vector<int> temp;
        while(std::getline(f,line) && (numlines>linenum || numlines<0)){
            std::stringstream ss(line);
            std::string num;
            while(std::getline(ss, num, delim)){
                temp.push_back(atoi(num.c_str()));
            }
            map[temp[0]].push_back(std::make_pair(temp[1],temp[2]));
            temp.clear();
            linenum++;
        }
        if(verbose){
            std::cout<<"loaded in "<<linenum<<" lines of text"<<std::endl;
            std::cout<<"last line:"<<std::endl<<line<<std::endl;
        }
    }else{
        if(verbose){
            std::cout<<"file could not be opened"<<std::endl;
        }
    }
}

void print_map( std::map<int,std::vector<std::pair<int,int> > > &map){
    std::map<int,std::vector<std::pair<int,int> > >::iterator user=map.begin();
    for(user; user!=map.end(); ++user){
        std::cout<<"user: "<<user->first<<std::endl;
        vector<pair<int, int> >::iterator rating=user->second.begin();
        for(rating; rating<user->second.end(); ++rating){
            std::cout<<"movie"<<rating->first<<", rating: "<<rating->second<<std::endl;
        }
    }
}

void sgdFactors(std::vector<std::vector<float> > &userFactors, 
        std::vector<std::vector<float> > &movieFactors, std::map<int,std::vector<std::pair<int,int> > > &map){
    float min, oldmin=0;
    do{
        oldmin=min;
        min=0;
        std::map<int,std::vector<std::pair<int,int> > >::iterator user=map.begin();
        for(user; user!=map.end(); ++user){
            vector<pair<int, int> >::iterator rating=user->second.begin();
            for(rating; rating<user->second.end(); ++rating){
                float err=rating->second-predict_rating(userFactors[user->first], movieFactors[rating->first]);
                min+=err*err;
                for(int i=0;i<movieFactors[0].size();++i){
                    //float mfv=movieFactors[rating->first][i];
                    movieFactors[rating->first][i]+=lrate*(err*userFactors[user->first][i] - 
                            lparam*movieFactors[rating->first][i]);
                    userFactors[user->first][i]+=lrate*(err*movieFactors[rating->first][i] - 
                            lparam*userFactors[user->first][i]);
                }
                float n_movie=norm2(movieFactors[rating->first]);
                float n_user=norm2(userFactors[rating->first]);
                min+=lparam*(n_movie+n_user);
            }
        }
        std::cout<<"err: "<<min<<std::endl;
    }while(oldmin-min>1e-3 || oldmin==0);
        
    std::map<int,std::vector<std::pair<int,int> > >::iterator user=map.begin();
    for(user; user!=map.end(); ++user){
        vector<pair<int, int> >::iterator rating=user->second.begin();
        for(rating; rating<user->second.end(); ++rating){
            float err=rating->second-predict_rating(userFactors[user->first], movieFactors[rating->first]);
            std::cout<<err<<std::endl;
        }
    }
}

void sgdFactorsCuda(std::vector<std::vector<float> > &userFactors, 
        std::vector<std::vector<float> > &movieFactors, std::map<int,std::vector<std::pair<int,int> > > &map){
        
    float *ratings;
    cudaMalloc((void**)&ratings, 3*get_num_ratings(map)*sizeof(float));
    
    int ratings_copied=0;
    std::map<int,std::vector<std::pair<int,int> > >::iterator user;
    for(user=map.begin();user!=map.end();++user){
        std::vector<float> v=getUserVector(user->first,map);
        cudaMemcpy(ratings+(ratings_copied*sizeof(float)), &v[0],v.size()*sizeof(float), cudaMemcpyHostToDevice);
        ratings_copied+=v.size();
    }
    
    int max_user=get_max_user(map);
    int max_movie=get_max_movie(map);
    float *q_i, *p_u, *min_d;
    cudaMalloc((void**)&p_u, max_user*userFactors.size()*sizeof(float));
    cudaMalloc((void**)&q_i, max_movie*movieFactors.size()*sizeof(float));
    cudaMalloc((void**)&min_d, sizeof(float));
    cudaMemcpy(p_u, &userFactors[0], max_user*userFactors.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(q_i, &movieFactors[0], max_movie*movieFactors.size()*sizeof(float), cudaMemcpyHostToDevice);

    float *min, oldmin=0;
    min=(float*)malloc(1*sizeof(float));
    *min=0;
    do{
        oldmin=*min;
        
        descendGradient<<<128,128>>>(get_num_ratings(map), ratings, p_u, q_i, min_d);

        cudaMemcpy(min, min_d, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout<<"err: "<<*min<<std::endl;
    }while(oldmin-*min>1e-3 || oldmin==0);

    cudaMemcpy(&userFactors[0], p_u, userFactors.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&movieFactors[0], q_i, movieFactors.size()*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(p_u);
    cudaFree(q_i);
    cudaFree(min_d);

    /*
    for(user=map.begin(); user!=map.end(); ++user){
        vector<pair<int, int> >::iterator rating=user->second.begin();
        for(rating; rating<user->second.end(); ++rating){
            float err=rating->second-predict_rating(userFactors[user->first], movieFactors[rating->first]);
            std::cout<<err<<std::endl;
        }
    }
    */
}

__global__ void descendGradient(int num_ratings, float *ratings, float *p_u, float *q_i, float *min){ 
    
    int tid=(blockIdx.x*blockDim.x)+threadIdx.x;

    min[0]=300.0;
    while(tid<num_ratings){
        float predicted;
        int userid=ratings[(tid*3)];
        int movieid=ratings[(tid*3)+1];
        int ratingval=ratings[(tid*3)+2];

        //dotNUMFACTORS(userid, movieid, p_u, q_i, &predicted);
        
        float err=ratingval-predicted;
        *min+=err*err;

        for(int i=0; i<NUM_FACTORS; ++i){
            q_i[(NUM_FACTORS*movieid)+i]+=LRATE*(err*p_u[(NUM_FACTORS*userid)+i] - LPARAM*q_i[(NUM_FACTORS*movieid)+i]);
            p_u[(NUM_FACTORS*userid)+i]+=LRATE*(err*q_i[(NUM_FACTORS*userid)+i] - LPARAM*q_i[(NUM_FACTORS*movieid)+i]);
        }
        float n2_movie, n2_user;
        norm2FACTOR(userid, p_u, &n2_user);
        norm2FACTOR(movieid, q_i, &n2_movie);
        
        *min+=lparam*(n2_movie+n2_user);
        __syncthreads();
        //atomicAdd(d_sum, x[tid]);
        min[0]+=(float)ratingval;
    }
}

float predict_rating(std::vector<float> &userFactors, std::vector<float> &movieFactors){
    float rating=0;
    //assumes same factors size for user and movies
    for(int i=0; i<userFactors.size();++i){
        rating+=userFactors[i]*movieFactors[i];
    }
    return rating;
}

int get_max_user(std::map<int,std::vector<std::pair<int,int> > > &map){
    int max_user=0;
    std::map<int,std::vector<std::pair<int,int> > >::iterator e=map.begin();
    for(e; e!=map.end(); ++e){
        if(e->first>max_user){
            max_user=e->first;
        }
    }
    return max_user;
}

int get_max_movie(std::map<int,std::vector<std::pair<int,int> > > &map){
    int max_movie=0;
    std::map<int,std::vector<std::pair<int,int> > >::iterator e=map.begin();
    for(e; e!=map.end(); ++e){
        vector<pair<int, int> >::iterator mr=e->second.begin();
        for(mr; mr<e->second.end(); ++mr){
            if(mr->first>max_movie){
                max_movie=e->first;
            }
        }
    }
    return max_movie;
}

int get_num_ratings(std::map<int,std::vector<std::pair<int,int> > > &map){
    int num_ratings=0;
    std::map<int,std::vector<std::pair<int,int> > >::iterator e=map.begin();
    for(e; e!=map.end(); ++e){
        vector<pair<int, int> >::iterator mr=e->second.begin();
        for(mr; mr<e->second.end(); ++mr){
            num_ratings++;
        }
    }
    return num_ratings;
}

std::vector<float> getUserVector(int user, std::map<int,std::vector<std::pair<int,int> > > &map){
    std::vector<float> usersRatings;
    vector<pair<int,int> >::iterator rating=map[user].begin();
    for(rating;rating<map[user].end();++rating){
        usersRatings.push_back(user);
        usersRatings.push_back(rating->first);
        usersRatings.push_back(rating->second);
    }
    return usersRatings;
}

float norm(std::vector<float> &vec){
    float sum=0;
    for(int i=0; i<vec.size(); ++i){
        sum+=vec[i]*vec[i];
    }
    return std::sqrt(sum);
}

float norm2(std::vector<float> &vec){
    float sum=0;
    for(int i=0; i<vec.size(); ++i){
        sum+=vec[i]*vec[i];
    }
    return sum;
}

template<class T>
void dump_linear(int nl, std::vector<T> &vec){
    int n=0;
    for(int i=0;i<vec.size();++i){
        ++n;
        std::cout<<vec[i]<<' ';
        if(n>=nl && nl>0){
            std::cout<<std::endl;
            n=0;
        }
    }
    std::cout<<std::endl;
}

__device__ void dotNUMFACTORS(int user, int movie, float *p_u, float *q_i, float *res){
    
    
    __shared__ float cache[NUM_FACTORS];
    
    int tid=(blockIdx.x*blockDim.x)+threadIdx.x;
    int cache_index=threadIdx.x;
    
    user=0;
    movie=0;
    float temp=0;

    while(tid<NUM_FACTORS){
        temp+=p_u[(user*NUM_FACTORS)+tid]*q_i[(movie*NUM_FACTORS)+tid];
        tid+=blockDim.x*gridDim.x;//should be unnecessary, since block sizes must be multiple of 32, which is the #factors
    }

    cache[cache_index]=temp;
    __syncthreads();
    
    int i=blockDim.x/2;
    while(i!=0){
        if(cache_index<i){
            cache[cache_index]+=cache[cache_index+i];
        }
        __syncthreads();
        i/=2;
    }

    if(cache_index==0){
        *res=cache[0];
    }
}

__device__ void norm2FACTOR(int vec_indx, float *vec, float *res){
    __shared__ float cache[NUM_FACTORS];
    
    int tid=(blockIdx.x*blockDim.x)+threadIdx.x;
    int cache_index=threadIdx.x;

    float temp=0;

    while(tid<NUM_FACTORS){
        temp+=vec[(vec_indx*NUM_FACTORS)+tid]*vec[(vec_indx*NUM_FACTORS)+tid];
        tid+=blockDim.x*gridDim.x;//should be unnecessary, since block sizes must be multiple of 32, which is the #factors
    }

    cache[cache_index]=temp;
    __syncthreads();
    
    int i=blockDim.x/2;
    while(i!=0){
        if(cache_index<i){
            cache[cache_index]+=cache[cache_index+i];
        }
        __syncthreads();
        i/=2;
    }

    if(cache_index==0){
        *res=cache[0];
    }
}



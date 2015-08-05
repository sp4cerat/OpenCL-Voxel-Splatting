#include <vector>

#include "RLE4.h"
#include "Core.h"
#include "VecMath.h"

struct CUBE256Instance
{
	vec3f pos,rot;
	int fragments;
	int cubeid;
};
struct CUBE256
{
	vec3f pos,rot;

	struct MipMap
	{
		int handle;
		std::vector<uint> voxels; // x y<<8 z<<16 color<<24
	};
	std::vector<MipMap> mipmaps;

	void set_voxel(int mipmap,uint x,uint y,uint z,uchar4 color)
	{
		if(x>255) return;if(x<0) return;
		if(y>255) return;if(x<0) return;
		if(z>255) return;if(x<0) return;
		if(mipmap>=mipmaps.size()) mipmaps.resize(mipmap+1);
		mipmaps[mipmap].voxels.push_back( uchar4(x,y,z,color.x).to_uint() );
	}
};
int cubes_x = 4; // 4*256
int cubes_y = 4;
int cubes_z = 4;
std::vector<CUBE256> cubes;
//std::vector<CUBE256Instance> cubeinstances;

void init_cubes()
{
	cubes.resize(cubes_x*cubes_y*cubes_z);
	
	int c=0;
	loopk(0,cubes_z)
	loopj(0,cubes_y)
	loopi(0,cubes_x)
	{
		cubes[c++].pos=vec3f(i,j,k);
	}
}
void set_voxel(int mipmap,uint x,uint y,uint z,uchar4 color)
{
	if(x>=cubes_x*256)return;
	if(y>=cubes_y*256)return;
	if(z>=cubes_z*256)return;
	int id=(x>>8)+(y>>8)*cubes_x+(z>>8)*cubes_x*cubes_y;
	cubes[id].set_voxel(mipmap,x&255,y&255,z&255,color);
}
cl_kernel cl_kernel_splat;
cl_kernel cl_kernel_postx;
cl_kernel cl_kernel_posty;
cl_kernel cl_kernel_colorize;

cl_mem data_gpu;
std::vector<int> data_cpu;

void init_octree_kernel()
{
	////////////////////////////////////////////
	static bool init=true;
	if(!init) return;
	init=false;
	////////////////////////////////////////////
	init_cubes();

	RLE4 rle;
	rle.load("../data/spherei.rle4");

	int voxelcount=0;

	loopi(0,10)
		data_cpu.push_back(0);

	//mipmap
	loopm(0,10)
	{
		int cubecount=0;
		int ofs=data_cpu.size();
		data_cpu[m]=ofs;
		data_cpu.push_back(0);
		loopi(0,cubes_x)
		loopj(0,cubes_y)
		loopk(0,cubes_z)
		{
			int cb=i+j*cubes_x+k*cubes_x*cubes_y;
			if(cubes[cb].mipmaps.size()>m)
			{
				cubecount++;
				int size=cubes[cb].mipmaps[m].voxels.size();
				//printf("voxel count %d %d %d : %d\n",i,j,k,size);

				data_cpu.push_back(i*256);
				data_cpu.push_back(j*256);
				data_cpu.push_back(k*256);
				data_cpu.push_back(size);
				if(m==0)voxelcount+=size;

				if(size>0)
				loopl(0,size)
					data_cpu.push_back(cubes[cb].mipmaps[m].voxels[l]);
			}
		}
		data_cpu[ofs]=cubecount;
		//printf("mipmap[%d] cbe count %d\n",m,cubecount);
	}

	int cubecount=0;

	
	printf("total size:%d MB %d voxels\n",
		(data_cpu.size()*4)/(1024*1024),data_cpu.size());

	data_gpu = clCreateBuffer(cxGPUContext, 
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		data_cpu.size()*4, 
		&data_cpu[0], 
		&ciErrNum);
	
    // create the kernel
    cl_kernel_splat = clCreateKernel(cpProgram, "splatting", &ciErrNum);
    cl_kernel_postx = clCreateKernel(cpProgram, "postx", &ciErrNum);
    cl_kernel_posty = clCreateKernel(cpProgram, "posty", &ciErrNum);
    cl_kernel_colorize = clCreateKernel(cpProgram, "colorize", &ciErrNum);

    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

	//mciSendString("open \"..\\1.mp3\" type MPEGVideo alias song1", NULL, 0, 0); 
	//mciSendString("play song1", NULL, 0, 0);
	//mciSendString("setaudio song1 volume to 1000", 0, 0, 0);
	//mciSendString("close song1", NULL, 0, 0);
}

void init_octree()
{
	screen.posx = 312;
	screen.posy = 300;
	screen.posz = 0;

	wglSwapIntervalEXT(0);

}

void exec_splatting()
{	
	// animate
	int t=(timeGetTime()>>3)&2047;
	screen.posy=-500-sin(float(t)*M_PI/1023)*50;
	screen.posx=sin(float(t)*2*M_PI/1023)*50-500;
	screen.posz=sin(float(t)*3*M_PI/1023)*50+200;

	wglSwapIntervalEXT(0);
	static matrix44 m;
	m.ident();
	m.translate(vector3(screen.posx,screen.posy,screen.posz));
	m.rotate_z(-screen.rotz );
	m.rotate_x(screen.rotx );
	m.rotate_y(screen.roty );
	m.transpose();

	int res_x=image_width;
	int res_y=image_height;

	int arg=0;	
	ciErrNum |= clSetKernelArg(cl_kernel_splat, arg++, sizeof(cl_mem), (void *) &(cl_pbos[1]));//dest pbo
	ciErrNum |= clSetKernelArg(cl_kernel_splat, arg++, sizeof(cl_mem), (void *) &(cl_pbos[0]));//src pbo
	ciErrNum |= clSetKernelArg(cl_kernel_splat, arg++, sizeof(cl_mem), (void *) &(data_gpu)); // octree
	ciErrNum |= clSetKernelArg(cl_kernel_splat, arg++, sizeof(cl_int)  , &res_x);
	ciErrNum |= clSetKernelArg(cl_kernel_splat, arg++, sizeof(cl_int)  , &res_y);
	ciErrNum |= clSetKernelArg(cl_kernel_splat, arg++, sizeof(cl_float)*4, &m.m[0]);
	ciErrNum |= clSetKernelArg(cl_kernel_splat, arg++, sizeof(cl_float)*4, &m.m[1]);
	ciErrNum |= clSetKernelArg(cl_kernel_splat, arg++, sizeof(cl_float)*4, &m.m[2]);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

	szLocalWorkSize[0] = 256;
    szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], 8192);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, cl_kernel_splat, 1, NULL,
                                      szGlobalWorkSize, szLocalWorkSize, 
									  0, NULL,NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

	clFlush (cqCommandQueue);

	arg=0;
	ciErrNum |= clSetKernelArg(cl_kernel_postx, arg++, sizeof(cl_mem), (void *) &(cl_pbos[1]));//dest pbo
	ciErrNum |= clSetKernelArg(cl_kernel_postx, arg++, sizeof(cl_mem), (void *) &(cl_pbos[0]));//dest pbo
	ciErrNum |= clSetKernelArg(cl_kernel_postx, arg++, sizeof(cl_int), &res_x);
	ciErrNum |= clSetKernelArg(cl_kernel_postx, arg++, sizeof(cl_int), &res_y);

	loopi(0,4)
	{
		ciErrNum |= clSetKernelArg(cl_kernel_postx, 4, sizeof(cl_int), &i); // iteration

		//oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		szLocalWorkSize[0] = 16;
		szLocalWorkSize[1] = 16;
		szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], res_x>>(1+i));
		szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], res_y>>(1+i));
		ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, cl_kernel_postx, 2, NULL,
										  szGlobalWorkSize, szLocalWorkSize, 
										  0, NULL,NULL);
		//oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);		
		clFlush (cqCommandQueue);
	}

	arg=0;
	ciErrNum |= clSetKernelArg(cl_kernel_posty, arg++, sizeof(cl_mem), (void *) &(cl_pbos[1]));//dest pbo
	ciErrNum |= clSetKernelArg(cl_kernel_posty, arg++, sizeof(cl_mem), (void *) &(cl_pbos[0]));//dest pbo
	ciErrNum |= clSetKernelArg(cl_kernel_posty, arg++, sizeof(cl_int), &res_x);
	ciErrNum |= clSetKernelArg(cl_kernel_posty, arg++, sizeof(cl_int), &res_y);

	for(int i=0;i>=0;i--)
	{
		ciErrNum |= clSetKernelArg(cl_kernel_posty, 4, sizeof(cl_int), &i); // iteration

		//oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		szLocalWorkSize[0] = 16;
		szLocalWorkSize[1] = 16;
		szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], res_x>>(i));
		szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], res_y>>(i));
		ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, cl_kernel_posty, 2, NULL,
										  szGlobalWorkSize, szLocalWorkSize, 
										  0, NULL,NULL);
		clFlush (cqCommandQueue);
	}	
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);		
	
	arg=0;
	ciErrNum |= clSetKernelArg(cl_kernel_colorize, arg++, sizeof(cl_mem), (void *) &(cl_pbos[1]));//dest pbo
	ciErrNum |= clSetKernelArg(cl_kernel_colorize, arg++, sizeof(cl_int)  , &res_x);
	ciErrNum |= clSetKernelArg(cl_kernel_colorize, arg++, sizeof(cl_int)  , &res_y);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	szLocalWorkSize[0] = 16;
	szLocalWorkSize[1] = 16;
    szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], res_x);
    szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], res_y);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, cl_kernel_colorize, 2, NULL,
                                      szGlobalWorkSize, szLocalWorkSize, 
									  0, NULL,NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	clFlush (cqCommandQueue);
	
	//clWaitForEvents(1, &event);
	//printf("%f\n",screen.roty);
	
}

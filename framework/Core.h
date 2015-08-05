#pragma once
////////////////////////////////////////////////////////////////////////////////
#define SCREEN_SIZE_X 1024
#define SCREEN_SIZE_Y 768
#define RENDER_SIZE 1024
#define THREAD_COUNT_X 16 //16
#define THREAD_COUNT_Y 16

#define OCTREE_DEPTH 10
#define OCTREE_DEPTH_AND ((1<<OCTREE_DEPTH)-1)

#define PERSISTENT_LG_X 0
#define PERSISTENT_LG_Y 0

#define PERSISTENT_X (1<<PERSISTENT_LG_X)
#define PERSISTENT_Y (1<<PERSISTENT_LG_Y)

//#define COARSE_OFFSET (0*RENDER_SIZE*SCREEN_SIZE_X)
#define COARSE_SCALE 4

//#define MAX_LOD 7
#define VIEW_DIST_MAX 40000
//#define LOD_ADJUST 2
//#define LOD_ADJUST_COARSE 2
#define LOD_ADJUST_COARSE 2 
#define LOD_ADJUST 1
//15/8
#define SCENE_LOOP 0
#define XOR_BREAK xory  //xory
////////////////////////////////////////////////////////////////////////////////
#define _USE_MATH_DEFINES
#define loopi(start_l,end_l) for ( int i=start_l;i<end_l;++i )
#define loopj(start_l,end_l) for ( int j=start_l;j<end_l;++j )
#define loopk(start_l,end_l) for ( int k=start_l;k<end_l;++k )
#define loopm(start_l,end_l) for ( int m=start_l;m<end_l;++m )
#define loopl(start_l,end_l) for ( int l=start_l;l<end_l;++l )
#define loopn(start_l,end_l) for ( int n=start_l;n<end_l;++n )
#define loop(var_l,start_l,end_l) for ( int var_l=start_l;var_l<end_l;++var_l )
#define loops(a_l,start_l,end_l,step_l) for ( a_l = start_l;a_l<end_l;a_l+=step_l )

#ifndef byte
#define byte unsigned char
#endif

#ifndef ushort
#define ushort unsigned short
#endif

#ifndef uint
#define uint unsigned int
#endif

#ifndef uchar
#define uchar unsigned char
#endif
/*
struct uchar4 
{
	unsigned char x,y,z,w;
	
	unsigned int to_uint()
	{
		return *((unsigned int*)this);
	};
};
*/
/*
class uchar4 
{ 
	public:

	uchar x,y,z,w;

	uchar4(){};
	uchar4(uint x,uint y,uint z,uint w)
	{
		this->x=x;
		this->y=y;
		this->z=z;
		this->w=w;
	}
	
	uint to_uint()
	{
		return *((unsigned int*)this);
	};
};
*/
////////////////////////////////////////////////////////////////////////////////
class Keyboard
{
	public:

	bool  key [256]; // actual
	bool  key2[256]; // before

	Keyboard(){ int a; loop(a,0,256) key[a] = key2[a]=0; }

	bool KeyDn(char a)//key down
	{
		return key[a];
	}
	bool KeyPr(char a)//pressed
	{
		return ((!key2[a]) && key[a] );
	}
	bool KeyUp(char a)//released
	{
		return ((!key[a]) && key2[a] );
	}
	void update()
	{
		int a;loop( a,0,256 ) key2[a] = key[a];
	}
};
////////////////////////////////////////////////////////////////////////////////
class Mouse
{
	public:

	bool  button[256];
	bool  button2[256];
	float mouseX,mouseY;
	float mouseDX,mouseDY;

	Mouse()
	{ 
		int a; loop(a,0,256) button[a] = button2[a]=0; 
		mouseX=mouseY=mouseDX=mouseDY= 0;
	}
	void update()
	{
		int a;loop( a,0,256 ) button2[a] = button[a];
	}
};
////////////////////////////////////////////////////////////////////////////////
class Screen
{
	public:

	int	 window_width;
	int	 window_height;
	bool fullscreen;

	float posx,posy,posz;
	float rotx,roty,rotz;
};
////////////////////////////////////////////////////////////////////////////////
extern Keyboard		keyboard;
extern Mouse		mouse;
extern Screen		screen;
////////////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus
extern "C" {
#endif 
////////////////////////////////////////////////////////////////////////////////
extern void	cpu_memcpy(void* dst, void* src, int count);
extern void	gpu_memcpy(void* dst, void* src, int count);
extern void*	gpu_malloc(int size);
extern int		cpu_to_gpu_delta;
////////////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus
}
#endif 
////////////////////////////////////////////////////////////////////////////////
void Core_ToggleFullscreen();
void Core_createPBO( uint* pbo,int image_width , int image_height,int bpp);
void Core_createTexture( uint* tex_name, unsigned int size_x, unsigned int size_y,int bpp);
void Core_Init(int window_width, int window_height, bool fullscreen,void (*display_func)(void));
////////////////////////////////////////////////////////////////////////////////

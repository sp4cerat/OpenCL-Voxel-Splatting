#pragma once

#define _USE_MATH_DEFINES
#define loopi(start_l,end_l) for ( int i=start_l;i<end_l;++i )

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
		return *((uint*)this);
	};
};

// define this function in your code
extern void set_voxel(int  mipmap,uint x,uint y,uint z,uchar4 color);

struct Map4
{
	int    sx,sy,sz,slabs_size;
	uint   *map_mem;
	uint   *map;
	ushort *slabs;
};

struct RLE4
{	
	/*------------------------------------------------------*/
	Map4 map [16];int nummaps;	
	/*------------------------------------------------------*/
	void load(char *filename);
	/*------------------------------------------------------*/
};

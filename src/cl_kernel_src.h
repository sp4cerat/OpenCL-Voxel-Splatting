char cl_kernel_src[]={"\n"\
"#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"\
"#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n"\
"\n"\
"#define loopi(start_l,end_l) for ( int i=start_l;i<end_l;++i )\n"\
"#define loopj(start_l,end_l) for ( int j=start_l;j<end_l;++j )\n"\
"#define loopk(start_l,end_l) for ( int k=start_l;k<end_l;++k )\n"\
"#define loopm(start_l,end_l) for ( int m=start_l;m<end_l;++m )\n"\
"#define loopl(start_l,end_l) for ( int l=start_l;l<end_l;++l )\n"\
"#define loopn(start_l,end_l) for ( int n=start_l;n<end_l;++n )\n"\
"\n"\
"__kernel    void splatting(\n"\
"			__global uint *p_screenBuffer,\n"\
"			__global uint *p_backBuffer,\n"\
"			__global uint *p_data,\n"\
"			int p_width, \n"\
"			int p_height,\n"\
"			float4 matx,\n"\
"			float4 maty,\n"\
"			float4 matz\n"\
"			)\n"\
"{\n"\
"	const int tx = get_local_id(0);\n"\
"	const int ty = get_local_id(1);\n"\
"	const int bw = get_local_size(0);\n"\
"	const int bh = get_local_size(1);\n"\
"	const int idx = get_global_id(0);\n"\
"	const int idy = get_global_id(1);\n"\
"	\n"\
"	loopj(- 10, 10)\n"\
"	loopk(   0, 20)\n"\
"	{\n"\
"		int addx=j*1024;\n"\
"		int addy= 0;\n"\
"		int addz=k*1024;\n"\
"\n"\
"		float4 p,pt;\n"\
"		p.x=addx+512;\n"\
"		p.y=addy+512;\n"\
"		p.z=addz+512;\n"\
"		p.w=1;\n"\
"		pt.x=dot(p,matx);\n"\
"		pt.y=dot(p,maty);\n"\
"		pt.z=dot(p,matz);\n"\
"		\n"\
"		if(pt.z<-600)continue;\n"\
"		\n"\
"		int sx=pt.x*p_width/pt.z+p_width/2;\n"\
"		if(sx >=  p_width*2) continue;\n"\
"		if(sx <  -p_width) continue;\n"\
"\n"\
"		int sy=pt.y*p_width/pt.z+p_height/2;\n"\
"		if(sy >=  p_height*2) continue;\n"\
"		if(sy <  -p_height) continue;\n"\
"				\n"\
"		int level=log2((pt.z*2  )/p_width);\n"\
"\n"\
"		if(level<0)level=0;\n"\
"		if(level>=9)continue;\n"\
"\n"\
"		int m=p_data[level];\n"\
"		int num_cubes=p_data[m];m++;\n"\
"\n"\
"		loopi(0,num_cubes)\n"\
"		{\n"\
"			int cx=(p_data[m]<<level)+addx;m++;\n"\
"			int cy=(p_data[m]<<level)+addy;m++;\n"\
"			int cz=(p_data[m]<<level)+addz;m++;\n"\
"			/*\n"\
"			int out_of_view=0;\n"\
"			if(0)\n"\
"			{\n"\
"				loopi(0,2)\n"\
"				loopj(0,2)\n"\
"				loopk(0,2)\n"\
"				{\n"\
"					int x=cx+i*256;\n"\
"					int y=cy+j*256;\n"\
"					int z=cz+k*256;\n"\
"\n"\
"					float4 p,pt;\n"\
"					p.x=x;p.y=y;p.z=z;p.w=1;\n"\
"					pt.x=dot(p,matx);\n"\
"					pt.y=dot(p,maty);\n"\
"					pt.z=dot(p,matz);\n"\
"\n"\
"					if(pt.z<0){ out_of_view++;continue;}\n"\
"				\n"\
"					int sx=pt.x*p_width/pt.z+p_width/2;\n"\
"					if(sx >= p_width) { out_of_view++;continue;}\n"\
"					if(sx <  0) { out_of_view++;continue;}\n"\
"\n"\
"					int sy=pt.y*p_width/pt.z+p_height/2;\n"\
"					if(sy >= p_height) { out_of_view++;continue;}\n"\
"					if(sy <  0) { out_of_view++;continue;}\n"\
"				}\n"\
"			}\n"\
"			*/\n"\
"			\n"\
"\n"\
"			int cc=p_data[m];m++;\n"\
"			int cd=m+idx; // cube data offset start\n"\
"			int ce=m+cc;  // cube data offset end\n"\
"			m+=cc; // voxel count\n"\
"\n"\
"			//if(out_of_view==8)continue;\n"\
"\n"\
"			int pack=0; int n=0;\n"\
"\n"\
"			while(cd<ce)\n"\
"			{	\n"\
"				pack=p_data[cd];\n"\
"				cd+=8192; \n"\
"\n"\
"				float4 p;\n"\
"				p.x=cx+((pack&0xff)<<level);\n"\
"				p.y=cy+(((pack>>8)&0xff)<<level);\n"\
"				p.z=cz+(((pack>>16)&0xff)<<level);\n"\
"				p.w=1;//p.x+=sin((p.y+maty.w)/100)*100;p.z+=cos((p.y+matx.w)/100)*100;\n"\
"\n"\
"				float4 pt;\n"\
"				pt.z=dot(p,matz);\n"\
"				if(pt.z<1) continue;\n"\
"				if(pt.z>65000 ) continue;\n"\
"\n"\
"				pt.x=dot(p,matx);\n"\
"				int sx=pt.x*p_width/pt.z+p_width/2;\n"\
"				if(sx >= p_width) continue;\n"\
"				if(sx <  0) continue;\n"\
"\n"\
"				pt.y=dot(p,maty);\n"\
"				int sy=pt.y*p_width/pt.z+p_height/2;\n"\
"				if(sy >= p_height) continue;\n"\
"				if(sy <  0) continue;\n"\
"\n"\
"				int sz=pt.z; \n"\
"				int ofs=sy*p_width + sx;\n"\
"				sz<<=8;\n"\
"								\n"\
"				if((p_screenBuffer[ofs]&0xffff00) <= sz) continue;\n"\
"\n"\
"				int ic=     ((pack>>24)&0xff);//color\n"\
"				atom_min(&p_screenBuffer[ofs],sz+ic);\n"\
"			}\n"\
"		}\n"\
"	}\n"\
"}\n"\
"__kernel    void postx(\n"\
"			__global uint *p_screenBuffer,\n"\
"			__global uint *p_backBuffer,\n"\
"			int p_width, \n"\
"			int p_height,\n"\
"			int p_iter\n"\
"			)\n"\
"{\n"\
"	//__local uint smem[4000];\n"\
"	//const int tid = get_local_id(0);\n"\
"	const int x   = get_global_id(0);\n"\
"	const int y   = get_global_id(1);\n"\
"	if( x >= (p_width  >>p_iter)) return;\n"\
"	if( y >= (p_height >>p_iter)) return;\n"\
"	\n"\
"	int of_read    = x*2+y*2*p_width+(p_width>>(p_iter));\n"\
"\n"\
"	if(p_iter==0)of_read-=p_width;\n"\
"\n"\
"	int of_xy_read = of_read+(p_height/2)*p_width;\n"\
"	int of_write   = x   +y*  p_width+(p_width>>(p_iter+1));\n"\
"	int of_xy_write= of_write+(p_height/2)*p_width;\n"\
"\n"\
"	int a1,a2,a3,a4;\n"\
"\n"\
"	if(p_iter>0)\n"\
"	{\n"\
"		a1=p_backBuffer[of_read];\n"\
"		a2=p_backBuffer[of_read+1];\n"\
"		a3=p_backBuffer[of_read+p_width];\n"\
"		a4=p_backBuffer[of_read+1+p_width];\n"\
"	}\n"\
"	else\n"\
"	{\n"\
"		a1=p_screenBuffer[of_read];\n"\
"		a2=p_screenBuffer[of_read+1];\n"\
"		a3=p_screenBuffer[of_read+p_width];\n"\
"		a4=p_screenBuffer[of_read+1+p_width];\n"\
"	}\n"\
"\n"\
"	\n"\
"	int xy=0;\n"\
"\n"\
"	if(p_iter==0)\n"\
"	{\n"\
"		int xo=0,yo=0;\n"\
"		if(a2<a1){ a1=a2;xo=1;yo=0; }\n"\
"		if(a3<a1){ a1=a3;xo=0;yo=1; }\n"\
"		if(a4<a1){ a1=a4;xo=1;yo=1; }\n"\
"		xo+=x*2;\n"\
"		yo+=y*2;\n"\
"		int scale=(a1>712*512)?1000:500;\n"\
"		int radius=(p_width*scale/((a1&0xffff00)+2));\n"\
"		if(radius>255)radius=255;\n"\
"		xy=(xo<<8)+(yo<<20)+radius;\n"\
"	}\n"\
"	else\n"\
"	{	\n"\
"		int ofs=0;\n"\
"		if(a2<a1){ a1=a2;ofs=1; }\n"\
"		if(a3<a1){ a1=a3;ofs=p_width; }\n"\
"		if(a4<a1){ a1=a4;ofs=1+p_width; }\n"\
"\n"\
"		xy=p_backBuffer[of_xy_read+ofs];\n"\
"\n"\
"		/*\n"\
"		int cmpx=(xy>>8)&4095;\n"\
"		int cmpy= xy>>20;\n"\
"		int cmpz= a1&0xffff00;\n"\
"		int cmpc= a1&255;\n"\
"		int cmpr= xy&255;\n"\
"\n"\
"		int sum=0;\n"\
"		int xsum=0;\n"\
"		int ysum=0;\n"\
"		int zsum=0;\n"\
"		int csum=0;\n"\
"		int rsum=0;\n"\
"\n"\
"		loopi(0,1)\n"\
"		loopj(0,1)\n"\
"		{\n"\
"			int z=p_backBuffer[of_read+i+j*p_width];\n"\
"			int c=z&255;\n"\
"			z&=0xffff00;\n"\
"			int rxy=p_backBuffer[of_xy_read++i+j*p_width];\n"\
"\n"\
"		}\n"\
"		*/\n"\
"\n"\
"		/*\n"\
"		int xy2=p_backBuffer[of_xy_write];\n"\
"		x2=(xy2>>8)&4095;\n"\
"		y2= xy2>>20;\n"\
"		z2=p_backBuffer[of_write];\n"\
"		r2= xy2 &255;\n"\
"		*/\n"\
"	}\n"\
"	p_backBuffer[of_write]=a1;\n"\
"	p_backBuffer[of_xy_write]=xy;\n"\
"}\n"\
"\n"\
"__kernel    void posty(\n"\
"			__global uint *p_screenBuffer,\n"\
"			__global uint *p_backBuffer,\n"\
"			int p_width, \n"\
"			int p_height,\n"\
"			int p_iter\n"\
"			)\n"\
"{\n"\
"	//local uint smem[4000];\n"\
"	//const int tid = get_local_id(0);\n"\
"	const int x   = get_global_id(0);\n"\
"	const int y   = get_global_id(1);\n"\
"	if( x >= (p_width  >>1)) return;\n"\
"	if( x >= (p_width  >>p_iter)) return;\n"\
"	if( y >= (p_height >>p_iter)) return;\n"\
"\n"\
"	int x_max=p_width  >>p_iter;\n"\
"	int y_max=p_height  >>p_iter;\n"\
"\n"\
"	int xy_of=(p_height/2)*p_width;\n"\
"	\n"\
"	int of_read    = (x/2)+(y/2)*p_width+(p_width>>(p_iter+1));\n"\
"\n"\
"	int of_xy_read = of_read+xy_of;\n"\
"\n"\
"	int of_write   =  x   + y *  p_width+(p_width>>p_iter);\n"\
"\n"\
"	if(p_iter==0) of_write-=p_width;\n"\
"\n"\
"	int of_xy_write= of_write+xy_of;\n"\
"\n"\
"	int xy2,x2,y2,z2,r2;\n"\
"\n"\
"	if(p_iter==0)\n"\
"	{\n"\
"		x2=x;\n"\
"		y2=y;\n"\
"		z2=p_screenBuffer[of_write];\n"\
"		int scale=(z2>512*512)?800:500;\n"\
"		r2=(p_width*scale/((z2&0xffff00)+2));\n"\
"	}\n"\
"	else\n"\
"	{\n"\
"		xy2=p_backBuffer[of_xy_write];\n"\
"		x2=(xy2>>8)&4095;\n"\
"		y2= xy2>>20;\n"\
"		z2=p_backBuffer[of_write];\n"\
"		r2= xy2 &255;\n"\
"	}\n"\
"	\n"\
"	int px=x<<p_iter;\n"\
"	int py=y<<p_iter;\n"\
"	if(p_iter>0) \n"\
"	{ \n"\
"		px+=1<<(p_iter-1);\n"\
"		py+=1<<(p_iter-1);\n"\
"	} \n"\
"\n"\
"//	if (abs(px-x1)+abs(py-y1) <= r1)\n"\
"//	float r= sqrt( convert_float( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) );\n"\
"\n"\
"	int zmin=0xffffff;\n"\
"\n"\
"	float sumx=0;\n"\
"	float sumy=0;\n"\
"	float sumz=0;\n"\
"	float sumc=0;\n"\
"	float sumr=0;\n"\
"	float sum=0;\n"\
"	int   sumcc=0;\n"\
"\n"\
"	loopi(-2,3) if(i+x/2>=0) if(i+x/2<x_max)\n"\
"	loopj(-2,3) if(j+y/2>=0) if(j+y/2<y_max)\n"\
"	{\n"\
"		int add=i + j*p_width;\n"\
"		int xy1=p_backBuffer[of_xy_read + add ];\n"\
"		int x1=(xy1>>8)&4095;\n"\
"		int y1= xy1>>20;\n"\
"		int z1=p_backBuffer[of_read + add ];\n"\
"		int r1= xy1 &255;\n"\
"	\n"\
"		float r= sqrt( convert_float( (x1-px)*(x1-px) + (y1-py)*(y1-py)) );\n"\
"		//(abs(x1-px)+abs(y1-py));//\n"\
"\n"\
"		if(r1>r)\n"\
"		if((z1&0xffff00)<zmin)\n"\
"		{\n"\
"			zmin=z1&0xffff00;\n"\
"			sumcc=z1&3;\n"\
"\n"\
"			//sumc=z1&0xff;\n"\
"		}\n"\
"	}\n"\
"\n"\
"	loopi(-2,3) if(i+x/2>=0) if(i+x/2<x_max)\n"\
"	loopj(-2,3) if(j+y/2>=0) if(j+y/2<y_max)\n"\
"	{\n"\
"		int add=i + j*p_width;\n"\
"		int xy1=p_backBuffer[of_xy_read + add ];\n"\
"		int x1=(xy1>>8)&4095;\n"\
"		int y1= xy1>>20;\n"\
"		int z1=p_backBuffer[of_read + add ];\n"\
"		int r1= xy1 &255;\n"\
"	\n"\
"		float r= sqrt( convert_float( (x1-px)*(x1-px) + (y1-py)*(y1-py)) );\n"\
"		//(abs(x1-px)+abs(y1-py));//\n"\
"\n"\
"		if(sumc==255) sumc=(z1&0xff);\n"\
"\n"\
"		if(r1>r)\n"\
"		if(abs((z1&0xffff00)-zmin)<(4*256))\n"\
"		{\n"\
"			float cmpr=r1;\n"\
"			float w=(fabs(r-cmpr));;//2;//-fabs(r-r1);//100/(10+fabs(r-r1));\n"\
"			w*=w;\n"\
"			w+=abs((z1&0xffff00)-zmin)/256;\n"\
"			w+=1/w;\n"\
"			sumx+=w*convert_float(x1);\n"\
"			sumy+=w*convert_float(y1);\n"\
"			sumz+=w*convert_float(z1&0xffff00);\n"\
"			sumr+=w*convert_float(r1);\n"\
"			sumc+=w*convert_float(z1&255);\n"\
"			sum+=w;\n"\
"		}\n"\
"	}\n"\
"	if(sum>0)\n"\
"	{\n"\
"		sumz/=sum;\n"\
"		sumx/=sum;\n"\
"		sumy/=sum;\n"\
"		sumc/=sum;\n"\
"		sumr/=sum;\n"\
"		sum=1;\n"\
"\n"\
"		if(abs((z2&0xffff00)-(convert_int(sumz)&0xffff00))<(4*256))\n"\
"		{\n"\
"			float w=1;//0.5;\n"\
"			sumx+=w*convert_float(x2);\n"\
"			sumy+=w*convert_float(y2);\n"\
"			sumz+=w*convert_float(z2&0xffff00);\n"\
"			sumc+=w*convert_float(z2&0xff);\n"\
"			sumcc=z2&3;\n"\
"			sumr+=w*convert_float(r2);\n"\
"			sum+=w;\n"\
"		}\n"\
"		sumz/=sum;\n"\
"		sumx/=sum;\n"\
"		sumy/=sum;\n"\
"		sumc/=sum;\n"\
"		sumr/=sum;\n"\
"\n"\
"		z2=(convert_int(sumz)&0xffff00)+(convert_int(sumc)&0xfc)+(sumcc&3);\n"\
"		xy2=convert_int(sumr)+(convert_int(sumx)<<8)+(convert_int(sumy)<<20);\n"\
"	}\n"\
"\n"\
"\n"\
"	if(p_iter==0)\n"\
"	{\n"\
"		p_screenBuffer[of_write]=z2;\n"\
"		//p_screenBuffer[of_xy_write]=xy2;\n"\
"	}\n"\
"	else\n"\
"	{\n"\
"		p_backBuffer[of_write]=z2;\n"\
"		p_backBuffer[of_xy_write]=xy2;\n"\
"	}\n"\
"}\n"\
"\n"\
"__kernel    void colorize(\n"\
"			__global uint *p_screenBuffer,\n"\
"			int p_width, \n"\
"			int p_height\n"\
"			)\n"\
"{\n"\
"	const int x = get_global_id(0);\n"\
"	const int y = get_global_id(1);\n"\
"	if( x >= p_width ) return;\n"\
"	if( y >= p_height ) return;\n"\
"\n"\
"	if( x >= p_width*3/4 )\n"\
"	{\n"\
"		return;\n"\
"	}\n"\
"\n"\
"	int of=x+y*p_width;\n"\
"	int a=p_screenBuffer[of];//&0xff;\n"\
"\n"\
"	float3 color_tab[4]={\n"\
"		{0.8,1.0,0.3},\n"\
"		{1.0,0.7,0.3},\n"\
"		{1.5,0.8,0.1},\n"\
"		{0.2,0.8,0.2}};\n"\
"\n"\
"	int col=a&3;\n"\
"	float3 rgb=color_tab[col];\n"\
"	float  i=convert_float((a&(255-7)));\n"\
"	int    r=i*rgb.x;if(r>255)r=255;\n"\
"	int    g=i*rgb.y;if(g>255)g=255;\n"\
"	int    b=i*rgb.z;if(b>255)b=255;\n"\
"	p_screenBuffer[of]=b+g*256+r*65536;\n"\
"\n"\
"	if((a&0xffff00)==0xffff00) p_screenBuffer[of]=0x0088cc;\n"\
"};"};

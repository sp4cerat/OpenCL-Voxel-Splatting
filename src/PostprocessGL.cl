#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#define loopi(start_l,end_l) for ( int i=start_l;i<end_l;++i )
#define loopj(start_l,end_l) for ( int j=start_l;j<end_l;++j )
#define loopk(start_l,end_l) for ( int k=start_l;k<end_l;++k )
#define loopm(start_l,end_l) for ( int m=start_l;m<end_l;++m )
#define loopl(start_l,end_l) for ( int l=start_l;l<end_l;++l )
#define loopn(start_l,end_l) for ( int n=start_l;n<end_l;++n )

__kernel    void splatting(
			__global uint *p_screenBuffer,
			__global uint *p_backBuffer,
			__global uint *p_data,
			int p_width, 
			int p_height,
			float4 matx,
			float4 maty,
			float4 matz
			)
{
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int bw = get_local_size(0);
	const int bh = get_local_size(1);
	const int idx = get_global_id(0);
	const int idy = get_global_id(1);

	float anim=matx.w/1000;
	
	loopj( -10, 10)
	loopk(   0,10)
	{
		int addx=j*1024*2;
		int addy=(((j^k)*24352)&1023);
		int addz=k*1024*2;

		float4 p,pt;
		p.x=addx+512;
		p.y=addy+512;
		p.z=addz+512;
		p.w=1;
		pt.x=dot(p,matx);
		pt.y=dot(p,maty);
		pt.z=dot(p,matz);
				
		if(pt.z<-600)continue;
		
		int sx=pt.x*p_width/pt.z+p_width/2;
		if(sx >=  p_width*2) continue;
		if(sx <  -p_width) continue;

		int sy=pt.y*p_width/pt.z+p_height/2;
		if(sy >=  p_height*2) continue;
		if(sy <  -p_height) continue;
				
		int level=log2((pt.z/2)/p_width);

		if(level<0)level=0;
		if(level>=9)continue;

		int m=p_data[level];
		int num_cubes=p_data[m];m++;

		loopi(0,num_cubes)
		{
			int cx=(p_data[m]<<level)+addx;m++;
			int cy=(p_data[m]<<level)+addy;m++;
			int cz=(p_data[m]<<level)+addz;m++;
			
			// cull cubes - calculation needs adjustment
			/*
			int out_of_view=0;
			if(1)
			{
				loopi(0,2)
				loopj(0,2)
				loopk(0,2)
				{
					int x=cx+i*256;
					int y=cy+j*256;
					int z=cz+k*256;

					float4 p,pt;
					p.x=x;p.y=y;p.z=z;p.w=1;
					pt.x=dot(p,matx);
					pt.y=dot(p,maty);
					pt.z=dot(p,matz);

					if(pt.z<0){ out_of_view++;continue;}
				
					int sx=pt.x*p_width/pt.z+p_width/2;
					if(sx >= p_width) { out_of_view++;continue;}
					if(sx <  0) { out_of_view++;continue;}

					int sy=pt.y*p_width/pt.z+p_height/2;
					if(sy >= p_height) { out_of_view++;continue;}
					if(sy <  0) { out_of_view++;continue;}
				}
			}
			*/
			

			int cc=p_data[m];m++;
			int cd=m+idx; // cube data offset start
			int ce=m+cc;  // cube data offset end
			m+=cc;        // voxel count

			// cull cube ?
			// if(out_of_view==8)continue;

			int pack=0; int n=0;

			while(cd<ce)
			{	
				pack=p_data[cd]; // x y z color

				cd+=8192; 

				float4 p;
				p.x=cx+((pack&0xff)<<level);
				p.y=cy+(((pack>>8)&0xff)<<level);
				p.z=cz+(((pack>>16)&0xff)<<level);
				p.w=1;

				//simple animation experiment
				/*
				float4 q;
				q.y=-127+(((pack*24345)&0xff)<<level);
				q.z=-127+((((pack*574345)>>8)&0xff)<<level);
				q.x=-127+((((pack*234646)>>16)&0xff)<<level);
				q.w=1;
				p=p+q*anim;
				*/

				float4 pt;
				pt.z=dot(p,matz);
				if(pt.z<1) continue;
				if(pt.z>65000 ) continue;

				pt.x=dot(p,matx);
				int sx=pt.x*p_width/pt.z+p_width/2;
				if(sx >= p_width) continue;
				if(sx <  0) continue;

				pt.y=dot(p,maty);
				int sy=pt.y*p_width/pt.z+p_height/2;
				if(sy >= p_height) continue;
				if(sy <  0) continue;

				int sz=pt.z; 
				int ofs=sy*p_width + sx;
				sz<<=8;
								
				//zbuffertest 1
				if((p_screenBuffer[ofs]&0xffff00) <= sz) continue;

				int ic=     ((pack>>24)&0xff);//integer color

				//zbuffertest 2
				atom_min(&p_screenBuffer[ofs],sz+ic);
			}
		}
	}
}
__kernel    void postx(
			__global uint *p_screenBuffer,
			__global uint *p_backBuffer,
			int p_width, 
			int p_height,
			int p_iter
			)
{
	//__local uint smem[4000];
	//const int tid = get_local_id(0);
	const int x   = get_global_id(0);
	const int y   = get_global_id(1);
	if( x >= (p_width  >>p_iter)) return;
	if( y >= (p_height >>p_iter)) return;
	
	int of_read    = x*2+y*2*p_width+(p_width>>(p_iter));

	if(p_iter==0)of_read-=p_width;

	int of_xy_read = of_read+(p_height/2)*p_width;
	int of_write   = x   +y*  p_width+(p_width>>(p_iter+1));
	int of_xy_write= of_write+(p_height/2)*p_width;

	int a1,a2,a3,a4;

	if(p_iter>0)
	{
		a1=p_backBuffer[of_read];
		a2=p_backBuffer[of_read+1];
		a3=p_backBuffer[of_read+p_width];
		a4=p_backBuffer[of_read+1+p_width];
	}
	else
	{
		a1=p_screenBuffer[of_read];
		a2=p_screenBuffer[of_read+1];
		a3=p_screenBuffer[of_read+p_width];
		a4=p_screenBuffer[of_read+1+p_width];
	}

	
	int xy=0;

	if(p_iter==0)
	{
		int xo=0,yo=0;
		if(a2<a1){ a1=a2;xo=1;yo=0; }
		if(a3<a1){ a1=a3;xo=0;yo=1; }
		if(a4<a1){ a1=a4;xo=1;yo=1; }
		xo+=x*2;
		yo+=y*2;
		int scale=(a1>712*512)?1200:800;
		int radius=(p_width*scale/((a1&0xffff00)+2));
		if(radius>255)radius=255;
		xy=(xo<<8)+(yo<<20)+radius;
	}
	else
	{	
		int ofs=0;
		if(a2<a1){ a1=a2;ofs=1; }
		if(a3<a1){ a1=a3;ofs=p_width; }
		if(a4<a1){ a1=a4;ofs=1+p_width; }

		xy=p_backBuffer[of_xy_read+ofs];

		/*
		int cmpx=(xy>>8)&4095;
		int cmpy= xy>>20;
		int cmpz= a1&0xffff00;
		int cmpc= a1&255;
		int cmpr= xy&255;

		int sum=0;
		int xsum=0;
		int ysum=0;
		int zsum=0;
		int csum=0;
		int rsum=0;

		loopi(0,1)
		loopj(0,1)
		{
			int z=p_backBuffer[of_read+i+j*p_width];
			int c=z&255;
			z&=0xffff00;
			int rxy=p_backBuffer[of_xy_read++i+j*p_width];

		}
		*/

		/*
		int xy2=p_backBuffer[of_xy_write];
		x2=(xy2>>8)&4095;
		y2= xy2>>20;
		z2=p_backBuffer[of_write];
		r2= xy2 &255;
		*/
	}
	p_backBuffer[of_write]=a1;
	p_backBuffer[of_xy_write]=xy;
}

__kernel    void posty(
			__global uint *p_screenBuffer,
			__global uint *p_backBuffer,
			int p_width, 
			int p_height,
			int p_iter
			)
{
	//local uint smem[4000];
	//const int tid = get_local_id(0);
	const int x   = get_global_id(0);
	const int y   = get_global_id(1);
	if( x >= (p_width  >>1)) return;
	if( x >= (p_width  >>p_iter)) return;
	if( y >= (p_height >>p_iter)) return;

	int x_max=p_width  >>p_iter;
	int y_max=p_height  >>p_iter;

	int xy_of=(p_height/2)*p_width;
	
	int of_read    = (x/2)+(y/2)*p_width+(p_width>>(p_iter+1));

	int of_xy_read = of_read+xy_of;

	int of_write   =  x   + y *  p_width+(p_width>>p_iter);

	if(p_iter==0) of_write-=p_width;

	int of_xy_write= of_write+xy_of;

	int xy2,x2,y2,z2,r2;

	if(p_iter==0)
	{
		x2=x;
		y2=y;
		z2=p_screenBuffer[of_write];
		int scale=(z2>512*512)?1200:800;
		r2=(p_width*scale/((z2&0xffff00)+2));
	}
	else
	{
		xy2=p_backBuffer[of_xy_write];
		x2=(xy2>>8)&4095;
		y2= xy2>>20;
		z2=p_backBuffer[of_write];
		r2= xy2 &255;
	}
	
	int px=x<<p_iter;
	int py=y<<p_iter;
	if(p_iter>0) 
	{ 
		px+=1<<(p_iter-1);
		py+=1<<(p_iter-1);
	} 

//	if (abs(px-x1)+abs(py-y1) <= r1)
//	float r= sqrt( convert_float( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) );

	int zmin=0xffffff;

	float sumx=0;
	float sumy=0;
	float sumz=0;
	float sumc=0;
	float sumr=0;
	float sum=0;
	int   sumcc=0;

	loopi(-2,3) if(i+x/2>=0) if(i+x/2<x_max)
	loopj(-2,3) if(j+y/2>=0) if(j+y/2<y_max)
	{
		int add=i + j*p_width;
		int xy1=p_backBuffer[of_xy_read + add ];
		int x1=(xy1>>8)&4095;
		int y1= xy1>>20;
		int z1=p_backBuffer[of_read + add ];
		int r1= xy1 &255;
	
		float r= sqrt( convert_float( (x1-px)*(x1-px) + (y1-py)*(y1-py)) );
		//(abs(x1-px)+abs(y1-py));//

		if(r1>r)
		if((z1&0xffff00)<zmin)
		{
			zmin=z1&0xffff00;
			sumcc=z1&3;

			//sumc=z1&0xff;
		}
	}

	loopi(-2,3) if(i+x/2>=0) if(i+x/2<x_max)
	loopj(-2,3) if(j+y/2>=0) if(j+y/2<y_max)
	{
		int add=i + j*p_width;
		int xy1=p_backBuffer[of_xy_read + add ];
		int x1=(xy1>>8)&4095;
		int y1= xy1>>20;
		int z1=p_backBuffer[of_read + add ];
		int r1= xy1 &255;
	
		float r= sqrt( convert_float( (x1-px)*(x1-px) + (y1-py)*(y1-py)) );
		//(abs(x1-px)+abs(y1-py));//

		if(sumc==255) sumc=(z1&0xff);

		if(r1>r)
		if(abs((z1&0xffff00)-zmin)<(4*256))
		{
			float cmpr=r1;
			float w=(fabs(r-cmpr));;//2;//-fabs(r-r1);//100/(10+fabs(r-r1));
			w*=w;
			w+=abs((z1&0xffff00)-zmin)/256;
			w+=1/w;
			sumx+=w*convert_float(x1);
			sumy+=w*convert_float(y1);
			sumz+=w*convert_float(z1&0xffff00);
			sumr+=w*convert_float(r1);
			sumc+=w*convert_float(z1&255);
			sum+=w;
		}
	}
	if(sum>0)
	{
		sumz/=sum;
		sumx/=sum;
		sumy/=sum;
		sumc/=sum;
		sumr/=sum;
		sum=1;

		if(abs((z2&0xffff00)-(convert_int(sumz)&0xffff00))<(4*256))
		{
			float w=1;//0.5;
			sumx+=w*convert_float(x2);
			sumy+=w*convert_float(y2);
			sumz+=w*convert_float(z2&0xffff00);
			sumc+=w*convert_float(z2&0xff);
			sumcc=z2&3;
			sumr+=w*convert_float(r2);
			sum+=w;
		}
		sumz/=sum;
		sumx/=sum;
		sumy/=sum;
		sumc/=sum;
		sumr/=sum;

		z2=(convert_int(sumz)&0xffff00)+(convert_int(sumc)&0xfc)+(sumcc&3);
		xy2=convert_int(sumr)+(convert_int(sumx)<<8)+(convert_int(sumy)<<20);
	}


	if(p_iter==0)
	{
		p_screenBuffer[of_write]=z2;
		//p_screenBuffer[of_xy_write]=xy2;
	}
	else
	{
		p_backBuffer[of_write]=z2;
		p_backBuffer[of_xy_write]=xy2;
	}
}

__kernel    void colorize(
			__global uint *p_screenBuffer,
			int p_width, 
			int p_height
			)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if( x >= p_width ) return;
	if( y >= p_height ) return;

	if( x >= p_width*3/4 )
	{
		return;
	}

	int of=x+y*p_width;
	int a=p_screenBuffer[of];//&0xff;

	float3 color_tab[4]={
		{0.8,1.0,0.3},
		{1.0,0.7,0.3},
		{1.5,0.8,0.1},
		{0.2,0.8,0.2}};

	int col=a&3;
	float3 rgb=color_tab[col];
	float  i=convert_float((a&(255-7)));
	int    r=i*rgb.x;if(r>255)r=255;
	int    g=i*rgb.y;if(g>255)g=255;
	int    b=i*rgb.z;if(b>255)b=255;
	p_screenBuffer[of]=b+g*256+r*65536;

	if((a&0xffff00)==0xffff00) p_screenBuffer[of]=0x0088cc;
}
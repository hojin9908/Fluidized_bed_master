////////////////////////////////////////////////////////////////////////
// switch ptype 함수
__host__ __device__ int functions_switch_ptype()
{

}


////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_switch_p_type_MCCI_CCI(int_t inout,part1*P1,part1*TP1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	//Real concn=P2[i].concn;
	uint_t p_type=P1[i].p_type;

	if(p_type==CONCRETE_SOL){
		Real temp=P1[i].temp;
		// cci 1 siliceous concrete ablation temperature
		if(temp>1523.15){
			P1[i].p_type=CONCRETE;
			TP1[i].p_type=CONCRETE;
		}
	}
}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_switch_p_type_air(int_t inout,part1*P1,part1*TP1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type!=9) return;
	if(P1[i].z>0.1) return;

	uint_t p_type=P1[i].p_type;
	Real tmp_h=P1[i].h;
	Real z0=P1[i].z;
	Real space;
	space=tmp_h/1.6;


	if(z0>(0.0)){

		P1[i].p_type=3;
		TP1[i].p_type=3;
	}
}


////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_switch_p_type_jet(int_t inout,part1*P1,part1*TP1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	//Real concn=P2[i].concn;
	uint_t p_type=P1[i].p_type;

	if(p_type==MOVING){
		Real zi=P1[i].z;
		// cci 1 siliceous concrete ablation temperature
		if(zi<0.4000*1e3){
			P1[i].p_type=2;
			TP1[i].p_type=2;
		}
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_correction_KGC_2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	Real search_range,tmp_h,tmp_A;
	Real xi,yi;
	Real tmpxx,tmpyy,tmpxy;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpxx=tmpyy=tmpxy=0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			//int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){

					if(P1[j].p_type<=1000){
						Real xj,yj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
						if(tdist>0&&tdist<search_range){
							Real tdwij,mj,rhoj,txx,txy,tyy,rtd,mtd;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rhoj=P1[j].rho;
							mtd=mj*tdwij;
							rtd=1.0/(rhoj*tdist);

							txx=-mtd*(xi-xj)*(xi-xj);
							txx*=rtd;
							txy=-mtd*(yi-yj)*(xi-xj);
							txy*=rtd;
							tyy=-mtd*(yi-yj)*(yi-yj);
							tyy*=rtd;

							tmpxx+=txx;
							tmpxy+=txy;
							tmpyy+=tyy;
						}

					}
					
				}
			}
		}
	}
	// save values to particle array

	Real tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=tmpyy*rtcmd;
		P3[i].Cm[0][1]=-tmpxy*rtcmd;
		P3[i].Cm[1][0]=-tmpxy*rtcmd;
		P3[i].Cm[1][1]=tmpxx*rtcmd;
	}else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
	}
}


__global__ void KERNEL_clc_correction_KGC_2D_sph(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	Real search_range,tmp_h,tmp_A;
	Real xi,yi;
	Real tmpxx,tmpyy,tmpxy;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpxx=tmpyy=tmpxy=0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			//int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){

					if(P1[j].p_type<=1000){
						Real xj,yj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
						if(tdist>0&&tdist<search_range){
							Real tdwij,mj,rhoj,txx,txy,tyy,rtd,mtd;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rhoj=P1[j].rho;
							mtd=mj*tdwij;
							rtd=1.0/(rhoj*tdist);

							txx=-mtd*(xi-xj)*(xi-xj);
							txx*=rtd;
							txy=-mtd*(yi-yj)*(xi-xj);
							txy*=rtd;
							tyy=-mtd*(yi-yj)*(yi-yj);
							tyy*=rtd;

							tmpxx+=txx;
							tmpxy+=txy;
							tmpyy+=tyy;
						}

					}
					
				}
			}
		}
	}
	// save values to particle array

	Real tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=tmpyy*rtcmd;
		P3[i].Cm[0][1]=-tmpxy*rtcmd;
		P3[i].Cm[1][0]=-tmpxy*rtcmd;
		P3[i].Cm[1][1]=tmpxx*rtcmd;
	}else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
	}
}


// __global__ void KERNEL_clc_correction_KGC_2D_sph(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
// {
// 	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2_sph) return;
// 	if(P1[i].i_type>i_type_crt) return;
// 	// if(P1[i].p_type>1000) return;

// 	int_t icell,jcell;
// 	Real search_range,tmp_h,tmp_A;
// 	Real xi,yi;
// 	Real tmpxx,tmpyy,tmpxy;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	search_range=k_search_kappa*tmp_h;	// search range

// 	xi=P1[i].x;
// 	yi=P1[i].y;

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;

// 	tmpxx=tmpyy=tmpxy=0;

// 	for(int_t y=-1;y<=1;y++){
// 		for(int_t x=-1;x<=1;x++){
// 			//int_t k=(icell+x)+k_NI*(jcell+y);
// 			int_t k=idx_cell(icell+x,jcell+y,0);
// 			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
// 			if(g_str[k]!=cu_memset){
// 				int_t fend=g_end[k];
// 				for(int_t j=g_str[k];j<fend;j++){

// 					if(P1[j].p_type<=1000){
// 						Real xj,yj,tdist;
// 						xj=P1[j].x;
// 						yj=P1[j].y;

// 						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// 						if(tdist>0&&tdist<search_range){
// 							Real tdwij,mj,rhoj,txx,txy,tyy,rtd,mtd;
// 							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
// 							mj=P1[j].m;
// 							rhoj=P1[j].rho;
// 							mtd=mj*tdwij;
// 							rtd=1.0/(rhoj*tdist);

// 							txx=-mtd*(xi-xj)*(xi-xj);
// 							txx*=rtd;
// 							txy=-mtd*(yi-yj)*(xi-xj);
// 							txy*=rtd;
// 							tyy=-mtd*(yi-yj)*(yi-yj);
// 							tyy*=rtd;

// 							tmpxx+=txx;
// 							tmpxy+=txy;
// 							tmpyy+=tyy;
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	// save values to particle array

// 	Real tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
// 	if(abs(tmpcmd)>Min_det){
// 		Real rtcmd=1.0/tmpcmd;
// 		P3[i].Cm[0][0]=tmpyy*rtcmd;
// 		P3[i].Cm[0][1]=-tmpxy*rtcmd;
// 		P3[i].Cm[1][0]=-tmpxy*rtcmd;
// 		P3[i].Cm[1][1]=tmpxx*rtcmd;
// 	}else{
// 		P3[i].Cm[0][0]=1;
// 		P3[i].Cm[0][1]=0;
// 		P3[i].Cm[1][0]=0;
// 		P3[i].Cm[1][1]=1;
// 	}
// }
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_correction_KGC_3D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real search_range,tmp_h,tmp_A;;
	Real tmpxx,tmpyy,tmpzz,tmpxy,tmpyz,tmpzx;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpxx=tmpyy=tmpzz=0;
	tmpxy=tmpyz=tmpzx=0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
					

						if(P1[j].p_type<=1000){

							Real xj,yj,zj,tdist;
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist>0&&tdist<search_range){
								Real tdwij,mj,rhoj,txx,txy,tyy,tzx,tyz,tzz,rtd,mtd;
								tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
								mj=P1[j].m;
								rhoj=P1[j].rho;
	
								mtd=mj*tdwij;
								rtd=1.0/(rhoj*tdist);
	
								txx=-mtd*(xi-xj)*(xi-xj);
								txx*=rtd;
								txy=-mtd*(yi-yj)*(xi-xj);
								txy*=rtd;
								tyy=-mtd*(yi-yj)*(yi-yj);
								tyy*=rtd;
								tzx=-mtd*(xi-xj)*(zi-zj);
								tzx*=rtd;
								tyz=-mtd*(yi-yj)*(zi-zj);
								tyz*=rtd;
								tzz=-mtd*(zi-zj)*(zi-zj);
								tzz*=rtd;
								tmpxx+=txx;
								tmpxy+=txy;
								tmpyy+=tyy;
								tmpzx+=tzx;
								tmpyz+=tyz;
								tmpzz+=tzz;
							}

						}
						
					}
				}
			}
		}
	}
	// save values to particle array
	Real tmpcmd;
	tmpcmd=tmpxx*(tmpyy*tmpzz-tmpyz*tmpyz);
	tmpcmd-=tmpxy*(tmpxy*tmpzz-tmpyz*tmpzx);
	tmpcmd+=tmpzx*(tmpxy*tmpyz-tmpyy*tmpzx);

	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=(tmpyy*tmpzz-tmpyz*tmpyz)*rtcmd;
		P3[i].Cm[0][1]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].Cm[0][2]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].Cm[1][0]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].Cm[1][1]=(tmpxx*tmpzz-tmpzx*tmpzx)*rtcmd;
		P3[i].Cm[1][2]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].Cm[2][0]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].Cm[2][1]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].Cm[2][2]=(tmpxx*tmpyy-tmpxy*tmpxy)*rtcmd;
	}
	else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[0][2]=0;
		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
		P3[i].Cm[1][2]=0;
		P3[i].Cm[2][0]=0;
		P3[i].Cm[2][1]=0;
		P3[i].Cm[2][2]=1;
	}
}

__global__ void KERNEL_clc_correction_KGC_3D_sph(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real search_range,tmp_h,tmp_A;;
	Real tmpxx,tmpyy,tmpzz,tmpxy,tmpyz,tmpzx;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpxx=tmpyy=tmpzz=0;
	tmpxy=tmpyz=tmpzx=0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
					

						if(P1[j].p_type<=1000){

							Real xj,yj,zj,tdist;
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist>0&&tdist<search_range){
								Real tdwij,mj,rhoj,txx,txy,tyy,tzx,tyz,tzz,rtd,mtd;
								tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
								mj=P1[j].m;
								rhoj=P1[j].rho;
	
								mtd=mj*tdwij;
								rtd=1.0/(rhoj*tdist);
	
								txx=-mtd*(xi-xj)*(xi-xj);
								txx*=rtd;
								txy=-mtd*(yi-yj)*(xi-xj);
								txy*=rtd;
								tyy=-mtd*(yi-yj)*(yi-yj);
								tyy*=rtd;
								tzx=-mtd*(xi-xj)*(zi-zj);
								tzx*=rtd;
								tyz=-mtd*(yi-yj)*(zi-zj);
								tyz*=rtd;
								tzz=-mtd*(zi-zj)*(zi-zj);
								tzz*=rtd;
								tmpxx+=txx;
								tmpxy+=txy;
								tmpyy+=tyy;
								tmpzx+=tzx;
								tmpyz+=tyz;
								tmpzz+=tzz;
							}

						}
						
					}
				}
			}
		}
	}
	// save values to particle array
	Real tmpcmd;
	tmpcmd=tmpxx*(tmpyy*tmpzz-tmpyz*tmpyz);
	tmpcmd-=tmpxy*(tmpxy*tmpzz-tmpyz*tmpzx);
	tmpcmd+=tmpzx*(tmpxy*tmpyz-tmpyy*tmpzx);

	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=(tmpyy*tmpzz-tmpyz*tmpyz)*rtcmd;
		P3[i].Cm[0][1]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].Cm[0][2]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].Cm[1][0]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
		P3[i].Cm[1][1]=(tmpxx*tmpzz-tmpzx*tmpzx)*rtcmd;
		P3[i].Cm[1][2]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].Cm[2][0]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
		P3[i].Cm[2][1]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
		P3[i].Cm[2][2]=(tmpxx*tmpyy-tmpxy*tmpxy)*rtcmd;
	}
	else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[0][2]=0;
		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
		P3[i].Cm[1][2]=0;
		P3[i].Cm[2][0]=0;
		P3[i].Cm[2][1]=0;
		P3[i].Cm[2][2]=1;
	}
}

// __global__ void KERNEL_clc_correction_KGC_3D_sph(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
// {
// 	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2_sph) return;
// 	if(P1[i].i_type>i_type_crt) return;
// 	// if(P1[i].p_type>1000) return;

// 	int_t icell,jcell,kcell;
// 	Real xi,yi,zi;
// 	Real search_range,tmp_h,tmp_A;;
// 	Real tmpxx,tmpyy,tmpzz,tmpxy,tmpyz,tmpzx;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	search_range=k_search_kappa*tmp_h;	// search range

// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

// 	tmpxx=tmpyy=tmpzz=0;
// 	tmpxy=tmpyz=tmpzx=0;
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
					

// 						if(P1[j].p_type<=1000){

// 							Real xj,yj,zj,tdist;
// 							xj=P1[j].x;
// 							yj=P1[j].y;
// 							zj=P1[j].z;
	
// 							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
// 							if(tdist>0&&tdist<search_range){
// 								Real tdwij,mj,rhoj,txx,txy,tyy,tzx,tyz,tzz,rtd,mtd;
// 								tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
// 								mj=P1[j].m;
// 								rhoj=P1[j].rho;
	
// 								mtd=mj*tdwij;
// 								rtd=1.0/(rhoj*tdist);
	
// 								txx=-mtd*(xi-xj)*(xi-xj);
// 								txx*=rtd;
// 								txy=-mtd*(yi-yj)*(xi-xj);
// 								txy*=rtd;
// 								tyy=-mtd*(yi-yj)*(yi-yj);
// 								tyy*=rtd;
// 								tzx=-mtd*(xi-xj)*(zi-zj);
// 								tzx*=rtd;
// 								tyz=-mtd*(yi-yj)*(zi-zj);
// 								tyz*=rtd;
// 								tzz=-mtd*(zi-zj)*(zi-zj);
// 								tzz*=rtd;
// 								tmpxx+=txx;
// 								tmpxy+=txy;
// 								tmpyy+=tyy;
// 								tmpzx+=tzx;
// 								tmpyz+=tyz;
// 								tmpzz+=tzz;
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	// save values to particle array
// 	Real tmpcmd;
// 	tmpcmd=tmpxx*(tmpyy*tmpzz-tmpyz*tmpyz);
// 	tmpcmd-=tmpxy*(tmpxy*tmpzz-tmpyz*tmpzx);
// 	tmpcmd+=tmpzx*(tmpxy*tmpyz-tmpyy*tmpzx);

// 	if(abs(tmpcmd)>Min_det){
// 		Real rtcmd=1.0/tmpcmd;
// 		P3[i].Cm[0][0]=(tmpyy*tmpzz-tmpyz*tmpyz)*rtcmd;
// 		P3[i].Cm[0][1]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
// 		P3[i].Cm[0][2]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
// 		P3[i].Cm[1][0]=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
// 		P3[i].Cm[1][1]=(tmpxx*tmpzz-tmpzx*tmpzx)*rtcmd;
// 		P3[i].Cm[1][2]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
// 		P3[i].Cm[2][0]=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
// 		P3[i].Cm[2][1]=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
// 		P3[i].Cm[2][2]=(tmpxx*tmpyy-tmpxy*tmpxy)*rtcmd;
// 	}
// 	else{
// 		P3[i].Cm[0][0]=1;
// 		P3[i].Cm[0][1]=0;
// 		P3[i].Cm[0][2]=0;
// 		P3[i].Cm[1][0]=0;
// 		P3[i].Cm[1][1]=1;
// 		P3[i].Cm[1][2]=0;
// 		P3[i].Cm[2][0]=0;
// 		P3[i].Cm[2][1]=0;
// 		P3[i].Cm[2][2]=1;
// 	}
// }


__global__ void KERNEL_clc_correction_FPM_2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	Real search_range,tmp_h,tmp_A;
  Real t11,t12,t13,t21,t22,t23,t31,t32,t33;
	Real xi,yi;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){

					if (P1[j].p_type<=1000){

						Real xj,yj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
	
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
						if(tdist>0&&tdist<search_range){
							Real twij,tdwij,mj,rhoj,rtd,mtd;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rhoj=P1[j].rho;
							// mtd=mj*tdwij;
							rtd=1.0/(rhoj*tdist);
	
							t11+=mj*twij*tdist*rtd;
							t12+=-mj*twij*(xi-xj)*tdist*rtd;
							t13+=-mj*twij*(yi-yj)*tdist*rtd;
	
							t21+=mj*tdwij*(xi-xj)*rtd;
							t22+=-mj*tdwij*(xi-xj)*(xi-xj)*rtd;
							t23+=-mj*tdwij*(yi-yj)*(xi-xj)*rtd;
	
							t31+=mj*tdwij*(yi-yj)*rtd;
							t32+=-mj*tdwij*(xi-xj)*(yi-yj)*rtd;
							t33+=-mj*tdwij*(yi-yj)*(yi-yj)*rtd;
						}
					}
					
				}
			}
		}
	}
	// save values to particle array

	Real tmpcmd=t11*(t22*t33-t23*t32)-t12*(t21*t33-t23*t31)+t13*(t21*t32-t22*t31);
	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=(t22*t33-t23*t32)*rtcmd;
		P3[i].Cm[0][1]=-(t12*t33-t13*t32)*rtcmd;
		P3[i].Cm[0][2]=(t12*t23-t13*t22)*rtcmd;

		P3[i].Cm[1][0]=-(t21*t33-t23*t31)*rtcmd;
		P3[i].Cm[1][1]=(t11*t33-t13*t31)*rtcmd;
		P3[i].Cm[1][2]=-(t11*t23-t13*t21)*rtcmd;

		P3[i].Cm[2][0]=(t21*t32-t22*t31)*rtcmd;
		P3[i].Cm[2][1]=-(t11*t32-t12*t31)*rtcmd;
		P3[i].Cm[2][2]=(t11*t22-t12*t21)*rtcmd;
	}else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[0][2]=0;

		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
		P3[i].Cm[1][2]=0;

		P3[i].Cm[2][0]=0;
		P3[i].Cm[2][1]=0;
		P3[i].Cm[2][2]=1;
	}
}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_correction_DFPM_2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	Real search_range,tmp_h,tmp_A;
	Real t11,t12,t13,t21,t22,t23,t31,t32,t33;
	Real xi,yi;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			//int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);
			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					
					if(P1[j].p_type<=1000){

						Real xj,yj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
	
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
						if(tdist>0&&tdist<search_range){
							Real twij,tdwij,mj,rhoj,rtd,mtd;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rhoj=P1[j].rho;
							// mtd=mj*tdwij;
							rtd=1.0/(rhoj*tdist);
	
							t11+=mj*twij*tdist*rtd;
							t12+=0;
							t13+=0;
	
							t21+=0;
							t22+=-mj*tdwij*(xi-xj)*(xi-xj)*rtd;
							t23+=0;
	
							t31+=0;
							t32+=0;
							t33+=-mj*tdwij*(yi-yj)*(yi-yj)*rtd;
						}

					}

					
				}
			}
		}
	}
	// save values to particle array

	Real tmpcmd=t11*t22*t33;
	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=(t22*t33)*rtcmd;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[0][2]=0;

		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=(t11*t33)*rtcmd;
		P3[i].Cm[1][2]=0;

		P3[i].Cm[2][0]=0;
		P3[i].Cm[2][1]=0;
		P3[i].Cm[2][2]=(t11*t22)*rtcmd;
	}else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[0][2]=0;

		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
		P3[i].Cm[1][2]=0;

		P3[i].Cm[2][0]=0;
		P3[i].Cm[2][1]=0;
		P3[i].Cm[2][2]=1;
	}
}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_correction_KGF_2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000)	return;

	int_t icell,jcell;
	Real search_range,tmp_h,tmp_A;
	Real t11,t12,t13,t21,t22,t23,t31,t32,t33;
	Real xi,yi;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			//int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					
					if(P1[j].p_type<=1000){
						Real xj,yj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
	
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
						if(tdist>0&&tdist<search_range){
							Real twij,mj,rhoj,rt,mtd;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rhoj=P1[j].rho;
							// mtd=mj*tdwij;
							// rtd=1.0/(rhoj*tdist);
							rt=1.0/rhoj;
	
							t11+=mj*twij*rt;
							t12+=-mj*twij*(xi-xj)*rt;
							t13+=-mj*twij*(yi-yj)*rt;
	
							t21+=-mj*twij*(xi-xj)*rt;
							t22+=mj*twij*(xi-xj)*(xi-xj)*rt;
							t23+=mj*twij*(yi-yj)*(xi-xj)*rt;
	
							t31+=-mj*twij*(yi-yj)*rt;
							t32+=mj*twij*(xi-xj)*(yi-yj)*rt;
							t33+=mj*twij*(yi-yj)*(yi-yj)*rt;
						}

					}
					
				}
			}
		}
	}
	// save values to particle array

	Real tmpcmd=t11*(t22*t33-t23*t32)-t12*(t21*t33-t23*t31)+t13*(t21*t32-t22*t31);
	if(abs(tmpcmd)>Min_det){
		Real rtcmd=1.0/tmpcmd;
		P3[i].Cm[0][0]=(t22*t33-t23*t32)*rtcmd;
		P3[i].Cm[0][1]=-(t12*t33-t13*t32)*rtcmd;
		P3[i].Cm[0][2]=(t12*t23-t13*t22)*rtcmd;

		P3[i].Cm[1][0]=-(t21*t33-t23*t31)*rtcmd;
		P3[i].Cm[1][1]=(t11*t33-t13*t31)*rtcmd;
		P3[i].Cm[1][2]=-(t11*t23-t13*t21)*rtcmd;

		P3[i].Cm[2][0]=(t21*t32-t22*t31)*rtcmd;
		P3[i].Cm[2][1]=-(t11*t32-t12*t31)*rtcmd;
		P3[i].Cm[2][2]=(t11*t22-t12*t21)*rtcmd;
	}else{
		P3[i].Cm[0][0]=1;
		P3[i].Cm[0][1]=0;
		P3[i].Cm[0][2]=0;

		P3[i].Cm[1][0]=0;
		P3[i].Cm[1][1]=1;
		P3[i].Cm[1][2]=0;

		P3[i].Cm[2][0]=0;
		P3[i].Cm[2][1]=0;
		P3[i].Cm[2][2]=1;
	}
}


void gradient_correction(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	dim3 b,t;
	t.x=128;
	b.x=(num_part2-1)/t.x+1;
	switch (kgc_solve){
		case KGC:
				if(dim==2) KERNEL_clc_correction_KGC_2D<<<b,t>>>(g_str,g_end,P1,P3);
				if(dim==3) KERNEL_clc_correction_KGC_3D<<<b,t>>>(g_str,g_end,P1,P3);
				cudaDeviceSynchronize();
				break;
		case FPM:
				if(dim==2) KERNEL_clc_correction_FPM_2D<<<b,t>>>(g_str,g_end,P1,P3);
				// if(dim==3) KERNEL_clc_correction_FPM_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
				// cudaDeviceSynchronize();
				break;
		case DFPM:
				if(dim==2) KERNEL_clc_correction_DFPM_2D<<<b,t>>>(g_str,g_end,P1,P3);
				// if(dim==3) KERNEL_clc_correction_DFPM_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
				// cudaDeviceSynchronize();
				break;
		case KGF:
				if(dim==2) KERNEL_clc_correction_KGF_2D<<<b,t>>>(g_str,g_end,P1,P3);
				// if(dim==3) KERNEL_clc_correction_KGF_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
				// cudaDeviceSynchronize();
				break;
		default:
				if(dim==2) KERNEL_clc_correction_KGC_2D<<<b,t>>>(g_str,g_end,P1,P3);
				if(dim==3) KERNEL_clc_correction_KGC_3D<<<b,t>>>(g_str,g_end,P1,P3);
				cudaDeviceSynchronize();
				break;
	}
}


void gradient_correction_sph(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	dim3 b,t;
	t.x=128;
	b.x=(num_part2_sph-1)/t.x+1;
	switch (kgc_solve){
		case KGC:
				if(dim==2) KERNEL_clc_correction_KGC_2D_sph<<<b,t>>>(g_str,g_end,P1,P3);
				if(dim==3) KERNEL_clc_correction_KGC_3D_sph<<<b,t>>>(g_str,g_end,P1,P3);
				cudaDeviceSynchronize();
				break;
		// case FPM:
		// 		if(dim==2) KERNEL_clc_correction_FPM_2D<<<b,t>>>(g_str,g_end,P1,P3);
		// 		// if(dim==3) KERNEL_clc_correction_FPM_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
		// 		// cudaDeviceSynchronize();
		// 		break;
		// case DFPM:
		// 		if(dim==2) KERNEL_clc_correction_DFPM_2D<<<b,t>>>(g_str,g_end,P1,P3);
		// 		// if(dim==3) KERNEL_clc_correction_DFPM_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
		// 		// cudaDeviceSynchronize();
		// 		break;
		// case KGF:
		// 		if(dim==2) KERNEL_clc_correction_KGF_2D<<<b,t>>>(g_str,g_end,P1,P3);
		// 		// if(dim==3) KERNEL_clc_correction_KGF_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
		// 		// cudaDeviceSynchronize();
		// 		break;
		default:
				if(dim==2) KERNEL_clc_correction_KGC_2D_sph<<<b,t>>>(g_str,g_end,P1,P3);
				if(dim==3) KERNEL_clc_correction_KGC_3D_sph<<<b,t>>>(g_str,g_end,P1,P3);
				cudaDeviceSynchronize();
				break;
	}
}


// void gradient_correction_sph(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
// {
// 	dim3 b,t;
// 	t.x=128;
// 	b.x=(num_part2_sph-1)/t.x+1;
// 	switch (kgc_solve){
// 		case KGC:
// 				if(dim==2) KERNEL_clc_correction_KGC_2D_sph<<<b,t>>>(g_str,g_end,P1,P3);
// 				if(dim==3) KERNEL_clc_correction_KGC_3D_sph<<<b,t>>>(g_str,g_end,P1,P3);
// 				cudaDeviceSynchronize();
// 				break;
// 		// case FPM:
// 		// 		if(dim==2) KERNEL_clc_correction_FPM_2D<<<b,t>>>(g_str,g_end,P1,P3);
// 		// 		// if(dim==3) KERNEL_clc_correction_FPM_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
// 		// 		// cudaDeviceSynchronize();
// 		// 		break;
// 		// case DFPM:
// 		// 		if(dim==2) KERNEL_clc_correction_DFPM_2D<<<b,t>>>(g_str,g_end,P1,P3);
// 		// 		// if(dim==3) KERNEL_clc_correction_DFPM_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
// 		// 		// cudaDeviceSynchronize();
// 		// 		break;
// 		// case KGF:
// 		// 		if(dim==2) KERNEL_clc_correction_KGF_2D<<<b,t>>>(g_str,g_end,P1,P3);
// 		// 		// if(dim==3) KERNEL_clc_correction_KGF_3D<<<b,t>>>(inout,g_str,g_end,P1,P3);
// 		// 		// cudaDeviceSynchronize();
// 		// 		break;
// 		default:
// 				if(dim==2) KERNEL_clc_correction_KGC_2D_sph<<<b,t>>>(g_str,g_end,P1,P3);
// 				if(dim==3) KERNEL_clc_correction_KGC_3D_sph<<<b,t>>>(g_str,g_end,P1,P3);
// 				cudaDeviceSynchronize();
// 				break;
// 	}
// }
////////////////////////////////////////////////////////////////////////
// calcuate color field for two-phase flow surface tension model (2017.05.08 jyb)
__global__ void KERNEL_clc_color_field2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real tmpn,tmpd;
	Real search_range,tmp_h,tmp_A;

	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpn=tmpd=0.0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){

					if(P1[j].p_type<=1000){

						Real xj,yj,tdist; //rr
						xj=P1[j].x;
						yj=P1[j].y;
	
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
						if(tdist<search_range){
							Real twij,mj,rhoj;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
	
							mj=P1[j].m;
							rhoj=P1[j].rho;
	
							tmpn+=mj*twij*((P1[j].p_type==0)||(ptypei==P1[j].p_type))/rhoj;
							tmpd+=mj*twij/rhoj;
						}

					}
					
				}
			}
		}
	}
	P3[i].cc=tmpn/tmpd;
}
////////////////////////////////////////////////////////////////////////
// calcuate color field for two-phase flow surface tension model (2017.05.08 jyb)
__global__ void KERNEL_clc_color_field3D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;


	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real tmpn,tmpd;
	Real search_range,tmp_h,tmp_A;

	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpn=tmpd=0.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				//int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){

						if(P1[j].p_type<=1000){

							Real xj,yj,zj,tdist; //rr
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
	
							if(tdist<search_range){
								Real twij,mj,rhoj;
								twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
	
								mj=P1[j].m;
								rhoj=P1[j].rho;
	
								tmpn+=mj*twij*(((P1[i].p_type==2)&(P1[j].p_type==0))||(ptypei==P1[j].p_type))/rhoj;
								tmpd+=mj*twij/rhoj;
							}

						}

						
					}
				}
			}
		}
	}
	P3[i].cc=tmpn/tmpd;
}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_prep2D(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi;
	Real uxi,uyi;
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy;
	Real tmp_ncx, tmp_ncy, tmp_nx, tmp_ny;
	Real tvis_t=0.0,th;
	Real nx_ci, ny_ci, nmag_ci;
	Real nxwi, nywi, adddw;
	Real nxti, nyti;

	xi=P1[i].x;
 	yi=P1[i].y;

	uxi=P1[i].ux;
	uyi=P1[i].uy;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}

	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	// 변수 초기화
	tmp_flt=tmp_SR=tmp_rhox=tmp_rhoy=0.0;
	tmp_nx=tmp_ny=tmp_ncx=tmp_ncy=0.0;

	// 계산
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;

			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					
					if(P1[j].p_type<=1000){
						Real xj,yj,uxj,uyj,uij2,mj,rhoj,ccj,tdwx,tdwy,tmp_wij,tmp_dwij,tdist,tmp_val;
						int_t ptypej;
	
						xj=P1[j].x;
						yj=P1[j].y;
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						mj=P1[j].m;
						rhoj=P1[j].rho;
						ptypej=P1[j].p_type;
						ccj=P3[j].cc;
	
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
						if(tdist<search_range){
	
							tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							tmp_dwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
	
							// dwij
							tdwx=tmp_dwij*(xi-xj)/tdist;
							tdwy=tmp_dwij*(yi-yj)/tdist;
	
							// filter
							if((tcount%k_freq_filt)==0) tmp_flt+=mj/rhoj*tmp_wij;
	
							// strain rate
							if((k_fv_solve==1)&&(k_turbulence_model!=Laminar))
							{
								uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj);
								tmp_val=-0.5*mj*(rhoi+rhoj)*uij2;
								tmp_val/=(rhoi*rhoj*tdist*tdist);
								tmp_SR+=tmp_val*(xi-xj)*tdwx+tmp_val*(yi-yj)*tdwy;
							}
	
							// gradient rho (for delta-sph)
							if(k_delSPH_solve==Antuono)
							{
								apply_gradient_correction_2D(P3[i].Cm,tmp_wij,tdwx,tdwy,&tdwx,&tdwy);
	
								tmp_rhox+=-(rhoj-rhoi)*(mj/rhoj)*tdwx;
								tmp_rhoy+=-(rhoj-rhoi)*(mj/rhoj)*tdwy;
							}
	
							// normal gradient for curvature
							if(k_fs_solve)
							{
								tmp_ncx+=-(mj/rhoj)*(ccj-cci)*((ptypej!=0)&(ptypei==ptypej))*tdwx;
								tmp_ncy+=-(mj/rhoj)*(ccj-cci)*((ptypej!=0)&(ptypei==ptypej))*tdwy;
	
								Real nC_s,nC_sx,nC_sy,nC_st;
	
								nC_s=((ptypej!=0)&(ptypei!=ptypej));
								nC_st=nC_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
								nC_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tmp_dwij;
	
								nC_sx=nC_st*(xj-xi)/tdist;
								nC_sy=nC_st*(yj-yi)/tdist;
	
								tmp_nx+=nC_sx;
								tmp_ny+=nC_sy;
	
								/////////////////////////////////////////////////////////////////////////////////// (yhs)
								nxwi += tdwx*((ptypei!=0)&(ptypej==0));
								nywi += tdwy*((ptypei!=0)&(ptypej==0));
							}
						}

					}				
				}
			}
		}
	}


	// strain_rate
	if((k_fv_solve==1)&&(k_turbulence_model!=Laminar)) {
		tmp_SR=max(1e-20,tmp_SR);
		P2[i].SR=sqrt(tmp_SR);

		if(k_turbulence_model==SPS) tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*tmp_SR;
		P3[i].vis_t=tvis_t*rhoi;
	}

	// reference density
	if(k_dim==2) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600));
	if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));

	// filter
	if((tcount%k_freq_filt)==0) P1[i].flt_s=tmp_flt;

	// switch p_type
	if(k_switch_ptype==1) P1[i].p_type=functions_switch_ptype();

	// gradient density
	if(k_delSPH_solve==Antuono){
		P1[i].grad_rhox=tmp_rhox;
		P1[i].grad_rhoy=tmp_rhoy;
	}

	// normal gradient for surface tension
	if(k_fs_solve==1){

		P3[i].nx_c=tmp_ncx;
 	    P3[i].ny_c=tmp_ncy;

		Real tmpnmg=sqrt(tmp_ncx*tmp_ncx+tmp_ncy*tmp_ncy);
		P3[i].nmag_c=tmpnmg;
		if(tmpnmg<NORMAL_THRESHOLD){
			P3[i].nx_c=0;
			P3[i].ny_c=0;
			P3[i].nmag_c=1e-20;
		}

		nx_ci=P3[i].nx_c;
		ny_ci=P3[i].ny_c;
		nmag_ci=P3[i].nmag_c;

				// KERNEL_clc_normal_gradient2D ---------------
		P3[i].nx=tmp_nx;
	    P3[i].ny=tmp_ny;

		Real ntmpnmg=sqrt(tmp_nx*tmp_nx+tmp_ny*tmp_ny);
		P3[i].nmag=ntmpnmg;
		if(ntmpnmg<NORMAL_THRESHOLD){
			P3[i].nx=0;
			P3[i].ny=0;
			P3[i].nmag=1e-20;
		}

				/////////////////////////////////////////////////////////////////////////////////// (yhs)

		adddw = sqrt(nxwi*nxwi+nywi*nywi);
		if(adddw > 1E+5){
			nxwi /= adddw;
			nywi /= adddw;}
		else{
			nxwi = 0.0;
			nywi = 0.0;
		}

		nxti = (nx_ci/nmag_ci - (nx_ci/nmag_ci*nxwi+ny_ci/nmag_ci*nywi)*nxwi);
		nyti = (ny_ci/nmag_ci - (nx_ci/nmag_ci*nxwi+ny_ci/nmag_ci*nywi)*nywi);
		Real nmagti = sqrt(nxti*nxti+nyti*nyti);
		if(nmagti>1e-5){
			nxti /= nmagti;
			nyti /= nmagti;}
		else{
			nxti=0.0;
			nyti=0.0;
		}

	}

	P3[i].nx_u = nx_ci/nmag_ci;
	P3[i].ny_u = ny_ci/nmag_ci;
	P3[i].nx_w = nxwi;
	P3[i].ny_w = nywi;
	P3[i].nx_t = nxti;
	P3[i].ny_t = nyti;
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_CSF_solid(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi;
	Real uxi,uyi;
	Real tmp_h,tmp_A,search_range;
	Real nmagci, nxti, nyti, nxwi, nywi;
	Real tlpoint_0,tlpoint_1,tlpoint_2;

	xi=P1[i].x;
    yi=P1[i].y;
	nmagci=P3[i].nmag_c;
	nxti=P3[i].nx_t;
	nxwi=P3[i].nx_w;
	nyti=P3[i].ny_t;
	nywi=P3[i].ny_w;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}

	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	// 변수 초기화
	tlpoint_0=tlpoint_1=tlpoint_2=0.0;

	// 계산
	for(int_t y=-3;y<=3;y++){
		for(int_t x=-3;x<=3;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;

			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					
					if(P1[j].p_type<=1000){

						Real xj,yj,tdist;
						int_t ptypej;
	
						xj=P1[j].x;
						yj=P1[j].y;
						ptypej=P1[j].p_type;
	
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
						if(tdist<search_range*2){
	
							if(k_fs_solve)
							{
								/////////////////////////////////////////////////////////////////////////////////// (yhs)
								tlpoint_0 +=(ptypej==0);
								tlpoint_1 +=(ptypej==1);
								tlpoint_2 +=(ptypej==2);
							}
						}

					}
				}
			}
		}
	}

		/////////////////////////////////////////////////////////////////////////////////// (yhs)

	if(tlpoint_0*tlpoint_1*tlpoint_2>0){
		Real xdir, ydir;
		xdir = nxti*sin(60*PI/180)-nxwi*cos(60*PI/180);
		ydir = nyti*sin(60*PI/180)-nywi*cos(60*PI/180);
		P3[i].nx_tl = xdir * (nmagci>0.5)/sqrt(xdir*xdir+ydir*ydir+1E-12);
		P3[i].ny_tl = ydir * (nmagci>0.5)/sqrt(xdir*xdir+ydir*ydir+1E-12);
	}
}




__global__ void KERNEL_clc_prep3D_prep(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real pori=P1[i].DEMpor;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
	Real tvis_t=0.0,th;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	Real m_ref;

	if (ptypei==-3)	m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else if (ptypei==9) m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else if (ptypei==0) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;	
	else if (ptypei==3) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else if (ptypei==1) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

	if ((ptypei==9)&&(zi<0.05)) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	
	// reference density
	if(k_dim==2) P2[i].rho_ref=m_ref/((tmp_h/1.600)*(tmp_h/1.600));
	//if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));
	if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	if((k_rho_type==Continuity)){
		P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
		P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
		P1[i].m=m_ref;
	} 
}



__global__ void KERNEL_clc_prep3D_prep_sph(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real pori=P1[i].DEMpor;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
	Real tvis_t=0.0,th;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	Real m_ref;

	if (ptypei==-3)	m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else if (ptypei==9) m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else if (ptypei==0) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;	
	else if (ptypei==3) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else if (ptypei==1) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	else m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

	if ((ptypei==9)&&(zi<0.05)) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	
	// reference density
	if(k_dim==2) P2[i].rho_ref=m_ref/((tmp_h/1.600)*(tmp_h/1.600));
	//if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));
	if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	if((k_rho_type==Continuity)){
		P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
		P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
		P1[i].m=m_ref;
	} 
}

// __global__ void KERNEL_clc_prep3D_prep_sph(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2_sph) return;
// 	if(P1[i].i_type>i_type_crt) return;
// 	if(P1[i].p_type>1000) return;

// 	int_t icell,jcell,kcell;
// 	int_t ptypei=P1[i].p_type;

// 	Real xi,yi,zi;
// 	Real uxi,uyi,uzi;
	
// 	Real mi=P1[i].m;
// 	Real rhoi=P1[i].rho;
// 	Real pori=P1[i].DEMpor;
// 	Real cci=P3[i].cc;
// 	Real tmp_h,tmp_A,search_range;
// 	Real tmp_flt,tmp_SR;
// 	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
// 	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
// 	Real tvis_t=0.0,th;

// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;

// 	uxi=P1[i].ux;
// 	uyi=P1[i].uy;
// 	uzi=P1[i].uz;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	th=tmp_h*L_SPS;
// 	search_range=k_search_kappa*tmp_h;	// search range

// 	Real m_ref;

// 	if (ptypei==-3)	m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	else if (ptypei==9) m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	else if (ptypei==0) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;	
// 	else if (ptypei==3) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	else if (ptypei==1) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	else m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

// 	if ((ptypei==9)&&(zi<0.05)) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	
// 	// reference density
// 	if(k_dim==2) P2[i].rho_ref=m_ref/((tmp_h/1.600)*(tmp_h/1.600));
// 	//if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));
// 	if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 	if((k_rho_type==Continuity)){
// 		P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 		P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 		P1[i].m=m_ref;
// 	} 
// }

__global__ void KERNEL_clc_prep3D(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real pori=P1[i].DEMpor;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
	Real tvis_t=0.0,th;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	Real m_ref;

	// if (ptypei==-3)	m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else if (ptypei==9) m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else if (ptypei==0) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;	
	// else if (ptypei==3) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else if (ptypei==1) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

	// if ((ptypei==9)&&(zi<0.05)) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	
	// // reference density
	// if(k_dim==2) P2[i].rho_ref=m_ref/((tmp_h/1.600)*(tmp_h/1.600));
	// //if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));
	// if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// if((k_rho_type==Continuity)){
	// 	P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// 	P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// 	P1[i].m=m_ref;
	// } 

	// __syncthreads();

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_flt=tmp_SR=tmp_rhox=tmp_rhoy=tmp_rhoz=0.0;
	tmp_nx=tmp_ny=tmp_nz=tmp_ncx=tmp_ncy=tmp_ncz=0.0;

	// 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						
						if(P1[j].p_type<=1000){
							Real xj,yj,zj,uxj,uyj,uzj,uij2,mj,rhoj,rho_refj,ccj, tdwx,tdwy,tdwz,tmp_wij,tmp_dwij,tdist,tmp_val;
							int_t ptypej;
	
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							rhoj=P1[j].rho;
							rho_refj=P2[j].rho_ref;
							ptypej=P1[j].p_type;
							ccj=P3[j].cc;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
							if(tdist<search_range){
								tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								tmp_dwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
	
								tdwx=tmp_dwij*(xi-xj)/tdist;
								tdwy=tmp_dwij*(yi-yj)/tdist;
								tdwz=tmp_dwij*(zi-zj)/tdist;
	
								// filter
								if((tcount%k_freq_filt)==0){

									// if ((ptypej==0)|(ptypej==9)){
									// 	tmp_flt+=mj/DENSITY_WATER*tmp_wij;
									// }
									// else{
										tmp_flt+=mj/rhoj*tmp_wij;
									// }


								} 
								// filter
								//if((tcount%k_freq_filt)==0) tmp_flt+=(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*tmp_wij;
	
								// strain rate
								if((k_fv_solve==1)&&(k_turbulence_model!=Laminar))
								{
									uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);
									tmp_val=-0.5*mj*(rhoi+rhoj)*uij2;
									tmp_val/=(rhoi*rhoj*tdist*tdist);
									tmp_SR+=tmp_val*(xi-xj)*tdwx+tmp_val*(yi-yj)*tdwy+tmp_val*(zi-zj)*tdwz;
								}
	
								// gradient rho (for delta-sph)
								if(k_delSPH_solve==Antuono)
								{
									apply_gradient_correction_3D(P3[i].Cm,tmp_wij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);

									Real rhoi_ref=P2[i].rho_ref;
									Real rhoj_ref=P2[j].rho_ref;
		
									tmp_rhox+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwx;
									tmp_rhoy+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwy;
									tmp_rhoz+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwz;
								}
	
								// normal gradient for curvature
								if(k_fs_solve)
								{
									int wall;
									wall=1;
									tmp_ncx+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwx;
									tmp_ncy+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwy;
									tmp_ncz+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwz;
	
									Real nC_s,nC_sx,nC_sy,nC_sz,nC_st;
	
									nC_s=(ptypei!=ptypej);
									nC_st=nC_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
									nC_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tmp_dwij;
	
									nC_sx=nC_st*(xj-xi)/tdist;
									nC_sy=nC_st*(yj-yi)/tdist;
									nC_sz=nC_st*(zj-zi)/tdist;
	
									tmp_nx+=nC_sx;
									tmp_ny+=nC_sy;
									tmp_nz+=nC_sz;
								}
							}
						}
					}
				}
			}
		}
	}

	// strain_rate
	if((k_fv_solve==1)&&(k_turbulence_model!=Laminar)) {
		tmp_SR=max(1e-20,tmp_SR);
		P2[i].SR=sqrt(tmp_SR);

		if(k_turbulence_model==SPS) tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*tmp_SR;
		P3[i].vis_t=tvis_t*rhoi;
	}

	// filter
	if((tcount%k_freq_filt)==0) P1[i].flt_s=tmp_flt;

	// gradient density
	if(k_delSPH_solve==Antuono){
		P1[i].grad_rhox=tmp_rhox;
		P1[i].grad_rhoy=tmp_rhoy;
		P1[i].grad_rhoz=tmp_rhoz;

		P1[i].test1=tmp_rhoy;
		P1[i].test5=tmp_rhoz;

	}

	// if((k_rho_type==Continuity) && (tcount==0)){
	// 	P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// 	P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// } 

	// P1[i].m=P1[i].rho*P1[i].vol;

	//if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));

	// normal gradient for surface tension
	if(k_fs_solve==1){
		Real tmpnmg=sqrt(tmp_ncx*tmp_ncx+tmp_ncy*tmp_ncy+tmp_ncz*tmp_ncz);

		P3[i].nx_c=tmp_ncx;
		P3[i].ny_c=tmp_ncy;
		P3[i].nz_c=tmp_ncz;

		P3[i].nmag_c=tmpnmg;
		if(tmpnmg<NORMAL_THRESHOLD){
			P3[i].nx_c=0;
			P3[i].ny_c=0;
			P3[i].nz_c=0;
			P3[i].nmag_c=1e-20;
		}

		// KERNEL_clc_normal_gradient3D ---------------
		P3[i].nx=tmp_nx;
		P3[i].ny=tmp_ny;
		P3[i].nz=tmp_nz;

		Real ntmpnmg=sqrt(tmp_nx*tmp_nx+tmp_ny*tmp_ny+tmp_nz*tmp_nz);
		P3[i].nmag=ntmpnmg;
		if(ntmpnmg<NORMAL_THRESHOLD){
			P3[i].nx=0;
			P3[i].ny=0;
			P3[i].nz=0;
			P3[i].nmag=1e-20;
		}
	}
}



__global__ void KERNEL_clc_prep3D_sph(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real pori=P1[i].DEMpor;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
	Real tvis_t=0.0,th;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	Real m_ref;

	// if (ptypei==-3)	m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else if (ptypei==9) m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else if (ptypei==0) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;	
	// else if (ptypei==3) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else if (ptypei==1) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	// else m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

	// if ((ptypei==9)&&(zi<0.05)) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	
	// // reference density
	// if(k_dim==2) P2[i].rho_ref=m_ref/((tmp_h/1.600)*(tmp_h/1.600));
	// //if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));
	// if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// if((k_rho_type==Continuity)){
	// 	P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// 	P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// 	P1[i].m=m_ref;
	// } 

	// __syncthreads();

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_flt=tmp_SR=tmp_rhox=tmp_rhoy=tmp_rhoz=0.0;
	tmp_nx=tmp_ny=tmp_nz=tmp_ncx=tmp_ncy=tmp_ncz=0.0;

	// 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						
						if(P1[j].p_type<=1000){
							Real xj,yj,zj,uxj,uyj,uzj,uij2,mj,rhoj,rho_refj,ccj, tdwx,tdwy,tdwz,tmp_wij,tmp_dwij,tdist,tmp_val;
							int_t ptypej;
	
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							rhoj=P1[j].rho;
							rho_refj=P2[j].rho_ref;
							ptypej=P1[j].p_type;
							ccj=P3[j].cc;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
							if(tdist<search_range){
								tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								tmp_dwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
	
								tdwx=tmp_dwij*(xi-xj)/tdist;
								tdwy=tmp_dwij*(yi-yj)/tdist;
								tdwz=tmp_dwij*(zi-zj)/tdist;
	
								// filter
								if((tcount%(k_freq_filt*k_decouple_stride))==0){

									// if ((ptypej==0)|(ptypej==9)){
									// 	tmp_flt+=mj/DENSITY_WATER*tmp_wij;
									// }
									// else{
										tmp_flt+=mj/rhoj*tmp_wij;
									// }


								} 
								// filter
								//if((tcount%(k_freq_filt*k_decouple_stride))==0) tmp_flt+=(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*tmp_wij;
	
								// strain rate
								if((k_fv_solve==1)&&(k_turbulence_model!=Laminar))
								{
									uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);
									tmp_val=-0.5*mj*(rhoi+rhoj)*uij2;
									tmp_val/=(rhoi*rhoj*tdist*tdist);
									tmp_SR+=tmp_val*(xi-xj)*tdwx+tmp_val*(yi-yj)*tdwy+tmp_val*(zi-zj)*tdwz;
								}
	
								// gradient rho (for delta-sph)
								if(k_delSPH_solve==Antuono)
								{
									apply_gradient_correction_3D(P3[i].Cm,tmp_wij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);

									Real rhoi_ref=P2[i].rho_ref;
									Real rhoj_ref=P2[j].rho_ref;
		
									tmp_rhox+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwx;
									tmp_rhoy+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwy;
									tmp_rhoz+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwz;
								}
	
								// normal gradient for curvature
								if(k_fs_solve)
								{
									int wall;
									wall=1;
									tmp_ncx+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwx;
									tmp_ncy+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwy;
									tmp_ncz+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwz;
	
									Real nC_s,nC_sx,nC_sy,nC_sz,nC_st;
	
									nC_s=(ptypei!=ptypej);
									nC_st=nC_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
									nC_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tmp_dwij;
	
									nC_sx=nC_st*(xj-xi)/tdist;
									nC_sy=nC_st*(yj-yi)/tdist;
									nC_sz=nC_st*(zj-zi)/tdist;
	
									tmp_nx+=nC_sx;
									tmp_ny+=nC_sy;
									tmp_nz+=nC_sz;
								}
							}
						}
					}
				}
			}
		}
	}

	// strain_rate
	if((k_fv_solve==1)&&(k_turbulence_model!=Laminar)) {
		tmp_SR=max(1e-20,tmp_SR);
		P2[i].SR=sqrt(tmp_SR);

		if(k_turbulence_model==SPS) tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*tmp_SR;
		P3[i].vis_t=tvis_t*rhoi;
	}

	// filter
	if((tcount%(k_freq_filt*k_decouple_stride))==0) P1[i].flt_s=tmp_flt;

	// gradient density
	if(k_delSPH_solve==Antuono){
		P1[i].grad_rhox=tmp_rhox;
		P1[i].grad_rhoy=tmp_rhoy;
		P1[i].grad_rhoz=tmp_rhoz;

		P1[i].test1=tmp_rhoy;
		P1[i].test5=tmp_rhoz;

	}

	// if((k_rho_type==Continuity) && (tcount==0)){
	// 	P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// 	P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
	// } 

	// P1[i].m=P1[i].rho*P1[i].vol;

	//if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));

	// normal gradient for surface tension
	if(k_fs_solve==1){
		Real tmpnmg=sqrt(tmp_ncx*tmp_ncx+tmp_ncy*tmp_ncy+tmp_ncz*tmp_ncz);

		P3[i].nx_c=tmp_ncx;
		P3[i].ny_c=tmp_ncy;
		P3[i].nz_c=tmp_ncz;

		P3[i].nmag_c=tmpnmg;
		if(tmpnmg<NORMAL_THRESHOLD){
			P3[i].nx_c=0;
			P3[i].ny_c=0;
			P3[i].nz_c=0;
			P3[i].nmag_c=1e-20;
		}

		// KERNEL_clc_normal_gradient3D ---------------
		P3[i].nx=tmp_nx;
		P3[i].ny=tmp_ny;
		P3[i].nz=tmp_nz;

		Real ntmpnmg=sqrt(tmp_nx*tmp_nx+tmp_ny*tmp_ny+tmp_nz*tmp_nz);
		P3[i].nmag=ntmpnmg;
		if(ntmpnmg<NORMAL_THRESHOLD){
			P3[i].nx=0;
			P3[i].ny=0;
			P3[i].nz=0;
			P3[i].nmag=1e-20;
		}
	}
}


// __global__ void KERNEL_clc_prep3D_sph(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2_sph) return;
// 	if(P1[i].i_type>i_type_crt) return;
// 	if(P1[i].p_type>1000) return;

// 	int_t icell,jcell,kcell;
// 	int_t ptypei=P1[i].p_type;

// 	Real xi,yi,zi;
// 	Real uxi,uyi,uzi;
	
// 	Real mi=P1[i].m;
// 	Real rhoi=P1[i].rho;
// 	Real pori=P1[i].DEMpor;
// 	Real cci=P3[i].cc;
// 	Real tmp_h,tmp_A,search_range;
// 	Real tmp_flt,tmp_SR;
// 	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
// 	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
// 	Real tvis_t=0.0,th;

// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;

// 	uxi=P1[i].ux;
// 	uyi=P1[i].uy;
// 	uzi=P1[i].uz;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	th=tmp_h*L_SPS;
// 	search_range=k_search_kappa*tmp_h;	// search range

// 	Real m_ref;

// 	// if (ptypei==-3)	m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	// else if (ptypei==9) m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	// else if (ptypei==0) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;	
// 	// else if (ptypei==3) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	// else if (ptypei==1) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	// else m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

// 	// if ((ptypei==9)&&(zi<0.05)) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
	
// 	// // reference density
// 	// if(k_dim==2) P2[i].rho_ref=m_ref/((tmp_h/1.600)*(tmp_h/1.600));
// 	// //if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));
// 	// if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 	// if((k_rho_type==Continuity)){
// 	// 	P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 	// 	P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 	// 	P1[i].m=m_ref;
// 	// } 

// 	// __syncthreads();

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

// 	// 초기화
// 	tmp_flt=tmp_SR=tmp_rhox=tmp_rhoy=tmp_rhoz=0.0;
// 	tmp_nx=tmp_ny=tmp_nz=tmp_ncx=tmp_ncy=tmp_ncz=0.0;

// 	// 계산
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

// 			//	if(k<0||k>=k_num_cells-1) continue;
// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
						
// 						if(P1[j].p_type<=1000){
// 							Real xj,yj,zj,uxj,uyj,uzj,uij2,mj,rhoj,rho_refj,ccj, tdwx,tdwy,tdwz,tmp_wij,tmp_dwij,tdist,tmp_val;
// 							int_t ptypej;
	
// 							xj=P1[j].x;
// 							yj=P1[j].y;
// 							zj=P1[j].z;
// 							uxj=P1[j].ux;
// 							uyj=P1[j].uy;
// 							uzj=P1[j].uz;
// 							mj=P1[j].m;
// 							rhoj=P1[j].rho;
// 							rho_refj=P2[j].rho_ref;
// 							ptypej=P1[j].p_type;
// 							ccj=P3[j].cc;
	
// 							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
// 							if(tdist<search_range){
// 								tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
// 								tmp_dwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
	
// 								tdwx=tmp_dwij*(xi-xj)/tdist;
// 								tdwy=tmp_dwij*(yi-yj)/tdist;
// 								tdwz=tmp_dwij*(zi-zj)/tdist;
	
// 								// filter
// 								if((tcount%k_freq_filt)==0){

// 									// if ((ptypej==0)|(ptypej==9)){
// 									// 	tmp_flt+=mj/DENSITY_WATER*tmp_wij;
// 									// }
// 									// else{
// 										tmp_flt+=mj/rhoj*tmp_wij;
// 									// }


// 								} 
// 								// filter
// 								//if((tcount%k_freq_filt)==0) tmp_flt+=(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*tmp_wij;
	
// 								// strain rate
// 								if((k_fv_solve==1)&&(k_turbulence_model!=Laminar))
// 								{
// 									uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);
// 									tmp_val=-0.5*mj*(rhoi+rhoj)*uij2;
// 									tmp_val/=(rhoi*rhoj*tdist*tdist);
// 									tmp_SR+=tmp_val*(xi-xj)*tdwx+tmp_val*(yi-yj)*tdwy+tmp_val*(zi-zj)*tdwz;
// 								}
	
// 								// gradient rho (for delta-sph)
// 								if(k_delSPH_solve==Antuono)
// 								{
// 									apply_gradient_correction_3D_sph(P3[i].Cm,tmp_wij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);

// 									Real rhoi_ref=P2[i].rho_ref;
// 									Real rhoj_ref=P2[j].rho_ref;
		
// 									tmp_rhox+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwx;
// 									tmp_rhoy+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwy;
// 									tmp_rhoz+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwz;
// 								}
	
// 								// normal gradient for curvature
// 								if(k_fs_solve)
// 								{
// 									int wall;
// 									wall=1;
// 									tmp_ncx+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwx;
// 									tmp_ncy+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwy;
// 									tmp_ncz+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwz;
	
// 									Real nC_s,nC_sx,nC_sy,nC_sz,nC_st;
	
// 									nC_s=(ptypei!=ptypej);
// 									nC_st=nC_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
// 									nC_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tmp_dwij;
	
// 									nC_sx=nC_st*(xj-xi)/tdist;
// 									nC_sy=nC_st*(yj-yi)/tdist;
// 									nC_sz=nC_st*(zj-zi)/tdist;
	
// 									tmp_nx+=nC_sx;
// 									tmp_ny+=nC_sy;
// 									tmp_nz+=nC_sz;
// 								}
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}

// 	// strain_rate
// 	if((k_fv_solve==1)&&(k_turbulence_model!=Laminar)) {
// 		tmp_SR=max(1e-20,tmp_SR);
// 		P2[i].SR=sqrt(tmp_SR);

// 		if(k_turbulence_model==SPS) tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*tmp_SR;
// 		P3[i].vis_t=tvis_t*rhoi;
// 	}

// 	// filter
// 	if((tcount%k_freq_filt)==0) P1[i].flt_s=tmp_flt;

// 	// gradient density
// 	if(k_delSPH_solve==Antuono){
// 		P1[i].grad_rhox=tmp_rhox;
// 		P1[i].grad_rhoy=tmp_rhoy;
// 		P1[i].grad_rhoz=tmp_rhoz;

// 		P1[i].test1=tmp_rhoy;
// 		P1[i].test5=tmp_rhoz;

// 	}

// 	// if((k_rho_type==Continuity) && (tcount==0)){
// 	// 	P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 	// 	P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 	// } 

// 	// P1[i].m=P1[i].rho*P1[i].vol;

// 	//if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));

// 	// normal gradient for surface tension
// 	if(k_fs_solve==1){
// 		Real tmpnmg=sqrt(tmp_ncx*tmp_ncx+tmp_ncy*tmp_ncy+tmp_ncz*tmp_ncz);

// 		P3[i].nx_c=tmp_ncx;
// 		P3[i].ny_c=tmp_ncy;
// 		P3[i].nz_c=tmp_ncz;

// 		P3[i].nmag_c=tmpnmg;
// 		if(tmpnmg<NORMAL_THRESHOLD){
// 			P3[i].nx_c=0;
// 			P3[i].ny_c=0;
// 			P3[i].nz_c=0;
// 			P3[i].nmag_c=1e-20;
// 		}

// 		// KERNEL_clc_normal_gradient3D ---------------
// 		P3[i].nx=tmp_nx;
// 		P3[i].ny=tmp_ny;
// 		P3[i].nz=tmp_nz;

// 		Real ntmpnmg=sqrt(tmp_nx*tmp_nx+tmp_ny*tmp_ny+tmp_nz*tmp_nz);
// 		P3[i].nmag=ntmpnmg;
// 		if(ntmpnmg<NORMAL_THRESHOLD){
// 			P3[i].nx=0;
// 			P3[i].ny=0;
// 			P3[i].nz=0;
// 			P3[i].nmag=1e-20;
// 		}
// 	}
// }
// __global__ void KERNEL_clc_prep3D(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type>i_type_crt) return;
// 	if(P1[i].p_type>1000) return;

// 	int_t icell,jcell,kcell;
// 	int_t ptypei=P1[i].p_type;

// 	Real xi,yi,zi;
// 	Real uxi,uyi,uzi;
	
// 	Real mi=P1[i].m;
// 	Real rhoi=P1[i].rho;
// 	Real pori=P1[i].DEMpor;
// 	Real cci=P3[i].cc;
// 	Real tmp_h,tmp_A,search_range;
// 	Real tmp_flt,tmp_SR;
// 	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
// 	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
// 	Real tvis_t=0.0,th;

// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;

// 	uxi=P1[i].ux;
// 	uyi=P1[i].uy;
// 	uzi=P1[i].uz;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	th=tmp_h*L_SPS;
// 	search_range=k_search_kappa*tmp_h;	// search range


	

// 	Real m_ref;

// 	if (ptypei==-3)	m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	else if (ptypei==9) m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	else if (ptypei==0) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;	
// 	else if (ptypei==3) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	else if (ptypei==1) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;
// 	else m_ref=DENSITY_WATER*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

// 	if ((ptypei==9)&&(zi<0.05)) m_ref=DENSITY_AIR*(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*pori;

// 	// reference mass
	
// 	// reference density
// 	if(k_dim==2) P2[i].rho_ref=m_ref/((tmp_h/1.600)*(tmp_h/1.600));
// 	//if(k_dim==3) P2[i].rho_ref=mi/((tmp_h/1.600)*(tmp_h/1.600)*(tmp_h/1.600));
// 	if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 	if((k_rho_type==Continuity)){
// 		P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 		P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 		P1[i].m=m_ref;
// 	} 

// 	__syncthreads();


// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

// 	// 초기화
// 	tmp_flt=tmp_SR=tmp_rhox=tmp_rhoy=tmp_rhoz=0.0;
// 	tmp_nx=tmp_ny=tmp_nz=tmp_ncx=tmp_ncy=tmp_ncz=0.0;


// 	// 계산
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

// 			//	if(k<0||k>=k_num_cells-1) continue;
// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
						
// 						if(P1[j].p_type<=1000){
// 							Real xj,yj,zj,uxj,uyj,uzj,uij2,mj,rhoj,rho_refj,ccj, tdwx,tdwy,tdwz,tmp_wij,tmp_dwij,tdist,tmp_val;
// 							int_t ptypej;
	
// 							xj=P1[j].x;
// 							yj=P1[j].y;
// 							zj=P1[j].z;
// 							uxj=P1[j].ux;
// 							uyj=P1[j].uy;
// 							uzj=P1[j].uz;
// 							mj=P1[j].m;
// 							rhoj=P1[j].rho;
// 							rho_refj=P2[j].rho_ref;
// 							ptypej=P1[j].p_type;
// 							ccj=P3[j].cc;
	
// 							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
// 							if(tdist<search_range){
// 								tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
// 								tmp_dwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
	
// 								tdwx=tmp_dwij*(xi-xj)/tdist;
// 								tdwy=tmp_dwij*(yi-yj)/tdist;
// 								tdwz=tmp_dwij*(zi-zj)/tdist;
	
// 								// filter
// 								if((tcount%k_freq_filt)==0){

// 									// if ((ptypej==0)|(ptypej==9)){
// 									// 	tmp_flt+=mj/DENSITY_WATER*tmp_wij;
// 									// }
// 									// else{
// 										tmp_flt+=mj/rhoj*tmp_wij;
// 									// }


// 								} 
// 								// filter
// 								//if((tcount%k_freq_filt)==0) tmp_flt+=(tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6)*tmp_wij;
	
// 								// strain rate
// 								if((k_fv_solve==1)&&(k_turbulence_model!=Laminar))
// 								{
// 									uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);
// 									tmp_val=-0.5*mj*(rhoi+rhoj)*uij2;
// 									tmp_val/=(rhoi*rhoj*tdist*tdist);
// 									tmp_SR+=tmp_val*(xi-xj)*tdwx+tmp_val*(yi-yj)*tdwy+tmp_val*(zi-zj)*tdwz;
// 								}
	
// 								// gradient rho (for delta-sph)
// 								if(k_delSPH_solve==Antuono)
// 								{
// 									apply_gradient_correction_3D(P3[i].Cm,tmp_wij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);

// 									Real rhoi_ref=P2[i].rho_ref;
// 									Real rhoj_ref=P2[j].rho_ref;
		
// 									tmp_rhox+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwx;
// 									tmp_rhoy+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwy;
// 									tmp_rhoz+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwz;
// 								}
	
// 								// normal gradient for curvature
// 								if(k_fs_solve)
// 								{
// 									int wall;
// 									wall=1;
// 									tmp_ncx+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwx;
// 									tmp_ncy+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwy;
// 									tmp_ncz+=-(mj/rhoj)*(ccj-cci)*wall*(ptypei==ptypej)*tdwz;
	
// 									Real nC_s,nC_sx,nC_sy,nC_sz,nC_st;
	
// 									nC_s=(ptypei!=ptypej);
// 									nC_st=nC_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
// 									nC_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tmp_dwij;
	
// 									nC_sx=nC_st*(xj-xi)/tdist;
// 									nC_sy=nC_st*(yj-yi)/tdist;
// 									nC_sz=nC_st*(zj-zi)/tdist;
	
// 									tmp_nx+=nC_sx;
// 									tmp_ny+=nC_sy;
// 									tmp_nz+=nC_sz;
// 								}
// 							}

// 						}
						
						
// 					}
// 				}
// 			}
// 		}
// 	}

// 	// strain_rate
// 	if((k_fv_solve==1)&&(k_turbulence_model!=Laminar)) {
// 		tmp_SR=max(1e-20,tmp_SR);
// 		P2[i].SR=sqrt(tmp_SR);

// 		if(k_turbulence_model==SPS) tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*tmp_SR;
// 		P3[i].vis_t=tvis_t*rhoi;
// 	}


	

// 	// filter
// 	if((tcount%k_freq_filt)==0) P1[i].flt_s=tmp_flt;


// 	// switch p_type
// 	if(k_switch_ptype==1) P1[i].p_type=functions_switch_ptype();

// 	// gradient density
// 	if(k_delSPH_solve==Antuono){
// 		P1[i].grad_rhox=tmp_rhox;
// 		P1[i].grad_rhoy=tmp_rhoy;
// 		P1[i].grad_rhoz=tmp_rhoz;

// 		P1[i].test1=tmp_rhoy;
// 		P1[i].test5=tmp_rhoz;

// 	}



// 	// if((k_rho_type==Continuity) && (tcount==0)){
// 	// 	P2[i].rho0=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 	// 	P1[i].rho=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));
// 	// } 

// 	// P1[i].m=P1[i].rho*P1[i].vol;





// 	//if(k_dim==3) P2[i].rho_ref=m_ref/((tmp_h/1.6)*(tmp_h/1.6)*(tmp_h/1.6));

// 	// normal gradient for surface tension
// 	if(k_fs_solve==1){

// 		Real tmpnmg=sqrt(tmp_ncx*tmp_ncx+tmp_ncy*tmp_ncy+tmp_ncz*tmp_ncz);

// 		P3[i].nx_c=tmp_ncx;
// 		P3[i].ny_c=tmp_ncy;
// 		P3[i].nz_c=tmp_ncz;

// 		P3[i].nmag_c=tmpnmg;
// 		if(tmpnmg<NORMAL_THRESHOLD){
// 			P3[i].nx_c=0;
// 			P3[i].ny_c=0;
// 			P3[i].nz_c=0;
// 			P3[i].nmag_c=1e-20;
// 		}

// 		// KERNEL_clc_normal_gradient3D ---------------
// 		P3[i].nx=tmp_nx;
// 		P3[i].ny=tmp_ny;
// 		P3[i].nz=tmp_nz;

// 		Real ntmpnmg=sqrt(tmp_nx*tmp_nx+tmp_ny*tmp_ny+tmp_nz*tmp_nz);
// 		P3[i].nmag=ntmpnmg;
// 		if(ntmpnmg<NORMAL_THRESHOLD){
// 			P3[i].nx=0;
// 			P3[i].ny=0;
// 			P3[i].nz=0;
// 			P3[i].nmag=1e-20;
// 		}
// 	}
// }




////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_prep3D_1(int_t*g_str,int_t*g_end,part1*P1, part2*P2, part3*P3, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real cci=P3[i].cc;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_SR;
	Real tmp_rhox,tmp_rhoy,tmp_rhoz;
	Real tmp_ncx, tmp_ncy, tmp_ncz, tmp_nx, tmp_ny, tmp_nz;
	Real tvis_t=0.0,th;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	th=tmp_h*L_SPS;
	search_range=k_search_kappa*tmp_h;	// search range

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_flt=tmp_SR=tmp_rhox=tmp_rhoy=tmp_rhoz=0.0;
	tmp_nx=tmp_ny=tmp_nz=tmp_ncx=tmp_ncy=tmp_ncz=0.0;


	// 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
					if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,uxj,uyj,uzj,uij2,mj,rhoj,ccj, tdwx,tdwy,tdwz,tmp_wij,tmp_dwij,tdist,tmp_val;
						int_t ptypej;

						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						uzj=P1[j].uz;
						mj=P1[j].m;
						rhoj=P1[j].rho;
						ptypej=P1[j].p_type;
						ccj=P3[j].cc;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;

						if(tdist<search_range){
							tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							tmp_dwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							tdwx=tmp_dwij*(xi-xj)/tdist;
							tdwy=tmp_dwij*(yi-yj)/tdist;
							tdwz=tmp_dwij*(zi-zj)/tdist;

							// filter
							if((tcount%k_freq_filt)==0) tmp_flt+=mj/rhoj*tmp_wij;

							// strain rate
							if((k_fv_solve==1)&&(k_turbulence_model!=Laminar))
							{
								uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);
								tmp_val=-0.5*mj*(rhoi+rhoj)*uij2;
								tmp_val/=(rhoi*rhoj*tdist*tdist);
								tmp_SR+=tmp_val*(xi-xj)*tdwx+tmp_val*(yi-yj)*tdwy+tmp_val*(zi-zj)*tdwz;
							}

							// gradient rho (for delta-sph)
							if(k_delSPH_solve==Antuono)
							{
								apply_gradient_correction_3D(P3[i].Cm,tmp_wij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);

								Real rhoi_ref=P2[i].rho_ref;
								Real rhoj_ref=P2[j].rho_ref;
	
								tmp_rhox+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwx;
								tmp_rhoy+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwy;
								tmp_rhoz+=-(rhoj/rhoj_ref-rhoi/rhoi_ref)*(mj/rhoj)*tdwz;
							}

							// normal gradient for curvature
							if((k_fs_solve)&(k_surf_model==2))
							{
								tmp_ncx+=-(mj/rhoj)*(ccj-cci)*(ptypei==ptypej)*tdwx;
								tmp_ncy+=-(mj/rhoj)*(ccj-cci)*(ptypei==ptypej)*tdwy;
								tmp_ncz+=-(mj/rhoj)*(ccj-cci)*(ptypei==ptypej)*tdwz;

								Real nC_s,nC_sx,nC_sy,nC_sz,nC_st;

								nC_s=(ptypei!=ptypej);
								nC_st=nC_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
								nC_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tmp_dwij;

								nC_sx=nC_st*(xj-xi)/tdist;
								nC_sy=nC_st*(yj-yi)/tdist;
								nC_sz=nC_st*(zj-zi)/tdist;

								tmp_nx+=nC_sx;
								tmp_ny+=nC_sy;
								tmp_nz+=nC_sz;
							}
						}
					}
				}
			}
		}
	}

	// strain_rate
	if((k_fv_solve==1)&&(k_turbulence_model!=Laminar)) {
		tmp_SR=max(1e-20,tmp_SR);
		P2[i].SR=sqrt(tmp_SR);

		if(k_turbulence_model==SPS) tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*tmp_SR;
		P3[i].vis_t=tvis_t*rhoi;
	}

	// reference density
	if(tcount==0){
		P2[i].rho_ref=P1[i].rho;
	}

	// filter
	if((tcount%k_freq_filt)==0) P1[i].flt_s=tmp_flt;

	// // switch p_type
	// if(k_switch_ptype==1) P1[i].p_type=functions_switch_ptype();

	// gradient density
	if(k_delSPH_solve==Antuono){
		P1[i].grad_rhox=tmp_rhox;
		P1[i].grad_rhoy=tmp_rhoy;
		P1[i].grad_rhoz=tmp_rhoz;
	}

	// normal gradient for surface tension
	if((k_fs_solve)&(k_surf_model==2)){

		P3[i].nx_c=tmp_ncx;
		P3[i].ny_c=tmp_ncy;
		P3[i].nz_c=tmp_ncz;

		Real tmpnmg=sqrt(tmp_ncx*tmp_ncx+tmp_ncy*tmp_ncy+tmp_ncz*tmp_ncz);
		P3[i].nmag_c=tmpnmg;
		if(tmpnmg<NORMAL_THRESHOLD){
			P3[i].nx_c=0;
			P3[i].ny_c=0;
			P3[i].nz_c=0;
			P3[i].nmag_c=1e-20;
		}

		// KERNEL_clc_normal_gradient3D ---------------
		P3[i].nx=tmp_nx;
		P3[i].ny=tmp_ny;
		P3[i].nz=tmp_nz;

		Real ntmpnmg=sqrt(tmp_nx*tmp_nx+tmp_ny*tmp_ny+tmp_nz*tmp_nz);
		P3[i].nmag=ntmpnmg;
		if(ntmpnmg<NORMAL_THRESHOLD){
			P3[i].nx=0;
			P3[i].ny=0;
			P3[i].nz=0;
			P3[i].nmag=1e-20;
		}
	}
}




__global__ void KERNEL_clc_prep2D_coupling(int_t*g_str,int_t*g_end,part1*P1, int_t tcount)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi;
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real tmp_h,tmp_A,search_range;
	Real tmp_flt,tmp_por;				//tmp_filter, tmp_porosity(DEM calculation)


	xi=P1[i].x;
	yi=P1[i].y;
	


	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	
	search_range=k_search_kappa*tmp_h;	// search range

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}

	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;


	// 초기화
	tmp_por=0.0;
	tmp_flt=0.0;


	// 계산
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;

			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){

					Real tmp_wij,tdist;
					Real xj,yj;

					xj=P1[j].x;
					yj=P1[j].y;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
					if(tdist<search_range){

						tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);


						Real mj,rhoj,radj;
						int_t ptypej;

						mj=P1[j].m;
						rhoj=P1[j].rho;
						radj=P1[j].rad;
						ptypej=P1[j].p_type;

						if(ptypei>1000){
							if(ptypej<=1000){
								tmp_flt+=mj/rhoj*tmp_wij;

							}	
							if(ptypej>1000)
								tmp_por+=tmp_wij*3.14159*radj*radj;
						}
					
						if(ptypei<=1000){
							if(ptypej>1000)		tmp_por+=tmp_wij*3.14159*radj*radj;
						}

					}	

				}
			}
		}
	}

	// filter
	if(ptypei>1000){
		P1[i].flt_s=tmp_flt;
	}

	// porosity
	P1[i].DEMpor=1.0-tmp_por;
	//P1[i].DEMpor=1.0-(tmp_por/(tmp_SPHflt+1.e-20));

}

__global__ void KERNEL_clc_prep3D_coupling(int_t*g_str,int_t*g_end,part1*P1, part2*P2, int_t tcount, Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1[i].p_type;

	Real xi,yi,zi;
	Real mi=P1[i].m;
	Real rhoi=P1[i].rho;
	Real tmp_h,tmp_A,search_range;
	Real tmp_SPHflt, tmp_DEMflt,tmp_DEMfltd,tmp_por;				//tmp_filter, tmp_porosity(DEM calculation)
	Real tmp_DEMfltd_2;


	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;


	tmp_h=1*P1[i].h;
	tmp_A=calc_tmpA(1.0*tmp_h);
	
	search_range=k_search_kappa*1.0*tmp_h;	// search range


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_por=0.0;
	tmp_SPHflt=0.0;
	tmp_DEMflt=0.0;
	tmp_DEMfltd=0.0;
	tmp_DEMfltd_2=0.0;
	
	// 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						
						Real tmp_wij,tdist;
						Real xj,yj,zj;


						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;

						
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
						if(tdist<search_range){

							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);


							Real mj,rhoj,radj;
							int_t ptypej;

							mj=P1[j].m;
							rhoj=P1[j].rho;
							radj=P1[j].rad;
							ptypej=P1[j].p_type;

							if(ptypei>1000){
								//if(ptypej<=1000){
								if((ptypej<=1000) && (ptypej!=0) && (ptypej!=MOVING) && (ptypej==1)){
									tmp_DEMfltd+=mj/rhoj*tmp_wij;
									tmp_DEMfltd_2+=mj/rhoj*tmp_wij;
								}	
								if((ptypej<=1000) && (ptypej!=0) && (ptypej!=MOVING) && (ptypej==3)){
									tmp_DEMflt+=mj/rhoj*tmp_wij;
								}	
								if(ptypej>1000) tmp_por+=tmp_wij*4.18879*radj*radj*radj;
							}

							if (ptypei<=1000){

								if((ptypej<=1000)){
								//if((ptypej<=1000)){
									tmp_SPHflt+=mj/rhoj*tmp_wij*(ptypej==1);
									//tmp_SPHflt+=mj/rhoj*1*(ptypej==1);	//simmple summation
								}
								if(ptypej>1000){

									tmp_por+=tmp_wij*4.18879*radj*radj*radj;
									//tmp_por+=1*4.18879*radj*radj*radj;	//simple summation

								}	  

							}
						
							

						}	

					}
				}
			}
		}
	}

	// filter
	if(ptypei>1000){
		P1[i].flt_s=tmp_DEMflt;
		P1[i].flt_sd=tmp_DEMfltd;
		P1[i].flt_sd_2=tmp_DEMfltd_2;
		P1[i].DEMpor=1.0-(tmp_por/(tmp_DEMflt+tmp_DEMfltd+1.0e-20));
	}
	
	if(ptypei<=1000) {


		//P1[i].DEMpor=1.0-(zi<0.186)*(zi>0.00)*(tmp_por/(tmp_SPHflt+1e-20));
		P1[i].DEMpor=1.0-(tmp_por/(tmp_SPHflt+1e-20));
		

		

	}

}


__global__ void KERNEL_clc_prep3D_coupling_sph(int_t*g_str_sph,int_t*g_end_sph,part1*P1_sph, part2*P2_sph, 
												int_t*g_str_dem,int_t*g_end_dem,part1*P1_dem, part2*P2_dem, 
												int_t tcount, Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1_sph[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1_sph[i].p_type;

	P1_sph[i].dDEMpor_prev = P1_sph[i].dDEMpor;

	Real xi,yi,zi;
	Real mi=P1_sph[i].m;
	Real rhoi=P1_sph[i].rho;
	Real tmp_h,tmp_A,search_range;
	Real tmp_SPHflt, tmp_DEMflt,tmp_DEMfltd,tmp_por;				//tmp_filter, tmp_porosity(DEM calculation)
	Real tmp_DEMfltd_2;
	Real tmp_dfltx, tmp_dflty, tmp_dfltz;



	tmp_dfltx=0.0; tmp_dflty=0.0; tmp_dfltz=0.0;

	xi=P1_sph[i].x;
	yi=P1_sph[i].y;
	zi=P1_sph[i].z;


	tmp_h=1*P1_sph[i].h;
	tmp_A=calc_tmpA(1.0*tmp_h);
	
	search_range=k_search_kappa*1.0*tmp_h;	// search range


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_por=0.0;
	tmp_SPHflt=0.0;
	tmp_DEMflt=0.0;
	tmp_DEMfltd=0.0;
	tmp_DEMfltd_2=0.0;
	
	// sph-sph 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str_sph[k]!=cu_memset){
					int_t fend=g_end_sph[k];
					for(int_t j=g_str_sph[k];j<fend;j++){

						Real tmp_wij,tdist;
						Real xj,yj,zj;


						xj=P1_sph[j].x;
						yj=P1_sph[j].y;
						zj=P1_sph[j].z;

						
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
						if(tdist<search_range){

							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);


							Real mj,rhoj,radj;
							int_t ptypej;

							mj=P1_sph[j].m;
							rhoj=P1_sph[j].rho;
							radj=P1_sph[j].rad;
							ptypej=P1_sph[j].p_type;

							if (ptypei<=1000){
								if((ptypej<=1000)){
								//if((ptypej<=1000)){
									tmp_SPHflt+=mj/rhoj*tmp_wij*(ptypej==1);
									//tmp_SPHflt+=mj/rhoj*1*(ptypej==1);	//simmple summation
								}
							}
						}	
					}
				}
			}
		}
	}


	// sph-dem 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str_dem[k]!=cu_memset){
					int_t fend=g_end_dem[k];
					for(int_t j=g_str_dem[k];j<fend;j++){

						Real tmp_wij,tdist;
						Real tmp_dwij;
						Real xj,yj,zj;
						Real tdwx,tdwy,tdwz;


						xj=P1_dem[j].x;
						yj=P1_dem[j].y;
						zj=P1_dem[j].z;

						
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
						if(tdist<search_range){

							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);
							tmp_dwij=calc_kernel_dwij(tmp_A,1.0*tmp_h,tdist);

							tdwx=tmp_dwij*(xi-xj)/tdist;
							tdwy=tmp_dwij*(yi-yj)/tdist;
							tdwz=tmp_dwij*(zi-zj)/tdist;

							Real mj,rhoj,radj;
							Real uxj, uyj, uzj;
							int_t ptypej;

							mj=P1_dem[j].m;
							rhoj=P1_dem[j].rho;
							radj=P1_dem[j].rad;
							ptypej=P1_dem[j].p_type;
							uxj=P1_dem[j].ux;
							uyj=P1_dem[j].uy;
							uzj=P1_dem[j].uz;

							if (ptypei<=1000){
								if(ptypej>1000){

									tmp_por+=tmp_wij*4.18879*radj*radj*radj;
									tmp_dfltx+=tdwx*uxj*4.18879*radj*radj*radj;
									tmp_dflty+=tdwy*uyj*4.18879*radj*radj*radj;
									tmp_dfltz+=tdwz*uzj*4.18879*radj*radj*radj;
									//tmp_por+=1*4.18879*radj*radj*radj;	//simple summation

								}	  
							}
						}	
					}
				}
			}
		}
	}
	if(ptypei<=1000) {
		//P1[i].DEMpor=1.0-(zi<0.186)*(zi>0.00)*(tmp_por/(tmp_SPHflt+1e-20));
		P1_sph[i].DEMpor=1.0-(tmp_por/(tmp_SPHflt+1e-20));
		P1_sph[i].dDEMpor=(tmp_dfltx+tmp_dflty+tmp_dfltz)/(tmp_SPHflt+1e-20);
	}
}


__global__ void KERNEL_clc_prep3D_coupling_dem(int_t*g_str_dem,int_t*g_end_dem,part1*P1_dem, part2*P2_dem,
												int_t*g_str_sph,int_t*g_end_sph,part1*P1_sph, part2*P2_sph,
												int_t tcount, Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1_dem[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	int_t ptypei=P1_dem[i].p_type;

	Real xi,yi,zi;
	Real mi=P1_dem[i].m;
	Real rhoi=P1_dem[i].rho;
	Real tmp_h,tmp_A,search_range;
	Real tmp_SPHflt, tmp_DEMflt,tmp_DEMfltd,tmp_por;				//tmp_filter, tmp_porosity(DEM calculation)
	Real tmp_DEMfltd_2;


	xi=P1_dem[i].x;
	yi=P1_dem[i].y;
	zi=P1_dem[i].z;


	tmp_h=1*P1_dem[i].h;
	tmp_A=calc_tmpA(1.0*tmp_h);
	
	search_range=k_search_kappa*1.0*tmp_h;	// search range


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	// 초기화
	tmp_por=0.0;
	tmp_SPHflt=0.0;
	tmp_DEMflt=0.0;
	tmp_DEMfltd=0.0;
	tmp_DEMfltd_2=0.0;
	
	// dem-sph 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str_sph[k]!=cu_memset){
					int_t fend=g_end_sph[k];
					for(int_t j=g_str_sph[k];j<fend;j++){

						Real tmp_wij,tdist;
						Real xj,yj,zj;


						xj=P1_sph[j].x;
						yj=P1_sph[j].y;
						zj=P1_sph[j].z;

						
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
						if(tdist<search_range){

							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);


							Real mj,rhoj,radj;
							int_t ptypej;

							mj=P1_sph[j].m;
							rhoj=P1_sph[j].rho;
							radj=P1_sph[j].rad;
							ptypej=P1_sph[j].p_type;

							if(ptypei>1000){
								//if(ptypej<=1000){
								if((ptypej<=1000) && (ptypej!=0) && (ptypej!=MOVING) && (ptypej==1)){
									tmp_DEMfltd+=mj/rhoj*tmp_wij;
									tmp_DEMfltd_2+=mj/rhoj*tmp_wij;
								}	
								if((ptypej<=1000) && (ptypej!=0) && (ptypej!=MOVING) && (ptypej==3)){
									tmp_DEMflt+=mj/rhoj*tmp_wij;
								}	
							}
						}	
					}
				}
			}
		}
	}

	// dem-dem 계산
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			//	if(k<0||k>=k_num_cells-1) continue;
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str_dem[k]!=cu_memset){
					int_t fend=g_end_dem[k];
					for(int_t j=g_str_dem[k];j<fend;j++){

						Real tmp_wij,tdist;
						Real xj,yj,zj;


						xj=P1_dem[j].x;
						yj=P1_dem[j].y;
						zj=P1_dem[j].z;

						
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
						if(tdist<search_range){

							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);


							Real mj,rhoj,radj;
							int_t ptypej;

							mj=P1_dem[j].m;
							rhoj=P1_dem[j].rho;
							radj=P1_dem[j].rad;
							ptypej=P1_dem[j].p_type;

							if(ptypei>1000){
								if(ptypej>1000) tmp_por+=tmp_wij*4.18879*radj*radj*radj;
							}
						}	
					}
				}
			}
		}
	}


	// filter
	if(ptypei>1000){
		P1_dem[i].flt_s=tmp_DEMflt;
		P1_dem[i].flt_sd=tmp_DEMfltd;
		P1_dem[i].flt_sd_2=tmp_DEMfltd_2;
		P1_dem[i].DEMpor=1.0-(tmp_por/(tmp_DEMflt+tmp_DEMfltd+1.0e-20));
	}
}
// __global__ void KERNEL_clc_prep3D_coupling_sph(int_t*g_str_sph,int_t*g_end_sph,part1*P1_sph, part2*P2_sph, 
// 												int_t*g_str_dem,int_t*g_end_dem,part1*P1_dem, part2*P2_dem, 
// 												int_t tcount, Real ttime)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2_sph) return;
// 	if(P1_sph[i].i_type>i_type_crt) return;

// 	int_t icell,jcell,kcell;
// 	int_t ptypei=P1_sph[i].p_type;

// 	Real xi,yi,zi;
// 	Real mi=P1_sph[i].m;
// 	Real rhoi=P1_sph[i].rho;
// 	Real tmp_h,tmp_A,search_range;
// 	Real tmp_SPHflt, tmp_DEMflt,tmp_DEMfltd,tmp_por;				//tmp_filter, tmp_porosity(DEM calculation)
// 	Real tmp_DEMfltd_2;

// 	xi=P1_sph[i].x;
// 	yi=P1_sph[i].y;
// 	zi=P1_sph[i].z;

// 	tmp_h=1*P1_sph[i].h;
// 	tmp_A=calc_tmpA(1.0*tmp_h);
	
// 	search_range=k_search_kappa*1.0*tmp_h;	// search range

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

// 	// 초기화
// 	tmp_por=0.0;
// 	tmp_SPHflt=0.0;
// 	tmp_DEMflt=0.0;
// 	tmp_DEMfltd=0.0;
// 	tmp_DEMfltd_2=0.0;
	
// 	// sph-sph 계산
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

// 			//	if(k<0||k>=k_num_cells-1) continue;
// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
// 				if(g_str_sph[k]!=cu_memset){
// 					int_t fend=g_end_sph[k];
// 					for(int_t j=g_str_sph[k];j<fend;j++){

// 						Real tmp_wij,tdist;
// 						Real xj,yj,zj;

// 						xj=P1_sph[j].x;
// 						yj=P1_sph[j].y;
// 						zj=P1_sph[j].z;

// 						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
// 						if(tdist<search_range){

// 							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);

// 							Real mj,rhoj,radj;
// 							int_t ptypej;

// 							mj=P1_sph[j].m;
// 							rhoj=P1_sph[j].rho;
// 							radj=P1_sph[j].rad;
// 							ptypej=P1_sph[j].p_type;

// 							if (ptypei<=1000){
// 								if((ptypej<=1000)){
// 								//if((ptypej<=1000)){
// 									tmp_SPHflt+=mj/rhoj*tmp_wij*(ptypej==1);
// 									//tmp_SPHflt+=mj/rhoj*1*(ptypej==1);	//simmple summation
// 								}
// 							}
// 						}	
// 					}
// 				}
// 			}
// 		}
// 	}

// 	// sph-dem 계산
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
// 			//	if(k<0||k>=k_num_cells-1) continue;
// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
// 				if(g_str_dem[k]!=cu_memset){
// 					int_t fend=g_end_dem[k];
// 					for(int_t j=g_str_dem[k];j<fend;j++){

// 						Real tmp_wij,tdist;
// 						Real xj,yj,zj;

// 						xj=P1_dem[j].x;
// 						yj=P1_dem[j].y;
// 						zj=P1_dem[j].z;

// 						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
// 						if(tdist<search_range){

// 							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);

// 							Real mj,rhoj,radj;
// 							int_t ptypej;

// 							mj=P1_dem[j].m;
// 							rhoj=P1_dem[j].rho;
// 							radj=P1_dem[j].rad;
// 							ptypej=P1_dem[j].p_type;

// 							if (ptypei<=1000){
// 								if(ptypej>1000){
// 									tmp_por+=tmp_wij*4.18879*radj*radj*radj;	// 4.1887 = 4pi/3
// 									//tmp_por+=1*4.18879*radj*radj*radj;		//simple summation
// 								}	  
// 							}
// 						}	
// 					}
// 				}
// 			}
// 		}
// 	}
// 	if(ptypei<=1000) {
// 		//P1[i].DEMpor=1.0-(zi<0.186)*(zi>0.00)*(tmp_por/(tmp_SPHflt+1e-20));
// 		P1_sph[i].DEMpor=1.0-(tmp_por/(tmp_SPHflt+1e-20));
// 	}
// }

// __global__ void KERNEL_clc_prep3D_coupling_dem(int_t*g_str_dem,int_t*g_end_dem,part1*P1_dem, part2*P2_dem,
// 												int_t*g_str_sph,int_t*g_end_sph,part1*P1_sph, part2*P2_sph,
// 												int_t tcount, Real ttime)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2_dem) return;
// 	if(P1_dem[i].i_type>i_type_crt) return;

// 	int_t icell,jcell,kcell;
// 	int_t ptypei=P1_dem[i].p_type;

// 	Real xi,yi,zi;
// 	Real mi=P1_dem[i].m;
// 	Real rhoi=P1_dem[i].rho;
// 	Real tmp_h,tmp_A,search_range;
// 	Real tmp_SPHflt, tmp_DEMflt,tmp_DEMfltd,tmp_por;				//tmp_filter, tmp_porosity(DEM calculation)
// 	Real tmp_DEMfltd_2;

// 	xi=P1_dem[i].x;
// 	yi=P1_dem[i].y;
// 	zi=P1_dem[i].z;

// 	tmp_h=1*P1_dem[i].h;
// 	tmp_A=calc_tmpA(1.0*tmp_h);
	
// 	search_range=k_search_kappa*1.0*tmp_h;	// search range

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

// 	// 초기화
// 	tmp_por=0.0;
// 	tmp_SPHflt=0.0;
// 	tmp_DEMflt=0.0;
// 	tmp_DEMfltd=0.0;
// 	tmp_DEMfltd_2=0.0;
	
// 	//dem-sph 계산
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

// 			//	if(k<0||k>=k_num_cells-1) continue;
// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
// 				if(g_str_sph[k]!=cu_memset){
// 					int_t fend=g_end_sph[k];
// 					for(int_t j=g_str_sph[k];j<fend;j++){

// 						Real tmp_wij,tdist;
// 						Real xj,yj,zj;

// 						xj=P1_sph[j].x;
// 						yj=P1_sph[j].y;
// 						zj=P1_sph[j].z;

// 						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
// 						if(tdist<search_range){

// 							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);

// 							Real mj,rhoj,radj;
// 							int_t ptypej;

// 							mj=P1_sph[j].m;
// 							rhoj=P1_sph[j].rho;
// 							radj=P1_sph[j].rad;
// 							ptypej=P1_sph[j].p_type;

// 							if(ptypei>1000){
// 								//if(ptypej<=1000){
// 								if((ptypej<=1000) && (ptypej!=0) && (ptypej!=MOVING) && (ptypej==1)){
// 									tmp_DEMfltd+=mj/rhoj*tmp_wij;
// 									tmp_DEMfltd_2+=mj/rhoj*tmp_wij;
// 								}	
// 								if((ptypej<=1000) && (ptypej!=0) && (ptypej!=MOVING) && (ptypej==3)){
// 									tmp_DEMflt+=mj/rhoj*tmp_wij;
// 								}	
// 							}
// 						}	
// 					}
// 				}
// 			}
// 		}
// 	}
// 	// dem-dem 계산
// 	//dem-sph 계산
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

// 			//	if(k<0||k>=k_num_cells-1) continue;
// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
// 				if(g_str_dem[k]!=cu_memset){
// 					int_t fend=g_end_dem[k];
// 					for(int_t j=g_str_dem[k];j<fend;j++){

// 						Real tmp_wij,tdist;
// 						Real xj,yj,zj;

// 						xj=P1_dem[j].x;
// 						yj=P1_dem[j].y;
// 						zj=P1_dem[j].z;

// 						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
	
// 						if(tdist<search_range){

// 							tmp_wij=calc_kernel_wij(tmp_A,1.0*tmp_h,tdist);

// 							Real mj,rhoj,radj;
// 							int_t ptypej;

// 							mj=P1_dem[j].m;
// 							rhoj=P1_dem[j].rho;
// 							radj=P1_dem[j].rad;
// 							ptypej=P1_dem[j].p_type;

// 							if(ptypei>1000){
// 								if(ptypej>1000) tmp_por+=tmp_wij*4.18879*radj*radj*radj;
// 							}
// 						}	
// 					}
// 				}
// 			}
// 		}
// 	}
// 	// filter
// 	if(ptypei>1000){
// 		P1_dem[i].flt_s=tmp_DEMflt;
// 		P1_dem[i].flt_sd=tmp_DEMfltd;
// 		P1_dem[i].flt_sd_2=tmp_DEMfltd_2;
// 		P1_dem[i].DEMpor=1.0-(tmp_por/(tmp_DEMflt+tmp_DEMfltd+1.0e-20));
// 	}
// }
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_surface_normal2D(int_t*g_str,int_t*g_end,Real tnd_ref,part1*P1,part1*TP1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell,non;
	Real xi,yi;
	Real tn_mag,w_dxi;
	Real tmpx,tmpy,tmpk;
	Real tmpx0,tmpy0;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	w_dxi=P1[i].w_dx;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=tmpk=0.0;
	non=0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0||k>=k_num_cells-1) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					xj=P1[j].x;
					yj=P1[j].y;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
					if(tdist>0&&tdist<search_range){

						Real tdwx,tdwy,tdwij,hj,mj,rhoj,ww;
						Real twij,rdist;

						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						rdist=1.0/(tdist+1e-10);

						tmpx+=(xi-xj)*rdist;
						tmpy+=(yi-yj)*rdist;
						tmpk+=twij;
						non++;

						//KERNEL_clc_particle_shifting_lind3D----------------------
						if(k_kgc_solve==1){
							// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}
						mj=P1[j].m;
						hj=P1[j].h;
						rhoj=P1[j].rho;

						ww=twij/w_dxi;
						tmpx0+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwx;
						tmpy0+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwy;
					}
				}
			}
		}
	}
	tmpx=tmpx/non;
	tmpy=tmpy/non;
	tn_mag=sqrt(tmpx*tmpx+tmpy*tmpy);

	Real lbl_surfi=(tn_mag>0.3)|(non<10);
	P3[i].lbl_surf=lbl_surfi;

	//KERNEL_clc_particle_shifting_lind3D----------------------
	Real dr_square=tmpx*tmpx+tmpy*tmpy;
	if((lbl_surfi<0.5)&(dr_square<0.01*tmp_h*tmp_h)){		// interior
		TP1[i].x=xi+tmpx0*(ptypei>0);
		TP1[i].y=yi+tmpy0*(ptypei>0);
	}
	//---------------------------------------------------------
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_surface_normal3D(int_t*g_str,int_t*g_end,Real tnd_ref,part1*P1,part1*TP1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell,kcell,non;
	Real xi,yi,zi;
	Real tn_mag,w_dxi;
	Real tmpx,tmpy,tmpz,tmpk;
	Real tmpx0,tmpy0,tmpz0;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	w_dxi=P1[i].w_dx;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=tmpk=0.0;
	tmpx0=tmpy0=tmpz0=0.0;
	non=0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0||k>=k_num_cells-1) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
						if(tdist>0&&tdist<search_range){
							Real twij,rdist;
							Real tdwx,tdwy,tdwz,tdwij,mj,hj,rhoj,ww;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							rdist=1.0/(tdist+1e-10);

							tmpx+=(xi-xj)*rdist;
							tmpy+=(yi-yj)*rdist;
							tmpz+=(zi-zj)*rdist;
							tmpk+=twij;
							non++;

							//KERNEL_clc_particle_shifting_lind3D----------------------
							if(k_kgc_solve==1){
								// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								// tdwz=((P3[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}
							mj=P1[j].m;
							rhoj=P1[j].rho;
							hj=P1[j].h;
							ww=twij/w_dxi;
							tmpx0+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwx;
							tmpy0+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwy;
							tmpz0+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwz;
							//---------------------------------------------------------
						}
					}
				}
			}
		}
	}
	tmpx=tmpx/non;
	tmpy=tmpy/non;
	tmpz=tmpz/non;
	tn_mag=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz+tmpz);

	Real lbl_surfi=(tn_mag>0.3)|(non<10);
	P3[i].lbl_surf=lbl_surfi;

	//KERNEL_clc_particle_shifting_lind3D----------------------
	Real dr_square=tmpx*tmpx+tmpy*tmpy+tmpz*tmpz;
	if ((lbl_surfi<0.5)&(dr_square<0.01*tmp_h*tmp_h)){		// interior
		TP1[i].x=xi+tmpx0*(ptypei>0);
		TP1[i].y=yi+tmpy0*(ptypei>0);
		TP1[i].z=zi+tmpz0*(ptypei>0);
	}
	//---------------------------------------------------------
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_surface_detect2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell,non;
	Real xi,yi,ptypei;
	Real rn;
	Real tmpx,tmpy;
	Real search_range;

	search_range=k_search_kappa*P1[i].h;

	xi=P1[i].x;
	yi=P1[i].y;
	ptypei=P1[i].p_type;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	non=0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0||k>=k_num_cells-1) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,rr,tdist,ptypej;
					xj=P1[j].x;
					yj=P1[j].y;
						ptypej=P1[j].p_type;

					rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					tdist=sqrt(rr);
					if(tdist<search_range){
						Real rd=1.0/(tdist+1e-10);
						tmpx+=(xi-xj)*(ptypei==ptypej)*rd;
						tmpy+=(yi-yj)*(ptypei==ptypej)*rd;
						non++;
					}
				}
			}
		}
	}
	rn=1.0/non;
	tmpx=tmpx*rn;
	tmpy=tmpy*rn;
	P3[i].lbl_surf=((sqrt(tmpx*tmpx+tmpy*tmpy))>0.25)|(non<10);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_surface_detect3D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t non;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rn;
	Real tmpx,tmpy,tmpz;
	Real search_range;
	Real ptypei;

	search_range=k_search_kappa*P1[i].h;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	ptypei = P1[i].p_type;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=0.0;
	non=0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0||k>=k_num_cells-1) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,rr,tdist,ptypej;
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;
						ptypej=P1[j].p_type;

						rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);
						tdist=sqrt(rr);
						if(tdist<search_range){
							Real rd=1.0/(tdist+1e-10);
							tmpx+=(xi-xj)*(ptypei==ptypej)*rd;
							tmpy+=(yi-yj)*(ptypei==ptypej)*rd;
							tmpz+=(zi-zj)*(ptypei==ptypej)*rd;
							non+=(ptypei==ptypej);
						}
					}
				}
			}
		}
	}
	rn=1.0/(non+1E-10);
	tmpx=tmpx*rn;
	tmpy=tmpy*rn;
	tmpz=tmpz*rn;
	P3[i].lbl_surf=((sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz)>0.3)|(non<10));
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_particle_shifting_lind2D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real w_dx_i,dr_square,hi;
	Real tmpx,tmpy;

	Real search_range,tmp_A;

	hi=P1[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	ptypei=P1[i].p_type;

	xi=P2[i].x0;
	yi=P2[i].y0;
	w_dx_i=P1[i].w_dx;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0||k>=k_num_cells-1) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					xj=P1[j].x;
					yj=P1[j].y;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
					if(tdist>0&&tdist<search_range){
						Real tdwx,tdwy,tdwij,twij,hj,mj,rhoj,ww;
						tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
						twij=calc_kernel_wij(tmp_A,hi,tdist);

						if(k_kgc_solve==1){
							// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}

						mj=P1[j].m;
						hj=P1[j].h;
						rhoj=P1[j].rho;

						ww=twij/w_dx_i;
						tmpx+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwx;
						tmpy+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwy;
					}
				}
			}
		}
	}

	dr_square=tmpx*tmpx+tmpy*tmpy;

	if((P3[i].lbl_surf<0.5)&(dr_square<0.01*hi*hi)){		// interior
		P2[i].x0=xi+tmpx*(ptypei>0);
		P2[i].y0=yi+tmpy*(ptypei>0);
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_particle_shifting_lind3D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real w_dx_i,hi,dr_square;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpz;

	hi=P1[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	ptypei=P1[i].p_type;

	xi=P2[i].x0;
	yi=P2[i].y0;
	zi=P2[i].z0;
	w_dx_i=P1[i].w_dx;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0||k>=k_num_cells-1) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
						if(tdist>0&&tdist<search_range){
							Real tdwx,tdwy,tdwz,tdwij,twij,mj,hj,rhoj,ww;
							tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
							twij=calc_kernel_wij(tmp_A,hi,tdist);
							if(k_kgc_solve==1){
								// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								// tdwz=((P3[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}

							mj=P1[j].m;
							rhoj=P1[j].rho;
							hj=P1[j].h;

							ww=twij/w_dx_i;
							tmpx+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwx;
							tmpy+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwy;
							tmpz+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwz;
						}
					}
				}
			}
		}
	}

	dr_square=tmpx*tmpx+tmpy*tmpy+tmpz*tmpz;

	if ((P3[i].lbl_surf<0.5)&(dr_square<0.01*hi*hi)){		// interior
		P2[i].x0=xi+tmpx*(ptypei>0);
		P2[i].y0=yi+tmpy*(ptypei>0);
		P2[i].z0=zi+tmpz*(ptypei>0);
	}
}
////////////////////////////////////////////////////////////////////////
// Gaussian Kernel function
__global__ void KERNEL_clc_gaussian_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,tmp_A;

	//dx/h=(2/3*h)/h
	tmp_R=2/3;

	//if(k_dim==1) tmp_A=1.0/(pow(PI,0.5)*P1[i].h);
	if(k_dim==2) tmp_A=1.0/(PI*pow(P1[i].h,2));
	if(k_dim==3) tmp_A=1.0/(pow(PI,1.5)*pow(P1[i].h,3));
	P1[i].w_dx=tmp_A*exp(-pow(tmp_R,2));
}
////////////////////////////////////////////////////////////////////////
// Quintic kernel
__global__ void KERNEL_clc_quintic_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,tmp_A;

	//dx/h=(2/3*h)/h
	tmp_R=2/3;

	//if(k_dim==1) tmp_A=1.0;
	if(k_dim==2) tmp_A=7.0/(478.0*PI*pow(P1[i].h,2));
	if(k_dim==3) tmp_A=3.0/(359.0*PI*pow(P1[i].h,3));

	Real tmpwdx;
	tmpwdx=pow(3.0-tmp_R,5);
	tmpwdx+=-6.0*pow(2.0-tmp_R,5);
	tmpwdx+=15.0*pow(1.0-tmp_R,5);
	tmpwdx*=tmp_A;
	P1[i].w_dx=tmpwdx;
	//P1[i].w_dx=tmp_A*(pow(3.0-tmp_R,5)-6.0*pow(2.0-tmp_R,5)+15.0*pow(1.0-tmp_R,5));
}
////////////////////////////////////////////////////////////////////////
// Quartic kernel
__global__ void KERNEL_clc_quartic_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,tmp_A;

	//dx/h=(2/3*h)/h
	tmp_R=2/3;

	//if(k_dim==1) tmp_A=1.0/P1[i].h;
	if(k_dim==2) tmp_A=15.0/(7.0*PI*pow(P1[i].h,2));
	if(k_dim==3) tmp_A=315.0/(208.0*PI*pow(P1[i].h,3));

	Real tmpwdx=0.0;
	if(tmp_R<2){
		tmpwdx=2.0/3.0-9.0/8.0*pow(tmp_R,2);
		tmpwdx+=19.0/24.0*pow(tmp_R,3);
		tmpwdx+=-5.0/32.0*pow(tmp_R,4);
		tmpwdx*=tmp_A;
	}
	P1[i].w_dx=tmpwdx;
	//P1[i].w_dx=(tmp_R<2)*tmp_A*(2.0/3.0-9.0/8.0*pow(tmp_R,2)+19.0/24.0*pow(tmp_R,3)-5.0/32.0*pow(tmp_R,4));
}
////////////////////////////////////////////////////////////////////////
// Wendland2 kernel
__global__ void KERNEL_clc_wendland2_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,tmp_h,tmp_C,tmpwdx;

	//dx/2h=(2/3*h)/h*0.5
	tmp_R=1/3;
	tmp_h=P1[i].h;

	// equation of Wendland 2 kernel function

	// if(k_dim==1){
	// 	tmp_C=1.25/(2*tmp_h);// 5./(4*(2h))
	// 	P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+3*tmp_R);
	// }

	if(k_dim==2){
		tmp_C=2.228169203286535/(4*tmp_h*tmp_h);				// 7.0/(pi*(2h)^2)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);
	}
	if(k_dim==3){
		tmp_C=3.342253804929802/(8*tmp_h*tmp_h*tmp_h);	// 21.0/(2*pi*(2h)^3)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);
	}
}
////////////////////////////////////////////////////////////////////////
// Wendland4 kernel
__global__ void KERNEL_clc_wendland4_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,tmp_h,tmp_C,tmpwdx;

	//dx/2h=(2/3*h)/h*0.5
	tmp_R=1/3;
	tmp_h=P1[i].h;

	// equation of Wendland 4 kernel function

	// if(k_dim==1){
	// 	tmp_C=1.5/(2*tmp_h);// 3./(2*(2h))
	// 	P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+5*tmp_R+8*tmp_R*tmp_R);
	// }

	if(k_dim==2){
		tmp_C=2.864788975654116/(4*tmp_h*tmp_h);					// 9./(pi*(2tmp_h)^2)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);
	}
	if(k_dim==3){
		tmp_C=4.923856051905513/(8*tmp_h*tmp_h*tmp_h);		// 495./(32*pi*(2tmp_h)^3)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C* (1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);
	}
}
////////////////////////////////////////////////////////////////////////
// Wendland6 kernel
__global__ void KERNEL_clc_wendland6_w_dx(part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real tmp_R,tmp_h,tmp_C,tmpwdx;

	//dx/2h=(2/3*h)/h*0.5
	tmp_R=1/3;
	tmp_h=P1[i].h;

	// equation of Wendland 6 kernel function

	// if(k_dim==1){
	// 	tmp_C=1.71875/(2*tmp_h);// 55./(32*(2tmp_h))
	// 	P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+7*tmp_R+19*tmp_R*tmp_R+21*tmp_R*tmp_R*tmp_R);
	// }

	if(k_dim==2){
		tmp_C=3.546881588905096/(4*tmp_h*tmp_h);// 78./(7*pi*(2tmp_h)^2)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);
	}
	if(k_dim==3){
		tmp_C=6.788953041263660/(8*tmp_h*tmp_h*tmp_h);// 1365./(64*pi*(2tmp_h)^3)
		tmpwdx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R);
		tmpwdx*=(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);
		P1[i].w_dx=tmpwdx;
		//P1[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);
	}
}
////////////////////////////////////////////////////////////////////////
void calculate_w_dx(part1*P1)
{
	dim3 b,t;
	t.x=128;
	b.x=(num_part2-1)/t.x+1;

	// Calculate kernel value for initial spacing
	if(kernel_type==Gaussian)	 KERNEL_clc_gaussian_w_dx<<<b,t>>>(P1);
	if(kernel_type==Quintic)	 KERNEL_clc_quintic_w_dx<<<b,t>>>(P1);
	if(kernel_type==Quartic)	 KERNEL_clc_quartic_w_dx<<<b,t>>>(P1);
	if(kernel_type==Wendland2) KERNEL_clc_wendland2_w_dx<<<b,t>>>(P1);
	if(kernel_type==Wendland4) KERNEL_clc_wendland4_w_dx<<<b,t>>>(P1);
	if(kernel_type==Wendland6) KERNEL_clc_wendland6_w_dx<<<b,t>>>(P1);
	cudaDeviceSynchronize();
}

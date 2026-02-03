////////////////////////////////////////////////////////////////////////

__global__ void KERNEL_clc_mass_init(int_t*g_str,int_t*g_end,part1*P1,part2*P2,int_t tcount)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].p_type>1000) return;
	if(P1[i].i_type>i_type_crt) return;

		//P2[i].rho_ref0 = P1[i].rho;
		//P2[i].rho_ref = P1[i].rho;
		P1[i].vol0 = pow(P1[i].h/1.6,k_dim);
		P1[i].vol = P1[i].vol0;

		P2[i].rho0=P1[i].rho;

		Real tempi = P1[i].temp;
		Real ptypei = P1[i].p_type;
		P1[i].cond = conductivity(tempi,ptypei);


}

__global__ void KERNEL_clc_mass_init_sph(int_t*g_str,int_t*g_end,part1*P1,part2*P2,int_t tcount)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].p_type>1000) return;
	if(P1[i].i_type>i_type_crt) return;

		//P2[i].rho_ref0 = P1[i].rho;
		//P2[i].rho_ref = P1[i].rho;
		P1[i].vol0 = pow(P1[i].h/1.6,k_dim);
		P1[i].vol = P1[i].vol0;

		P2[i].rho0=P1[i].rho;

		Real tempi = P1[i].temp;
		Real ptypei = P1[i].p_type;
		P1[i].cond = conductivity(tempi,ptypei);


}

__global__ void KERNEL_clc_mass_update(int_t*g_str,int_t*g_end,part1*P1,part2*P2,int_t tcount)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].p_type>1000) return;
	if(P1[i].i_type>i_type_crt) return;
	if((P1[i].elix==1.0)&(P1[i].eliy==1.0)) return;
	// if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method

	Real rhoi=P1[i].rho;

	P1[i].m=rhoi*P1[i].vol;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_volume_update(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,int_t tcount)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if((P1[i].elix<1e-8)&(P1[i].eliy<1e-8)&(tcount>0)) return;
	// if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	

	// Real trho=P1[i].rho;
	// P1[i].vol0 = (P1[i].h/1.6)*(P1[i].h/1.6)*(P1[i].h/1.6);
	// P2[i].rho0=trho;
	

	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real rhoi,mi;
	// Real rho_ref_i;
	Real search_range,tmp_h,tmp_A,tmp_R;
	int p_type_i;
	Real tmpx,tmpy,tmprho, filt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	p_type_i=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	rhoi=P1[i].rho;
	mi=P1[i].m;
	// rho_ref_i=P2[i].rho_ref;

	// calculate I,J,K in cell
if((k_x_max==k_x_min)){icell=0;}
else{icell=min(floor((xi-k_x_min)/k_dcell),k_NI-1);}
if((k_y_max==k_y_min)){jcell=0;}
else{jcell=min(floor((yi-k_y_min)/k_dcell),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmp_R=filt=0.0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					Real uxj, uyj, rhoj,mj;
					Real volj;
					int p_type_j;
					int itype;

					xj=P1[j].x;
					yj=P1[j].y;
					itype=P1[j].i_type;
					mj=P1[j].m;

					if(P1[j].p_type<1000){
						if(itype!=4){
					xj=P1[j].x;
					yj=P1[j].y;
					rhoj=P1[j].rho;
					volj=P1[j].vol;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;

					if(tdist<search_range){
						Real twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						
						tmp_R+=twij;

						filt+=mj/rhoj/P1[j].vol0*twij;
					}
				}
			}
				}
			}
		}
	}
	P1[i].vol=P1[i].vol0*1.0/tmp_R*filt;
	// P1[i].vol=1.0/tmp_R*filt;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_volume_update3D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3,int_t tcount)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if((P1[i].elix<1e-8)&(P1[i].eliy<1e-8)&(tcount>0)) return;
	// if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method
	
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rhoi,mi;
	// Real rho_ref_i;
	Real search_range,tmp_h,tmp_A,tmp_R;
	int p_type_i;
	Real tmpx,tmpy,tmprho, filt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	p_type_i=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	rhoi=P1[i].rho;
	mi=P1[i].m;
	// rho_ref_i=P2[i].rho_ref;

	// calculate I,J,K in cell
// calculate I,J,K in cell
if((k_x_max==k_x_min)){icell=0;}
else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
if((k_y_max==k_y_min)){jcell=0;}
else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
if((k_z_max==k_z_min)){kcell=0;}
else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// out-of-range handling
if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

tmp_R=filt=0.0;
for(int_t z=-3;z<=3;z++){
	for(int_t y=-3;y<=3;y++){
		for(int_t x=-3;x<=3;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
			int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;


	
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,zj,tdist;
					Real uxj, uyj, rhoj,mj;
					Real volj;
					int p_type_j;
					int itype;

					xj=P1[j].x;
					yj=P1[j].y;
					itype=P1[j].i_type;
					mj=P1[j].m;

					if(P1[j].p_type<1000){
						if(itype!=4){
					xj=P1[j].x;
					yj=P1[j].y;
					zj=P1[j].z;
					rhoj=P1[j].rho;
					volj=P1[j].vol;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;

					if(tdist<search_range){
						Real twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						
						tmp_R+=twij;

						filt+=mj/rhoj/P1[j].vol0*twij;
					}
				}
			}
				}
			}

			}
		}
	}
	P1[i].vol=P1[i].vol0*1.0/tmp_R*filt;
	// P1[i].vol=1.0/tmp_R*filt;
}




////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_mass_sum2D(int_t*g_str,int_t*g_end,part1*P1,part1*TP1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell;
	Real xi,yi;
	Real search_range,tmp_h,tmp_A,tmp_R;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=TP1[i].x;
	yi=TP1[i].y;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmp_R=0.0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					xj=TP1[j].x;
					yj=TP1[j].y;

					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
					if(tdist<search_range){
						Real twij,mj;
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
						mj=P1[j].m;
						tmp_R+=mj*twij;
					}
				}
			}
		}
	}
	// TP1[i].rho=tmp_R/P1[i].flt_s;
	P1[i].rho=tmp_R/P1[i].flt_s;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_mass_sum3D(int_t*g_str,int_t*g_end,part1*P1,part1*TP1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;

	Real search_range,tmp_h,tmp_A,tmp_R;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=TP1[i].x;
	yi=TP1[i].y;
	zi=TP1[i].z;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmp_R=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=TP1[j].x;
						yj=TP1[j].y;
						zj=TP1[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
						if(tdist<search_range){
							Real twij,mj;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							tmp_R+=mj*twij;
						}
					}
				}
			}
		}
	}
	P1[i].rho=tmp_R/P1[i].flt_s;
	// TP1[i].rho=tmp_R/P1[i].flt_s;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_mass_sum_norm2D(int_t*g_str,int_t*g_end,part1*P1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	Real xi,yi;
	Real rho_ref_i;
	Real DEMpor_i;
	Real search_range,tmp_h,tmp_A,tmp_R;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	rho_ref_i=P2[i].rho_ref;
	DEMpor_i=P1[i].DEMpor;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmp_R=0.0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
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
						if(tdist<search_range){
							Real twij,mj,rho_ref_j;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rho_ref_j=P2[j].rho_ref;
							tmp_R+=(mj/rho_ref_j)*twij;
						}

					}


					
				}
			}
		}
	}
	P1[i].rho=rho_ref_i*tmp_R/P1[i].flt_s;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_mass_sum_norm3D(int_t*g_str,int_t*g_end,part1*P1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rho_ref_i, flti;
	Real DEMpor_i;
	Real search_range,tmp_h,tmp_A,tmp_R,tmp_flt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	rho_ref_i=P2[i].rho_ref;
	flti=P1[i].flt_s;
	DEMpor_i=P1[i].DEMpor;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmp_R=0.0;
	tmp_flt=0.0;
	for(int_t z=-3;z<=3;z++){
		for(int_t y=-3;y<=3;y++){
			for(int_t x=-3;x<=3;x++){
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
							if(tdist<search_range){
								Real twij,mj,rho_ref_j,rhoj;
								twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								mj=P1[j].m;
								rhoj=P1[j].rho;
								rho_ref_j=P2[j].rho_ref;
								tmp_R+=(mj/rho_ref_j)*twij;
								tmp_flt+=(mj/rhoj)*twij;
							}

						}
	
					}
				}
			}
		}
	}
	// if (flti==0){
	// 	P1[i].rho=rho_ref_i;
	// }
	// else{
	// 	P1[i].rho=rho_ref_i*tmp_R/P1[i].flt_s;
	// }
	// if((k_count%k_freq_filt)==0) P1[i].flt_s=tmp_flt;
	P1[i].rho=rho_ref_i*tmp_R/P1[i].flt_s;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_reference_density(part1*P1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;
	if(P1[i].p_type>1000) return;

	Real m,h;

	m=P1[i].m;
	h=P1[i].h;
	P2[i].rho_ref=m/pow(h/1.9,k_dim);  //reference_density2(tp_type,ttemp,m,h,k_dim);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_renormalization_norm2D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	Real xi,yi;
	Real rho_ref_i;
	Real search_range,tmp_h,tmp_A;
	Real tmpn,tmpd;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	rho_ref_i=P2[i].rho_ref;

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
						Real xj,yj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
	
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
						if(tdist<search_range){
							Real twij,mj,rho0j,rho_ref_j;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rho0j=P2[j].rho0;
							rho_ref_j=P2[j].rho_ref;
	
							tmpn+=rho_ref_i*(mj/rho_ref_j)*twij;
							tmpd+=(mj/rho0j)*twij;
						}

					}
					
				
				}
			}
		}
	}
	P2[i].rho0=tmpn/tmpd;
	P3[i].drho=0.0;
	//P1[i].rho=tmpn/tmpd;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_renormalization_norm3D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rho_ref_i;
	Real search_range,tmp_h,tmp_A;
	Real tmpn,tmpd;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	rho_ref_i=P2[i].rho_ref;

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
							if(tdist<search_range){
								Real twij,mj,rho0j,rho_ref_j;
								twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
	
								mj=P1[j].m;
								rho0j=P2[j].rho0;
								rho_ref_j=P2[j].rho_ref;
	
								tmpn+=rho_ref_i*(mj/rho_ref_j)*twij;
								tmpd+=(mj/rho0j)*twij;
							}


						}
						
						
					}
				}
			}
		}
	}
	P2[i].rho0=tmpn/tmpd;
	//P3[i].drho=0.0;
	//P1[i].rho=tmpn/tmpd;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_continuity_norm2D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real rhoi;
	Real grad_rhoxi,grad_rhoyi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,tmp_R;
	int p_type_i;

	p_type_i=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux*(p_type_i>0);
	uyi=P1[i].uy*(p_type_i>0);
	rhoi=P1[i].rho;
	tmp_h=P1[i].h;
	grad_rhoxi=P1[i].grad_rhox;
	grad_rhoyi=P1[i].grad_rhoy;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=tmp_R=0.0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y);
			int_t k=idx_cell(icell+x,jcell+y,0);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					
					if(P1[j].p_type<=1000){

						Real xj,yj,tdist;
						xj=P1[j].x;
						yj=P1[j].y;
	
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
						if(tdist<search_range){
							Real mj,tdwx,tdwy,uxj,uyj,rhoj,grad_rhoxj,grad_rhoyj,phi_ij,tmprho,tmpr;
							int p_type_j;
							Real tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
	
							p_type_j=P1[j].p_type;
	
							mj=P1[j].m;
							rhoj=P1[j].rho;
							uxj=P1[j].ux*(p_type_j>0);
							uyj=P1[j].uy*(p_type_j>0);
							grad_rhoxj=P1[j].grad_rhox;
							grad_rhoyj=P1[j].grad_rhoy;
							tmpr=0.0;
	
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
	
							if(k_kgc_solve>0){
								Real twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);
							}
	
							tmprho=mj*(rhoi/rhoj);
							tmpx+=(uxi-uxj)*tmprho*tdwx;
							tmpy+=(uyi-uyj)*tmprho*tdwy;
	
							if(k_delSPH_solve>0) {
								phi_ij=(grad_rhoxi+grad_rhoxj)*(xj-xi);
								phi_ij+=(grad_rhoyi+grad_rhoyj)*(yj-yi);
								phi_ij=(-0.5*phi_ij)*(k_delSPH_solve==Antuono)+(rhoj-rhoi);
								tmpr=-2*(delta*1.0)*tmp_h*k_soundspeed*(mj/rhoj);
								tmpr*=phi_ij*tdwij/tdist;
								tmp_R+=tmpr;
							}
						}


					}

					
				}
			}
		}
	}
	P3[i].drho=(tmpx+tmpy)*(rhoi/P2[i].rho_ref)+tmp_R*(k_delSPH_solve>0);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_continuity_norm3D(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type<=0)	return;	

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rhoi;
	Real grad_rhoxi,grad_rhoyi,grad_rhozi,pori;
	Real uxi,uyi,uzi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,tmpz,tmp_R;
	Real tmppx,tmppy,tmppz;
	int p_type_i;

	p_type_i=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;

	//uxi=P1[i].ux*(p_type_i!=0);
	//uyi=P1[i].uy*(p_type_i!=0);
	//uzi=P1[i].uz*(p_type_i!=0);

	rhoi=P1[i].rho;
	pori=P1[i].DEMpor;

	grad_rhoxi=P1[i].grad_rhox;
	grad_rhoyi=P1[i].grad_rhoy;
	grad_rhozi=P1[i].grad_rhoz;
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

	tmpx=tmpy=tmpz=tmp_R=0.0;
	tmppx=tmppy=tmppz=0.0;
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
							int p_type_j;
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
							
							p_type_j=P1[j].p_type;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
							if(tdist<search_range){
								Real mj,tdwx,tdwy,tdwz,rhoj,uxj,uyj,uzj,porj,tmprho,grad_rhoxj,grad_rhoyj,grad_rhozj,phi_ij,tmpr;
	
								Real tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
	
								grad_rhoxj=P1[j].grad_rhox;
								grad_rhoyj=P1[j].grad_rhoy;
								grad_rhozj=P1[j].grad_rhoz;
	
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
	
								if(k_kgc_solve>0){
									Real twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
									apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
								}

								//if (P1[j].p_type==0) printf("%f",uzj);
	
							  mj=P1[j].m;
							  porj=P1[j].DEMpor;
								uxj=P1[j].ux;
								uyj=P1[j].uy;
								uzj=P1[j].uz;
								rhoj=P1[j].rho;
								//if (P1[j].p_type==0) printf("%f",uzj);

								tmprho=mj*((rhoi)/(rhoj));
								tmpx+=(uxi-uxj)*tmprho*tdwx;
								tmpy+=(uyi-uyj)*tmprho*tdwy;
								tmpz+=(uzi-uzj)*tmprho*tdwz;
								tmppx+=uxi*mj/rhoj*(rhoi/pori-rhoj/porj)*tdwx;
								tmppy+=uyi*mj/rhoj*(rhoi/pori-rhoj/porj)*tdwy;
								tmppz+=uzi*mj/rhoj*(rhoi/pori-rhoj/porj)*tdwz;
	
								if(k_delSPH_solve>0) {
									Real rhoi_ref=P2[i].rho_ref;
									Real rhoj_ref=P2[j].rho_ref;
	
									phi_ij=(grad_rhoxi+grad_rhoxj)*(xj-xi);
									phi_ij+=(grad_rhoyi+grad_rhoyj)*(yj-yi);
									phi_ij+=(grad_rhozi+grad_rhozj)*(zj-zi);
									phi_ij=(-0.5*phi_ij)*(k_delSPH_solve==Antuono)+(rhoj/rhoj_ref-rhoi/rhoi_ref);
									
									tmpr=-2*(delta)*tmp_h*rhoi_ref*k_soundspeed*(mj/rhoj);
									tmpr*=phi_ij*tdwij/tdist;
									tmp_R+=tmpr;

									

								}
	
							}

						}
						
						
					}
				}
			}
		}
	}
	P3[i].drho=(tmpx+tmpy+tmpz)+pori*(tmppx+tmppy+tmppz)*(1-P1[i].elix)+tmp_R*(k_delSPH_solve>0);
	//P1[i].test1=(tmpx+tmpy+tmpz);
	
	//P1[i].test5=tmp_R;
}


////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_continuity_norm3D_1(int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if((P1[i].elix>1)&(P1[i].eliy>1)) return;
	if(k_open_boundary>0 && P1[i].buffer_type>0) return;
	if(P1[i].p_type>=1000)	return;	
	if(P1[i].p_type<=0)	return;	

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rhoi;
	Real grad_rhoxi,grad_rhoyi,grad_rhozi;
	Real uxi,uyi,uzi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,tmpz,tmp_R;
	Real tmppx,tmppy,tmppz;
	int p_type_i;

	p_type_i=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	rhoi=P1[i].rho;

	grad_rhoxi=P1[i].grad_rhox;
	grad_rhoyi=P1[i].grad_rhoy;
	grad_rhozi=P1[i].grad_rhoz;
	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	// calculate I,J,K in cell
// calculate I,J,K in cell
if((k_x_max==k_x_min)){icell=0;}
else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
if((k_y_max==k_y_min)){jcell=0;}
else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
if((k_z_max==k_z_min)){kcell=0;}
else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// out-of-range handling
if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

tmpx=tmpy=tmpz=tmp_R=0.0;
tmppx=tmppy=tmppz=0.0;
for(int_t z=-1;z<=1;z++){
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
			int_t k=idx_cell(icell+x,jcell+y,kcell+z);

			if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						int p_type_j;
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;
						p_type_j=P1[j].p_type;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
						if(tdist<search_range){
							Real mj,tdwx,tdwy,tdwz,rhoj,uxj,uyj,uzj,tmprho,grad_rhoxj,grad_rhoyj,grad_rhozj,phi_ij,tmpr;

							Real tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							grad_rhoxj=P1[j].grad_rhox;
							grad_rhoyj=P1[j].grad_rhoy;
							grad_rhozj=P1[j].grad_rhoz;

							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
							tdwz=tdwij*(zi-zj)/tdist;

							if(k_kgc_solve>0){
								Real twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
							}

						  	mj=P1[j].m;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							rhoj=P1[j].rho;
							tmprho=mj*(rhoi/rhoj);
							tmpx+=(uxi-uxj)*tmprho*tdwx;
							tmpy+=(uyi-uyj)*tmprho*tdwy;
							tmpz+=(uzi-uzj)*tmprho*tdwz;
							tmppx+=uxi*mj/rhoj*(rhoi-rhoj)*tdwx;
							tmppy+=uyi*mj/rhoj*(rhoi-rhoj)*tdwy;
							tmppz+=uzi*mj/rhoj*(rhoi-rhoj)*tdwz;

							if(k_delSPH_solve>0) {
								// phi_ij=(grad_rhoxi+grad_rhoxj)*(xj-xi);
								// phi_ij+=(grad_rhoyi+grad_rhoyj)*(yj-yi);
								// phi_ij+=(grad_rhozi+grad_rhozj)*(zj-zi);
								// phi_ij=(-0.5*phi_ij)*(k_delSPH_solve==Antuono)+(rhoj-rhoi);
								// tmpr=-2*delta*tmp_h*k_soundspeed*(mj/rhoj);
								// tmpr*=phi_ij*tdwij/tdist;
								// tmp_R+=tmpr;

								Real rhoi_ref=P2[i].rho_ref;
								Real rhoj_ref=P2[j].rho_ref;

								phi_ij=(grad_rhoxi+grad_rhoxj)*(xj-xi);
								phi_ij+=(grad_rhoyi+grad_rhoyj)*(yj-yi);
								phi_ij+=(grad_rhozi+grad_rhozj)*(zj-zi);
								phi_ij=(-0.5*phi_ij)*(k_delSPH_solve==Antuono)+(rhoj/rhoj_ref-rhoi/rhoi_ref);
								tmpr=-2*delta*tmp_h*rhoi_ref*k_soundspeed*(mj/rhoj);
								tmpr*=phi_ij*tdwij/tdist;
								tmp_R+=tmpr;
							}

						}
					}
				}
			}
		}
	}
	P3[i].drho=(tmpx+tmpy+tmpz)+(tmppx+tmppy+tmppz)*(1-P1[i].elix)+tmp_R*(k_delSPH_solve>0);
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_density_renormalization_norm2D(int_t*g_str,int_t*g_end,part1*P1,part1*TP1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	Real xi,yi;
	Real rho_ref_i;
	Real search_range,tmp_h,tmp_A;
	Real tmpn,tmpd;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=TP1[i].x;
	yi=TP1[i].y;
	rho_ref_i=P2[i].rho_ref;

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
						Real xj,yj,tdist;
						xj=TP1[j].x;
						yj=TP1[j].y;
	
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
						if(tdist<search_range){
							Real twij,mj,rhoj,rho_ref_j;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							mj=P1[j].m;
							rhoj=TP1[j].rho;
							rho_ref_j=P2[j].rho_ref;
	
							tmpn+=rho_ref_i*(mj/rho_ref_j)*twij;
							tmpd+=(mj/rhoj)*twij;
						}


					}

					
				}
			}
		}
	}
	TP1[i].rho=tmpn/tmpd;
	//P1[i].rho=tmpn/tmpd;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_density_renormalization_norm3D(int_t*g_str,int_t*g_end,part1*P1,part1*TP1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rho_ref_i;
	Real search_range,tmp_h,tmp_A;
	Real tmpn,tmpd;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	xi=TP1[i].x;
	yi=TP1[i].y;
	zi=TP1[i].z;
	rho_ref_i=P2[i].rho_ref;

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
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

					if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						
						if(P1[j].p_type<=1000){
							Real xj,yj,zj,tdist;
							xj=TP1[j].x;
							yj=TP1[j].y;
							zj=TP1[j].z;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist<search_range){
								Real twij,mj,rhoj,rho_ref_j;
								twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
	
								mj=P1[j].m;
								rhoj=TP1[j].rho;
								rho_ref_j=P2[j].rho_ref;
	
								tmpn+=rho_ref_i*(mj/rho_ref_j)*twij;
								tmpd+=(mj/rhoj)*twij;
							}
							
						}
						
						
					}
				}
			}
		}
	}
	TP1[i].rho=tmpn/tmpd;
	//P1[i].rho=tmpn/tmpd;
}

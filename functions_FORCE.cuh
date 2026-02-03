/*
////////////////////////////////////////////////////////////////////////
// pressure forece calculation
__global__ void KERNEL_clc_pressure_force2D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t icell,jcell;
	Real xi,yi;
	Real pi,rhoi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy;

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	tmp_h=Pa11[i].h;
	pi=Pa11[i].pres;
	rhoi=Pa11[i].rho;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	tmpx=tmpy=0.0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,rr,tdist;
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					tdist=sqrt(rr);
					if(tdist>0&&tdist<search_range){
						Real tdwx,tdwy,C_p; //mj,rhoj,pj

						if(k_kgc_solve==1){
							Real tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							Real tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}
						//mj=Pa11[j].m;
						//rhoj=Pa11[j].rho;
						//pj=Pa11[j].pres;

						C_p=-Pa11[j].m*(pi+Pa11[j].pres)/(rhoi*Pa11[j].rho);

						tmpx+=C_p*tdwx;
						tmpy+=C_p*tdwy;
					}
				}
			}
		}
	}
	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
}
////////////////////////////////////////////////////////////////////////
// pressure forece calculation
__global__ void KERNEL_clc_pressure_force3D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real pi,rhoi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,tmpz;

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	pi=Pa11[i].pres;
	tmp_h=Pa11[i].h;
	rhoi=Pa11[i].rho;

	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	tmpx=tmpy=tmpz=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,rr,tdist;
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);
						tdist=sqrt(rr);
						if(tdist>0&&tdist<search_range){
							Real tdwx,tdwy,tdwz,C_p; //mj,rhoj,pj

							if(k_kgc_solve==1){
								Real tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
								tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								tdwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								Real tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}
							//mj=Pa11[j].m;
							//rhoj=Pa11[j].rho;
							//pj=Pa11[j].pres;

							C_p=-Pa11[j].m*(pi+Pa11[j].pres)/(rhoi*Pa11[j].rho);

							tmpx+=C_p*tdwx;
							tmpy+=C_p*tdwy;
							tmpz+=C_p*tdwz;
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
	Pa11[i].ftotalz=tmpz;
}
////////////////////////////////////////////////////////////////////////
// viscous force calculation
__global__ void KERNEL_add_viscous_force2D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real uxi,uyi;
	Real rhoi,tempi,visi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy;

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	ptypei=Pa11[i].p_type;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	tmp_h=Pa11[i].h;
	tempi=Pa11[i].temp;
	rhoi=Pa11[i].rho;

	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range
	visi=viscosity(tempi,ptypei);
	tmpx=tmpy=0.0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,rr,tdist;
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					tdist=sqrt(rr);
					if(tdist>0&&tdist<search_range){
						int_t ptypej;
						Real tdwx,tdwy,tdwij,mj,rhoj,uxj,uyj,tempj,visj,C_v;

						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						if(k_kgc_solve==1){
							tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}

						ptypej=Pa11[j].p_type;
						uxj=Pa11[j].ux;
						uyj=Pa11[j].uy;
						mj=Pa11[j].m;
						tempj=Pa11[j].temp;
						rhoj=Pa11[j].rho;

						visj=viscosity(tempj,ptypej);
						//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
						//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy)/tdist/tdist;
						C_v=(4*mj*(visi*visj)*((xi-xj)*tdwx+(yi-yj)*tdwy))/((rhoi*rhoj)*(visi+visj)*tdist*tdist);

						tmpx+=C_v*(uxi-uxj);
						tmpy+=C_v*(uyi-uyj);
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
}
////////////////////////////////////////////////////////////////////////
// viscous force calculation
__global__ void KERNEL_add_viscous_force3D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real rhoi,tempi,visi;
	Real uxi,uyi,uzi;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,tmpz;

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	ptypei=Pa11[i].p_type;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	tempi=Pa11[i].temp;
	rhoi=Pa11[i].rho;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range
	visi=viscosity(tempi,ptypei);

	tmpx=tmpy=tmpz=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,rr,tdist;
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);
						tdist=sqrt(rr);
						if(tdist>0&&tdist<search_range){
							int_t ptypej;
							Real tdwx,tdwy,tdwz,tdwij,uxj,uyj,uzj,mj,tempj,rhoj,visj,C_v;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							if(k_kgc_solve==1){
								tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								tdwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}

							ptypej=Pa11[j].p_type;
							uxj=Pa11[j].ux;
							uyj=Pa11[j].uy;
							uzj=Pa11[j].uz;
							mj=Pa11[j].m;
							tempj=Pa11[j].temp;
							rhoj=Pa11[j].rho;

							visj=viscosity(tempj,ptypej);
							//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
							//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
							C_v=(4*mj*(visi*visj)*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz))/((rhoi*rhoj)*(visi+visj)*tdist*tdist);

							tmpx+=C_v*(uxi-uxj);
							tmpy+=C_v*(uyi-uyj);
							tmpz+=C_v*(uzi-uzj);
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
}
////////////////////////////////////////////////////////////////////////
// viscous force calculation
__global__ void KERNEL_add_HB_viscous_force2D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t x,y,k,fend,j;
	int_t icell,jcell;
	Real rr,tdwij;
	Real xi,yi,xj,yj;

	Real mj,rhoi,rhoj;
	Real visi,visj;
	Real uxi,uyi,uxj,uyj;
	Real tdwx,tdwy,tdist;
	Real C_v;
	Real tmpx,tmpy;
	Real search_range,tmp_h,tmp_A;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	rhoi=Pa11[i].rho;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;

	tmpx=tmpy=0.0;
	for(y=-1;y<=1;y++){
		for(x=-1;x<=1;x++){
			k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				fend=g_end[k];
				for(j=g_str[k];j<fend;j++){
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					rr=0.0;
					rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					tdist=sqrt(rr);
					if(tdist>0&&tdist<search_range){
						tdwx=tdwy=0.0;
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

						if(k_kgc_solve==1){
							tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}

						mj=Pa11[j].m;
						rhoj=Pa11[j].rho;
						uxj=Pa11[j].ux;
						uyj=Pa11[j].uy;

						visi=Pa13[i].vis_t;
						visj=Pa13[j].vis_t;
						//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
						C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj+1e-20))*((xi-xj)*tdwx+(yi-yj)*tdwy)/tdist/tdist;

						tmpx+=C_v*(uxi-uxj);
						tmpy+=C_v*(uyi-uyj);
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
}
////////////////////////////////////////////////////////////////////////
// viscous force calculation
__global__ void KERNEL_add_HB_viscous_force3D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t x,y,z,k,fend,j;
	int_t icell,jcell,kcell;
	Real rr,tdwij;
	Real xi,yi,zi,xj,yj,zj;

	Real mj,rhoi,rhoj;
	Real visi,visj;
	Real uxi,uyi,uzi,uxj,uyj,uzj;
	Real tdwx,tdwy,tdwz,tdist;
	Real C_v;
	Real tmpx,tmpy,tmpz;
	Real search_range,tmp_h,tmp_A;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	rhoi=Pa11[i].rho;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;

	tmpx=tmpy=tmpz=0.0;
	for(z=-1;z<=1;z++){
		for(y=-1;y<=1;y++){
			for(x=-1;x<=1;x++){
				k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					fend=g_end[k];
					for(j=g_str[k];j<fend;j++){
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						rr=0.0;
						rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);
						tdist=sqrt(rr);
						if(tdist>0&&tdist<search_range){
							tdwx=tdwy=tdwz=0.0;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							if(k_kgc_solve==1){
								tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								tdwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}

							mj=Pa11[j].m;
							rhoj=Pa11[j].rho;
							uxj=Pa11[j].ux;
							uyj=Pa11[j].uy;
							uzj=Pa11[j].uz;

							visi=Pa13[i].vis_t;
							visj=Pa13[j].vis_t;
							//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
							C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj+1e-20))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;

							tmpx+=C_v*(uxi-uxj);
							tmpy+=C_v*(uyi-uyj);
							tmpz+=C_v*(uzi-uzj);
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_artificial_viscous_force2D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t icell,jcell;
	Real xi,yi;

	Real mi,rhoi,hi; //ci,cj,c_ij
	Real uxi,uyi;
	Real tmpx,tmpy;
	Real search_range,tmp_A;

	hi=Pa11[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;

	tmpx=tmpy=0.0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,rr,tdist;
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					tdist=sqrt(rr);
					if(tdist>0&&tdist<search_range){
						Real uxj,uyj,uij_xij;

						uxj=Pa11[j].ux;
						uyj=Pa11[j].uy;
						uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj);

						if(uij_xij<0){	//else P_ij=0;
							Real tdwx,tdwy,hj,rhoj,rho_ij,phi_ij,P_ij,h_ij;
							if(k_kgc_solve==1){
								Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
								tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
								tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
							}else{
								Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
							}
							hj=Pa11[j].h;
							rhoj=Pa11[j].rho;
							rho_ij=(rhoi+rhoj)*0.5;
							h_ij=(hi+hj)*0.5;

							phi_ij=h_ij*uij_xij/(tdist*tdist+0.01*h_ij*h_ij);
							P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/rho_ij;
							tmpx+=-(P_ij)*tdwx;
							tmpy+=-(P_ij)*tdwy;
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_artificial_viscous_force3D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real mi,rhoi,hi;	//c_ij,ci,cj
	Real uxi,uyi,uzi;
	Real tmpx,tmpy,tmpz;
	Real search_range,tmp_A;

	hi=Pa11[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;

	//c_ij=(k_soundspeed+k_soundspeed)*0.5;	//ci=cj=tsoundspeed;
	tmpx=tmpy=tmpz=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
						if(tdist>0&&tdist<search_range){
							Real uxj,uyj,uzj,uij_xij;
							uxj=Pa11[j].ux;
							uyj=Pa11[j].uy;
							uzj=Pa11[j].uz;
							uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
							if(uij_xij<0){	//else P_ij=0;
								Real tdwx,tdwy,tdwz,hj,rhoj,rho_ij,phi_ij,P_ij,h_ij;
								if(k_kgc_solve==1){
									Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
									tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
									tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
									tdwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
								}else{
									Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
									tdwx=tdwij*(xi-xj)/tdist;
									tdwy=tdwij*(yi-yj)/tdist;
									tdwz=tdwij*(zi-zj)/tdist;
								}


								hj=Pa11[j].h;
								rhoj=Pa11[j].rho;
								rho_ij=(rhoi+rhoj)*0.5;
								h_ij=(hi+hj)*0.5;

								phi_ij=h_ij*uij_xij/(tdist*tdist+0.01*h_ij*h_ij);
								P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/rho_ij;
								tmpx+=-(P_ij)*tdwx;
								tmpy+=-(P_ij)*tdwy;
								tmpz+=-(P_ij)*tdwz;
							}
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_gravity_force(part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	switch(k_dim){
		case 3:
			Pa11[i].ftotalz+=-Gravitational_CONST;				// z-directional gravitational force
			break;
		case 2:
			Pa11[i].ftotaly+=-Gravitational_CONST;				// y-directional gravitational force
			break;
		default:
			break;
	}
}
//*/
////////////////////////////////////////////////////////////////////////
// calcuate color field for two-phase flow surface tension model (2017.05.08 jyb)
__global__ void KERNEL_clc_color_field2D(int_t*g_str,int_t*g_end,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real tmpn,tmpd;
	Real search_range,tmp_h,tmp_A;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	tmpn=tmpd=0.0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist; //rr
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);tdist=sqrt(rr);
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
					if(tdist<search_range){
						Real twij,mj,rhoj;
						// calculate contribution of j particle on density variation (drho)
						// kernel & distance
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);

						mj=Pa11[j].m;
						rhoj=Pa11[j].rho;

						tmpn+=mj*twij*(ptypei==Pa11[j].p_type)/rhoj;
						tmpd+=mj*twij/rhoj;
					}
				}
			}
		}
	}
	Pa11[i].cc=tmpn/tmpd;
}
////////////////////////////////////////////////////////////////////////
// calcuate color field for two-phase flow surface tension model (2017.05.08 jyb)
__global__ void KERNEL_clc_color_field3D(int_t*g_str,int_t*g_end,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real tmpn,tmpd;
	Real search_range,tmp_h,tmp_A;

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	ptypei=Pa11[i].p_type;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	tmpn=tmpd=0.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist; //rr
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);tdist=sqrt(rr);
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

						if(tdist<search_range){
							Real twij,mj,rhoj;
							// calculate contribution of j particle on density variation (drho)
							// kernel & distance
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);

							mj=Pa11[j].m;
							rhoj=Pa11[j].rho;

							tmpn+=mj*twij*(ptypei==Pa11[j].p_type)/rhoj;
							tmpd+=mj*twij/rhoj;
						}
					}
				}
			}
		}
	}
	Pa11[i].cc=tmpn/tmpd;
}
////////////////////////////////////////////////////////////////////////
// calcuate normal gradient vector for two-phase flow surface tension model (2017.04.20 jyb)
__global__ void KERNEL_clc_normal_gradient_c2D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real cci;
	Real tmpx,tmpy;
	Real search_range,tmp_h,tmp_A;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	cci=Pa11[i].cc;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	tmpx=tmpy=0.0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist; //rr
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);tdist=sqrt(rr);
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

					if(tdist>0&&tdist<search_range){
						Real tdwij,mj,rhoj,ccj,C_s;
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

						mj=Pa11[j].m;
						rhoj=Pa11[j].rho;
						ccj=Pa11[j].cc;

						//(fluid_1_i*fluid_1_j+fluid_2_i*fluid_2_j+boundary_i*boundary_j)
						C_s=(mj/rhoj)*(ccj-cci)*(ptypei==Pa11[j].p_type)*tdwij/tdist;

						tmpx+=C_s*(xj-xi);
						tmpy+=C_s*(yj-yi);
					}
				}
			}
		}
	}
	Pa13[i].nx_c=tmpx;
	Pa13[i].ny_c=tmpy;
	Real tmpnmg=sqrt(tmpx*tmpx+tmpy*tmpy);
	Pa13[i].nmag_c=tmpnmg;
	if(tmpnmg<NORMAL_THRESHOLD){
		Pa13[i].nx_c=0;
		Pa13[i].ny_c=0;
		Pa13[i].nmag_c=1e-20;
	}
}
////////////////////////////////////////////////////////////////////////
// calcuate normal gradient vector for two-phase flow surface tension model (2017.04.20 jyb)
__global__ void KERNEL_clc_normal_gradient_c3D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real cci;
	Real tmpx,tmpy,tmpz;
	Real search_range,tmp_h,tmp_A;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	cci=Pa11[i].cc;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	tmpx=tmpy=tmpz=0.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);tdist=sqrt(rr);
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
						if(tdist>0&&tdist<search_range){
							Real tdwij,mj,rhoj,ccj,C_s;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							mj=Pa11[j].m;
							rhoj=Pa11[j].rho;
							ccj=Pa11[j].cc;

							//(fluid_1_i*fluid_1_j+fluid_2_i*fluid_2_j+boundary_i*boundary_j)
							C_s=(mj/rhoj)*(ccj-cci)*(ptypei==Pa11[j].p_type)*tdwij/tdist;

							tmpx+=C_s*(xj-xi);
							tmpy+=C_s*(yj-yi);
							tmpz+=C_s*(zj-zi);
						}
					}
				}
			}
		}
	}
	Pa13[i].nx_c=tmpx;
	Pa13[i].ny_c=tmpy;
	Pa13[i].nz_c=tmpz;
	Real tmpnmg=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);
	Pa13[i].nmag_c=tmpnmg;
	if(tmpnmg<NORMAL_THRESHOLD){
		Pa13[i].nx_c=0;
		Pa13[i].ny_c=0;
		Pa13[i].nz_c=0;
		Pa13[i].nmag_c=1e-20;
	}
}
////////////////////////////////////////////////////////////////////////
// calcuate normal gradient vector for two-phase flow surface tension model (2017.04.20 jyb)
__global__ void KERNEL_clc_normal_gradient2D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;
	Real tmpx,tmpy;
	Real xi,yi;
	Real mi,rhoi;
	Real search_range,tmp_h,tmp_A;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	xi=Pa11[i].x;
	yi=Pa11[i].y;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	tmpx=tmpy=0.0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);tdist=sqrt(rr);
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
					if(tdist>0&&tdist<search_range){
						Real tdwij,mj,rhoj,C_s,C_sx,C_sy,C_st;
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

						mj=Pa11[j].m;
						rhoj=Pa11[j].rho;

						C_s=(ptypei!=Pa11[j].p_type);

						//tmpx+=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))*(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij*(xj-xi)/tdist;
						//tmpy+=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))*(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij*(yj-yi)/tdist;
						C_st=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
						C_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij;
						C_sx=C_st*(xj-xi)/tdist;
						C_sy=C_st*(yj-yi)/tdist;

						tmpx+=C_sx;
						tmpy+=C_sy;


					}
				}
			}
		}
	}
	Pa13[i].nx=tmpx;
	Pa13[i].ny=tmpy;
	Real tmpnmg=sqrt(tmpx*tmpx+tmpy*tmpy);
	Pa13[i].nmag=tmpnmg;
	if(tmpnmg<NORMAL_THRESHOLD){
		Pa13[i].nx=0;
		Pa13[i].ny=0;
		Pa13[i].nmag=1e-20;
	}
}
////////////////////////////////////////////////////////////////////////
// calcuate normal gradient vector for two-phase flow surface tension model (2017.04.20 jyb)
__global__ void KERNEL_clc_normal_gradient3D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real tmpx,tmpy,tmpz;
	Real xi,yi,zi;
	Real mi,rhoi;
	Real search_range,tmp_h,tmp_A;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	tmpx=tmpy=tmpz=0.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);tdist=sqrt(rr);
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
						if(tdist>0&&tdist<search_range){
							Real tdwij,mj,rhoj,C_s,C_sx,C_sy,C_sz,C_st;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							mj=Pa11[j].m;
							rhoj=Pa11[j].rho;
							C_s=(ptypei!=Pa11[j].p_type);

							C_st=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj));
							C_st*=(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij;

							C_sx=C_st*(xj-xi)/tdist;
							C_sy=C_st*(yj-yi)/tdist;
							C_sz=C_st*(zj-zi)/tdist;

							tmpx+=C_sx;
							tmpy+=C_sy;
							tmpz+=C_sz;
						}
					}
				}
			}
		}
	}
	Pa13[i].nx=tmpx;
	Pa13[i].ny=tmpy;
	Pa13[i].nz=tmpz;
	Real tmpnmg=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);
	Pa13[i].nmag=tmpnmg;

	if(tmpnmg<NORMAL_THRESHOLD){
		Pa13[i].nx=0;
		Pa13[i].ny=0;
		Pa13[i].nz=0;
		Pa13[i].nmag=1e-20;
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_surface_tension2D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;

	Real xi,yi;
	Real sigmai,rhoi,hi,tempi;
	Real nxi,nyi,nmagi;
	Real nx_ci,ny_ci,nmag_ci,curvi;
	Real search_range,tmp_A;
	Real tmpn,tmpd;

	hi=Pa11[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	tempi=Pa11[i].temp;
	rhoi=Pa11[i].rho;
	nxi=Pa13[i].nx;
	nyi=Pa13[i].ny;
	nmagi=Pa13[i].nmag;
	nx_ci=Pa13[i].nx_c;
	ny_ci=Pa13[i].ny_c;
	nmag_ci=Pa13[i].nmag_c;

	sigmai=sigma(tempi,ptypei);

	tmpn=tmpd=0.0;
	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist;
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					//tdist=sqrt(rr);
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

					if(tdist>0&&tdist<search_range){
						int_t ptypej;
						Real tdwij,mj,rhoj,nx_cj,ny_cj,nmag_cj,Phi_s,tmpnt;
						tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

						ptypej=Pa11[j].p_type;
						mj=Pa11[j].m;
						rhoj=Pa11[j].rho;
						nx_cj=Pa13[j].nx_c;
						ny_cj=Pa13[j].ny_c;
						nmag_cj=Pa13[j].nmag_c;
						Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);

						//tmpnt=k_dim*(mj/rhoj)*(((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi)+((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi))*tdwij/tdist;
						tmpnt=((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi);
						tmpnt+=((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi);
						tmpnt*=k_dim*(mj/rhoj)*tdwij/tdist;
						tmpn+=tmpnt;
						tmpd+=(mj/rhoj)*tdist*abs(tdwij);
					}
				}
			}
		}
	}
	if((nmagi>0.1/hi)&(tmpn>0)) Pa13[i].curv=tmpn/tmpd;
	else Pa13[i].curv=0;

	curvi=Pa13[i].curv;
	Pa11[i].ftotalx+=sigmai*curvi*nxi/rhoi;
	Pa11[i].ftotaly+=sigmai*curvi*nyi/rhoi;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_surface_tension3D(int_t*g_str,int_t*g_end,part11*Pa11,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei,ptypej;
	int_t icell,jcell,kcell;

	Real xi,yi,zi;
	Real sigmai,rhoi,hi,tempi;
	Real nxi,nyi,nzi,nmagi;
	Real nx_ci,ny_ci,nz_ci,nmag_ci,curvi;
	Real search_range,tmp_A;
	Real tmpn,tmpd;

	hi=Pa11[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	ptypei=Pa11[i].p_type;
	tempi=Pa11[i].temp;
	rhoi=Pa11[i].rho;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	nxi=Pa13[i].nx;
	nyi=Pa13[i].ny;
	nzi=Pa13[i].nz;
	nmagi=Pa13[i].nmag;
	nx_ci=Pa13[i].nx_c;
	ny_ci=Pa13[i].ny_c;
	nz_ci=Pa13[i].nz_c;
	nmag_ci=Pa13[i].nmag_c;

	sigmai=sigma(tempi,ptypei);

	tmpn=tmpd=0.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist;
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);
						//tdist=sqrt(rr);
						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
						if(tdist>0&&tdist<search_range){
							Real tdwij,mj,rhoj,nx_cj,ny_cj,nz_cj,nmag_cj,Phi_s,tmpnt;
							tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

							ptypej=Pa11[j].p_type;
							mj=Pa11[j].m;
							rhoj=Pa11[j].rho;
							nx_cj=Pa13[j].nx_c;
							ny_cj=Pa13[j].ny_c;
							nz_cj=Pa13[j].nz_c;
							nmag_cj=Pa13[j].nmag_c;

							Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);

							//tmpnt=k_dim*(mj/rhoj)*(((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi)+((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi)+((nz_ci/nmag_ci)-Phi_s*(nz_cj/nmag_cj))*(zj-zi))*tdwij/tdist;
							tmpnt=((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi);
							tmpnt+=((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi);
							tmpnt+=((nz_ci/nmag_ci)-Phi_s*(nz_cj/nmag_cj))*(zj-zi);
							tmpnt*=k_dim*(mj/rhoj)*tdwij/tdist;

							tmpn+=tmpnt;
							tmpd+=(mj/rhoj)*tdist*abs(tdwij);
						}
					}
				}
			}
		}
	}
	if ((nmagi>0.1/hi)&(tmpn>0)) Pa13[i].curv=tmpn/tmpd;
	else Pa13[i].curv=0;

	curvi=Pa13[i].curv;
	Pa11[i].ftotalx+=sigmai*curvi*nxi/rhoi;
	Pa11[i].ftotaly+=sigmai*curvi*nyi/rhoi;
	Pa11[i].ftotalz+=sigmai*curvi*nzi/rhoi;
}
/*
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_interface_sharpness2D(int_t*g_str,int_t*g_end,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real mi,pi,rhoi,mri,mi8;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	ptypei=Pa11[i].p_type;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	mi=Pa11[i].m;
	pi=Pa11[i].pres;
	rhoi=Pa11[i].rho;

	mri=(mi/rhoi);
	mi8=0.08/mi;
	tmpx=tmpy=0.0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,rr,tdist;
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					tdist=sqrt(rr);

					if(tdist>0&&tdist<search_range){
						int_t ptypej,flag;
						Real mj,pj,rhoj,tdwij,C_i,mrj;

						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

						ptypej=Pa11[j].p_type;
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						mj=Pa11[j].m;
						pj=Pa11[j].pres;
						rhoj=Pa11[j].rho;

						mrj=(mj/rhoj);
						flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
						C_i=mi8*(abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag))*tdwij/tdist;

						//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
						// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
						tmpx+=C_i*(xj-xi);
						tmpy+=C_i*(yj-yi);
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_interface_sharpness3D(int_t*g_str,int_t*g_end,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real mi,pi,rhoi,mri,mi8;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy,tmpz;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	mi=Pa11[i].m;
	pi=Pa11[i].pres;
	rhoi=Pa11[i].rho;
	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	mri=(mi/rhoi);
	mi8=0.08/mi;
	tmpx=tmpy=tmpz=0.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,rr,tdist;
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);
						tdist=sqrt(rr);

						if(tdist>0&&tdist<search_range){
							int_t ptypej,flag;
							Real mj,pj,rhoj,tdwij,C_i,mrj;

							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							ptypej=Pa11[j].p_type;
							xj=Pa11[j].x;
							yj=Pa11[j].y;
							zj=Pa11[j].z;
							mj=Pa11[j].m;
							pj=Pa11[j].pres;
							rhoj=Pa11[j].rho;

							mrj=(mj/rhoj);
							flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
							C_i=mi8*(abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag))*tdwij/tdist;

							//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
							// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
							tmpx+=C_i*(xj-xi);
							tmpy+=C_i*(yj-yi);
							tmpz+=C_i*(zj-zi);
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_boundary_force2D(int_t*g_str,int_t*g_end,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	uint_t p_type_i,p_type_j;
	int_t x,y,k,fend,j;
	int_t icell,jcell;
	Real rr;
	Real tmpx,tmpy;

	Real xi,yi,xj,yj;
	Real mi,mj;
	Real twij,tdist;
	Real fb_ij;

	Real search_range,tmp_h,tmp_A;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	p_type_i=Pa11[i].p_type;
	mi=Pa11[i].m;
	xi=Pa11[i].x;
	yi=Pa11[i].y;

	tmpx=tmpy=0.0;

	for(y=-1;y<=1;y++){
		for(x=-1;x<=1;x++){
			k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				fend=g_end[k];
				for(j=g_str[k];j<fend;j++){
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					rr=0.0;
					rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					tdist=sqrt(rr);

					if(tdist<search_range){
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);

						p_type_j=Pa11[j].p_type;
						mj=Pa11[j].m;

						if((p_type_i==FLUID)&(p_type_j!=FLUID)){
							fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
							tmpx+=fb_ij*(xi-xj);
							tmpy+=fb_ij*(yi-yj);
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_boundary_force3D(int_t*g_str,int_t*g_end,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	uint_t p_type_i,p_type_j;
	int_t x,y,z,k,fend,j;
	int_t icell,jcell,kcell;
	Real rr;
	Real tmpx,tmpy,tmpz;

	Real xi,yi,zi,xj,yj,zj;
	Real mi,mj;
	Real twij,tdist;
	Real fb_ij;

	Real search_range,tmp_h,tmp_A;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	p_type_i=Pa11[i].p_type;
	mi=Pa11[i].m;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	tmpx=tmpy=tmpz=0.0;

	for(z=-1;z<=1;z++){
		for(y=-1;y<=1;y++){
			for(x=-1;x<=1;x++){
				k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					fend=g_end[k];
					for(j=g_str[k];j<fend;j++){
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						rr=0.0;
						rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);
						tdist=sqrt(rr);

						if(tdist<search_range){
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);

							p_type_j=Pa11[j].p_type;
							mj=Pa11[j].m;

							if((p_type_i==FLUID)&(p_type_j!=FLUID)){
								fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
								tmpx+=fb_ij*(xi-xj);
								tmpy+=fb_ij*(yi-yj);
								tmpz+=fb_ij*(zi-zj);
							}
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
}
////////////////////////////////////////////////////////////////////////
// natural convection force (boussinesq approximation)
__global__ void KERNEL_add_boussinesq_force2D(int_t*g_str,int_t*g_end,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	uint_t p_type_i,p_type_j;
	Real tempi,tempj,betai;			// beta : thermal expansion coefficient
	Real mj,rhoj,twij;
	Real xi,yi,xj,yj;

	int_t x,y,k,fend,j;
	int_t icell,jcell;
	Real rr,tdist;
	Real tmpn,tmpd;
	Real search_range,tmp_h,tmp_A;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	p_type_i=Pa11[i].p_type;
	tempi=Pa11[i].temp;

	tmpn=0.0;
	tmpd=1.0;

	for(y=-1;y<=1;y++){
		for(x=-1;x<=1;x++){
			k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				fend=g_end[k];
				for(j=g_str[k];j<fend;j++){
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					rr=0.0;
					rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);
					tdist=sqrt(rr);

					if(tdist<search_range){
						twij=calc_kernel_wij(tmp_A,tmp_h,tdist);

						p_type_j=Pa11[j].p_type;
						mj=Pa11[j].m;
						rhoj=Pa11[j].rho;
						tempj=Pa11[j].temp;

						betai=thermal_expansion(tempi,p_type_i);

						if((p_type_i!=BOUNDARY)&(p_type_i!=MOVING)){
							tmpn+=mj*(tempj-tempi)*twij*(p_type_i==p_type_j)/rhoj;
							tmpd+=mj*twij*(p_type_i==p_type_j)/rhoj;
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotaly+=-betai*Gravitational_CONST*(tmpn/tmpd);				// y-directional gravitational force
}
////////////////////////////////////////////////////////////////////////
// natural convection force (boussinesq approximation)
__global__ void KERNEL_add_boussinesq_force3D(int_t*g_str,int_t*g_end,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	uint_t p_type_i,p_type_j;
	Real tempi,tempj,betai;			// beta : thermal expansion coefficient
	Real mj,rhoj,twij;
	Real xi,yi,zi,xj,yj,zj;

	int_t x,y,z,k,fend,j;
	int_t icell,jcell,kcell;
	Real rr,tdist;
	Real tmpn,tmpd;
	Real search_range,tmp_h,tmp_A;

	tmp_h=Pa11[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	p_type_i=Pa11[i].p_type;
	tempi=Pa11[i].temp;

	tmpn=0.0;
	tmpd=1.0;

	for(z=-1;z<=1;z++){
		for(y=-1;y<=1;y++){
			for(x=-1;x<=1;x++){
				k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					fend=g_end[k];
					for(j=g_str[k];j<fend;j++){
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						rr=0.0;
						rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj);
						tdist=sqrt(rr);

						if(tdist<search_range){
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);

							p_type_j=Pa11[j].p_type;
							mj=Pa11[j].m;
							rhoj=Pa11[j].rho;
							tempj=Pa11[j].temp;

							betai=thermal_expansion(tempi,p_type_i);

							if((p_type_i!=BOUNDARY)&(p_type_i!=MOVING)){
								tmpn+=mj*(tempj-tempi)*twij*(p_type_i==p_type_j)/rhoj;
								tmpd+=mj*twij*(p_type_i==p_type_j)/rhoj;
							}
						}
					}
				}
			}
		}
	}
	Pa11[i].ftotalz+=-betai*Gravitational_CONST*(tmpn/tmpd);				// z-directional gravitational force
}
//*/
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_ftotal(part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	Real tmpx,tmpy,tmpz;
	tmpx=Pa11[i].ftotalx;
	tmpy=Pa11[i].ftotaly;
	tmpz=Pa11[i].ftotalz;

	Pa11[i].ftotal=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);
}
////////////////////////////////////////////////////////////////////////
__global__ void coco_viscous_force2D(int_t*g_str,int_t*g_end,part11*Pa11,part12*Pa12,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpn,tmpd;
	Real concni;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	hi=Pa11[i].h;
	tempi=Pa11[i].temp;
	pi=Pa11[i].pres;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	concni=Pa12[i].concn;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	mi8=0.08/mi;
	mri=(mi/rhoi);

	visi=viscosity(tempi,ptypei);
	betai=thermal_expansion(tempi,ptypei);

	tmpx=tmpy=0.0;
	tmpn=0.0;
	tmpd=1.0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist; //rr
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);tdist=sqrt(rr);
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

					if(tdist>0&&tdist<search_range){
						int_t ptypej;
						Real tdwx,tdwy,uxj,uyj,mj,tempj,rhoj,pj,hj;
						Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

						if(k_kgc_solve==1){
							tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}

						ptypej=Pa11[j].p_type;
						uxj=Pa11[j].ux;
						uyj=Pa11[j].uy;
						mj=Pa11[j].m;
						tempj=Pa11[j].temp;
						rhoj=Pa11[j].rho;
						pj=Pa11[j].pres;
						hj=Pa11[j].h;

						// viscousity
						if(k_fp_solve){
							Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
							tmpx+=C_p*tdwx;
							tmpy+=C_p*tdwy;
						}
						if(k_fv_solve){
							Real visj,C_v;
							visj=viscosity(tempj,ptypej);

							C_v=(xi-xj)*tdwx+(yi-yj)*tdwy;
							C_v*=(visi*visj)/(visi+visj+1e-20);
							C_v*=4*0.5875*(((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))/mi);
							C_v/=tdist*tdist;

							tmpx+=C_v*(uxi-uxj);
							tmpy+=C_v*(uyi-uyj);
						}
						if(k_fva_solve){
							Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj);
							if(uij_xij<0){
								Real h_ij,phi_ij,P_ij;
								h_ij=(hi+hj)*0.5;
								phi_ij=h_ij*uij_xij;
								phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
								P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
								P_ij/=(rhoi+rhoj)*0.5;
								//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
								tmpx+=-(P_ij)*tdwx;
								tmpy+=-(P_ij)*tdwy;
							}
						}
						if(k_interface_solve){
							int_t flag;
							Real mrj,C_i;
							flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
							mrj=mj/rhoj;
							C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
							C_i*=mi8*tdwij/tdist;
							//C_i=mi8*(abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag))*tdwij/tdist;
							//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
							// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
							tmpx+=C_i*(xj-xi);
							tmpy+=C_i*(yj-yi);

						}
						if(k_fb_solve){
							if((ptypei==FLUID)&(ptypej!=FLUID)){
								Real twij=calc_kernel_wij(tmp_A,hi,tdist);
								Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
								tmpx+=fb_ij*(xi-xj);
								tmpy+=fb_ij*(yi-yj);
							}
						}
					}
				}
			}
		}
	}
	// y-directional gravitational force
	if(k_fg_solve) tmpy+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpy+=Gravitational_CONST*(alpha_T*(tempi-T_ref0));

	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void coco_turbulence_viscous_force2D(int_t*g_str,int_t*g_end,part11*Pa11,part12*Pa12,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpn,tmpd;
	Real concni;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	hi=Pa11[i].h;
	tempi=Pa11[i].temp;
	pi=Pa11[i].pres;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	concni=Pa12[i].concn;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	mi8=0.08/mi;
	mri=(mi/rhoi);

	visi=viscosity(tempi,ptypei)+Pa13[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	tmpx=tmpy=0.0;
	tmpn=0.0;
	tmpd=1.0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist; //rr
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);tdist=sqrt(rr);
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

					if(tdist>0&&tdist<search_range){
						int_t ptypej;
						Real tdwx,tdwy,uxj,uyj,mj,tempj,rhoj,pj,hj;
						Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

						if(k_kgc_solve==1){
							tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}

						ptypej=Pa11[j].p_type;
						uxj=Pa11[j].ux;
						uyj=Pa11[j].uy;
						mj=Pa11[j].m;
						tempj=Pa11[j].temp;
						rhoj=Pa11[j].rho;
						pj=Pa11[j].pres;
						hj=Pa11[j].h;

						if(k_fp_solve){
							Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
							tmpx+=C_p*tdwx;
							tmpy+=C_p*tdwy;
						}
						if(k_fv_solve){
							Real visj,C_v;
							visj=viscosity(tempj,ptypej)+Pa13[j].vis_t;
							C_v=(xi-xj)*tdwx+(yi-yj)*tdwy;
							C_v*=4*(mj/(rhoi*rhoj));
							C_v*=(visi*visj)/(visi+visj+1e-20);
							C_v/=tdist*tdist;

							tmpx+=C_v*(uxi-uxj);
							tmpy+=C_v*(uyi-uyj);
						}
						if(k_fva_solve){
							Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj);
							if(uij_xij<0){
								Real h_ij,phi_ij,P_ij;
								h_ij=(hi+hj)*0.5;
								phi_ij=h_ij*uij_xij;
								phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
								P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
								P_ij/=(rhoi+rhoj)*0.5;
								//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
								tmpx+=-(P_ij)*tdwx;
								tmpy+=-(P_ij)*tdwy;
							}
						}
						if(k_interface_solve){
							int_t flag;
							Real mrj,C_i;
							flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
							mrj=mj/rhoj;
							C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
							C_i*=mi8*tdwij/tdist;
							//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
							// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
							tmpx+=C_i*(xj-xi);
							tmpy+=C_i*(yj-yi);

						}
						if(k_fb_solve){
							if((ptypei==FLUID)&(ptypej!=FLUID)){
								Real twij=calc_kernel_wij(tmp_A,hi,tdist);
								Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
								tmpx+=fb_ij*(xi-xj);
								tmpy+=fb_ij*(yi-yj);
							}
						}
					}
				}
			}
		}
	}
	// y-directional gravitational force
	if(k_fg_solve) tmpy+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpy+=Gravitational_CONST*(alpha_T*(tempi-T_ref0)-alpha_S*(concni-S_ref0));

	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void coco_SPS_viscous_force2D(int_t*g_str,int_t*g_end,part11*Pa11,part12*Pa12,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpn,tmpd;
	Real tau_xx_i,tau_xy_i,tau_yx_i,tau_yy_i;
	Real concni;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	hi=Pa11[i].h;
	tempi=Pa11[i].temp;
	pi=Pa11[i].pres;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	concni=Pa12[i].concn;

	tau_xx_i=Pa13[i].Sxx;
	tau_xy_i=Pa13[i].Sxy;
	tau_yx_i=tau_xy_i;
	tau_yy_i=Pa13[i].Syy;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	mi8=0.08/mi;
	mri=(mi/rhoi);

	visi=viscosity(tempi,ptypei)+Pa13[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	tmpx=tmpy=0.0;
	tmpn=0.0;
	tmpd=1.0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist; //rr
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);tdist=sqrt(rr);
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

					if(tdist>0&&tdist<search_range){
						int_t ptypej;
						Real tdwx,tdwy,uxj,uyj,mj,tempj,rhoj,pj,hj;
						Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

						if(k_kgc_solve==1){
							tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}

						ptypej=Pa11[j].p_type;
						uxj=Pa11[j].ux;
						uyj=Pa11[j].uy;
						mj=Pa11[j].m;
						tempj=Pa11[j].temp;
						rhoj=Pa11[j].rho;
						pj=Pa11[j].pres;
						hj=Pa11[j].h;

						if(k_fp_solve){
							Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
							tmpx+=C_p*tdwx;
							tmpy+=C_p*tdwy;
						}
						if(k_fv_solve){
							Real visj,C_v,C_vx,C_vy;
							Real tau_xx_j,tau_xy_j,tau_yx_j,tau_yy_j;
							tau_xx_j=Pa13[j].Sxx;
							tau_xy_j=Pa13[j].Sxy;
							tau_yx_j=tau_xy_j;
							tau_yy_j=Pa13[j].Syy;

							visj=viscosity(tempj,ptypej);
							C_v=(xi-xj)*tdwx+(yi-yj)*tdwy;
							C_v*=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj));
							C_v/=tdist*tdist;
							C_vx=(tau_xx_i/(rhoi*rhoi)+tau_xx_j/(rhoj*rhoj)*tdwx);
							C_vx+=(tau_xy_i/(rhoi*rhoi)+tau_xy_j/(rhoj*rhoj)*tdwy);
							C_vx*=mj;
							C_vx+=C_v*(uxi-uxj);
							C_vy=(tau_yx_i/(rhoi*rhoi)+tau_yx_j/(rhoj*rhoj)*tdwx);
							C_vy+=(tau_yy_i/(rhoi*rhoi)+tau_yy_j/(rhoj*rhoj)*tdwy);
							C_vy*=mj;
							C_vy+=C_v*(uyi-uyj);

							tmpx+=C_vx;
							tmpy+=C_vy;
							//tmpx+=C_v*(uxi-uxj)+mj*((tau_xx_i/(rhoi*rhoi)+tau_xx_j/(rhoj*rhoj)*tdwx)+(tau_xy_i/(rhoi*rhoi)+tau_xy_j/(rhoj*rhoj)*tdwy));
							//tmpy+=C_v*(uyi-uyj)+mj*((tau_yx_i/(rhoi*rhoi)+tau_yx_j/(rhoj*rhoj)*tdwx)+(tau_yy_i/(rhoi*rhoi)+tau_yy_j/(rhoj*rhoj)*tdwy));
						}
						if(k_fva_solve){
							Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj);
							if(uij_xij<0){
								Real h_ij,phi_ij,P_ij;
								h_ij=(hi+hj)*0.5;
								phi_ij=h_ij*uij_xij;
								phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
								P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
								P_ij/=(rhoi+rhoj)*0.5;
								//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
								tmpx+=-(P_ij)*tdwx;
								tmpy+=-(P_ij)*tdwy;
							}
						}
						if(k_interface_solve){
							int_t flag;
							Real mrj,C_i;
							flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
							mrj=mj/rhoj;
							C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
							C_i*=mi8*tdwij/tdist;
							//C_i=mi8*(abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag))*tdwij/tdist;
							//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
							// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
							tmpx+=C_i*(xj-xi);
							tmpy+=C_i*(yj-yi);
						}
						if(k_fb_solve){
							if((ptypei==FLUID)&(ptypej!=FLUID)){
								Real twij=calc_kernel_wij(tmp_A,hi,tdist);
								Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
								tmpx+=fb_ij*(xi-xj);
								tmpy+=fb_ij*(yi-yj);
							}
						}
					}
				}
			}
		}
	}
	// y-directional gravitational force
	if(k_fg_solve) tmpy+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpy+=Gravitational_CONST*(alpha_T*(tempi-T_ref0)-alpha_S*(concni-S_ref0));

	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void coco_HB_viscous_force2D(int_t*g_str,int_t*g_end,part11*Pa11,part12*Pa12,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpn,tmpd;
	Real concni;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	hi=Pa11[i].h;
	tempi=Pa11[i].temp;
	pi=Pa11[i].pres;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	visi=Pa13[i].vis_t;
	concni=Pa12[i].concn;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	mi8=0.08/mi;
	mri=(mi/rhoi);

	betai=thermal_expansion(tempi,ptypei);

	tmpx=tmpy=0.0;
	tmpn=0.0;
	tmpd=1.0;

	for(int_t y=-1;y<=1;y++){
		for(int_t x=-1;x<=1;x++){
			int_t k=(icell+x)+k_NI*(jcell+y);
			if(k<0) continue;
			if(g_str[k]!=cu_memset){
				int_t fend=g_end[k];
				for(int_t j=g_str[k];j<fend;j++){
					Real xj,yj,tdist; //rr
					xj=Pa11[j].x;
					yj=Pa11[j].y;

					//rr=(xi-xj)*(xi-xj)+(yi-yj)*(yi-yj);tdist=sqrt(rr);
					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));

					if(tdist>0&&tdist<search_range){
						int_t ptypej;
						Real tdwx,tdwy,uxj,uyj,mj,tempj,rhoj,pj,hj;
						Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

						if(k_kgc_solve==1){
							tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}

						ptypej=Pa11[j].p_type;
						uxj=Pa11[j].ux;
						uyj=Pa11[j].uy;
						mj=Pa11[j].m;
						tempj=Pa11[j].temp;
						rhoj=Pa11[j].rho;
						pj=Pa11[j].pres;
						hj=Pa11[j].h;


						if(k_fp_solve){
							Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
							tmpx+=C_p*tdwx;
							tmpy+=C_p*tdwy;
						}
						if(k_fv_solve){
							Real visj,C_v;
							visj=Pa13[j].vis_t;
							//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
							//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj+1e-20))*((xi-xj)*tdwx+(yi-yj)*tdwy)/tdist/tdist;
							C_v=(xi-xj)*tdwx+(yi-yj)*tdwy;
							C_v*=(visi*visj)/(visi+visj+1e-20);
							C_v*=4*(mj/(rhoi*rhoj));
							C_v/=tdist*tdist;

							tmpx+=C_v*(uxi-uxj);
							tmpy+=C_v*(uyi-uyj);
						}
						if(k_fva_solve){
							Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj);
							if(uij_xij<0){
								Real h_ij,phi_ij,P_ij;
								h_ij=(hi+hj)*0.5;
								phi_ij=h_ij*uij_xij;
								phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
								P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
								P_ij/=(rhoi+rhoj)*0.5;
								//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
								tmpx+=-(P_ij)*tdwx;
								tmpy+=-(P_ij)*tdwy;
							}
						}

						if(k_interface_solve){
							int_t flag;
							Real mrj,C_i;
							flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
							mrj=mj/rhoj;
							C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
							C_i*=mi8*tdwij/tdist;
							//C_i=mi8*(abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag))*tdwij/tdist;
							//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
							// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
							tmpx+=C_i*(xj-xi);
							tmpy+=C_i*(yj-yi);
						}
						if(k_fb_solve){
							if((ptypei==FLUID)&(ptypej!=FLUID)){
								Real twij=calc_kernel_wij(tmp_A,hi,tdist);
								Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
								tmpx+=fb_ij*(xi-xj);
								tmpy+=fb_ij*(yi-yj);
							}
						}
					}
				}
			}
		}
	}
	// y-directional gravitational force
	if(k_fg_solve) tmpy+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpy+=Gravitational_CONST*(alpha_T*(tempi-T_ref0)-alpha_S*(concni-S_ref0));

	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void coco_viscous_force3D(int_t*g_str,int_t*g_end,part11*Pa11,part12*Pa12,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpz,tmpn,tmpd;
	Real concni;

	hi=Pa11[i].h;
	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	tempi=Pa11[i].temp;
	pi=Pa11[i].pres;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	concni=Pa12[i].concn;

	mi8=0.08/mi;
	mri=(mi/rhoi);

	visi=viscosity(tempi,ptypei);
	betai=thermal_expansion(tempi,ptypei);

	tmpx=tmpy=tmpz=0.0;
	tmpn=0.0;
	tmpd=1.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist; //rr
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

						if(tdist>0&&tdist<search_range){
							int_t ptypej;
							Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj;
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
							Real rdist=1.0/tdist;
							if(k_kgc_solve){
								tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)*rdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)*rdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)*rdist));
								tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)*rdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)*rdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)*rdist));
								tdwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)*rdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)*rdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)*rdist));
								//tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								//tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								//tdwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)*rdist;
								tdwy=tdwij*(yi-yj)*rdist;
								tdwz=tdwij*(zi-zj)*rdist;
								//tdwx=tdwij*(xi-xj)/tdist;
								//tdwy=tdwij*(yi-yj)/tdist;
								//tdwz=tdwij*(zi-zj)/tdist;
							}

							ptypej=Pa11[j].p_type;
							uxj=Pa11[j].ux;
							uyj=Pa11[j].uy;
							uzj=Pa11[j].uz;
							mj=Pa11[j].m;
							hj=Pa11[j].h;
							tempj=Pa11[j].temp;
							pj=Pa11[j].pres;
							rhoj=Pa11[j].rho;

							// pressure
							if(k_fp_solve)
							{
								Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
								tmpx+=C_p*tdwx;
								tmpy+=C_p*tdwy;
								tmpz+=C_p*tdwz;
							}
							// viscousity
							if(k_fv_solve)
							{
								Real visj,C_v;
								//
								visj=viscosity(tempj,ptypej);
								//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz;
								C_v*=(visi*visj)/(visi+visj+1e-20);
								C_v*=4*0.5875*(((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))/mi);
								C_v*=rdist*rdist;

								tmpx+=C_v*(uxi-uxj);
								tmpy+=C_v*(uyi-uyj);
								tmpz+=C_v*(uzi-uzj);
							}
							// artificial
							if(k_fva_solve)
							{
								Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
								if(uij_xij<0){
									Real h_ij,phi_ij,P_ij;
									//
									h_ij=(hi+hj)*0.5;
									phi_ij=h_ij*uij_xij;
									phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
									P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
									P_ij/=(rhoi+rhoj)*0.5;
									//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
									tmpx+=-(P_ij)*tdwx;
									tmpy+=-(P_ij)*tdwy;
									tmpz+=-(P_ij)*tdwz;
								}
							}
							// interface
							if(k_interface_solve)
							{
								int_t flag;
								Real mrj,C_i;
								//
								flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
								mrj=mj/rhoj;
								C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
								C_i*=mi8*tdwij/tdist;

								//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
								// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
								tmpx+=C_i*(xj-xi);
								tmpy+=C_i*(yj-yi);
								tmpz+=C_i*(zj-zi);
							}
							//boundary
							if(k_fb_solve)
							{
								if((ptypei==FLUID)&(ptypej!=FLUID)){
									Real twij=calc_kernel_wij(tmp_A,hi,tdist);
									Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
									tmpx+=fb_ij*(xi-xj);
									tmpy+=fb_ij*(yi-yj);
									tmpz+=fb_ij*(zi-zj);
								}
							}
							// End of FORCE Computation
						}
					}
				}
			}
		}
	}
	// z-directional gravitational force
	if(k_fg_solve) tmpz+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpz+=Gravitational_CONST*(alpha_T*(tempi-T_ref0)-alpha_S*(concni-S_ref0));

	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
	Pa11[i].ftotalz=tmpz;
}
////////////////////////////////////////////////////////////////////////
__global__ void coco_turbulence_viscous_force3D(int_t*g_str,int_t*g_end,part11*Pa11,part12*Pa12,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpz,tmpn,tmpd;
	Real concni;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	hi=Pa11[i].h;
	tempi=Pa11[i].temp;
	pi=Pa11[i].pres;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	concni=Pa12[i].concn;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	mi8=0.08/mi;
	mri=(mi/rhoi);

	visi=viscosity(tempi,ptypei)+Pa13[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	tmpx=tmpy=tmpz=0.0;
	tmpn=0.0;
	tmpd=1.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist; //rr
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

						if(tdist>0&&tdist<search_range){
							int_t ptypej;
							Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj;
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

							if(k_kgc_solve==1){
								tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								tdwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}

							ptypej=Pa11[j].p_type;
							uxj=Pa11[j].ux;
							uyj=Pa11[j].uy;
							uzj=Pa11[j].uz;
							mj=Pa11[j].m;
							tempj=Pa11[j].temp;
							rhoj=Pa11[j].rho;
							pj=Pa11[j].pres;
							hj=Pa11[j].h;


							if(k_fp_solve){
								Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
								tmpx+=C_p*tdwx;
								tmpy+=C_p*tdwy;
								tmpz+=C_p*tdwz;
							}
							if(k_fv_solve){
								Real visj,C_v;
								visj=viscosity(tempj,ptypej)+Pa13[j].vis_t;
								//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz;
								C_v*=(visi*visj)/(visi+visj+1e-20);
								C_v*=4*(mj/(rhoi*rhoj));
								C_v/=tdist*tdist;

								tmpx+=C_v*(uxi-uxj);
								tmpy+=C_v*(uyi-uyj);
								tmpz+=C_v*(uzi-uzj);
							}

							if(k_fva_solve){
								Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
								if(uij_xij<0){
									Real h_ij,phi_ij,P_ij;
									//
									h_ij=(hi+hj)*0.5;
									phi_ij=h_ij*uij_xij;
									phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
									P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
									P_ij/=(rhoi+rhoj)*0.5;
									//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
									tmpx+=-(P_ij)*tdwx;
									tmpy+=-(P_ij)*tdwy;
									tmpz+=-(P_ij)*tdwz;
								}
							}

							if(k_interface_solve){
								int_t flag;
								Real mrj,C_i;
								//
								flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
								mrj=mj/rhoj;
								C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
								C_i*=mi8*tdwij/tdist;

								//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
								// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
								tmpx+=C_i*(xj-xi);
								tmpy+=C_i*(yj-yi);
								tmpz+=C_i*(zj-zi);
							}
							if(k_fb_solve){
								if((ptypei==FLUID)&(ptypej!=FLUID)){
									Real twij=calc_kernel_wij(tmp_A,hi,tdist);
									Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));

									tmpx+=fb_ij*(xi-xj);
									tmpy+=fb_ij*(yi-yj);
									tmpz+=fb_ij*(zi-zj);
								}
							}
						}
					}
				}
			}
		}
	}
	// z-directional gravitational force
	if(k_fg_solve) tmpz+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpz+=Gravitational_CONST*(alpha_T*(tempi-T_ref0)-alpha_S*(concni-S_ref0));

	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
	Pa11[i].ftotalz=tmpz;
}
////////////////////////////////////////////////////////////////////////
__global__ void coco_SPS_viscous_force3D(int_t*g_str,int_t*g_end,part11*Pa11,part12*Pa12,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpz,tmpn,tmpd;
	Real concni;
	Real tau_xx_i,tau_xy_i,tau_xz_i,tau_yx_i,tau_yy_i,tau_yz_i,tau_zx_i,tau_zy_i,tau_zz_i;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	hi=Pa11[i].h;
	tempi=Pa11[i].temp;
	pi=Pa11[i].pres;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	concni=Pa12[i].concn;
	tau_xx_i=Pa13[i].Sxx;
	tau_xy_i=Pa13[i].Sxy;
	tau_xz_i=Pa13[i].Sxz;
	tau_yx_i=tau_xy_i;
	tau_yy_i=Pa13[i].Syy;
	tau_yz_i=Pa13[i].Syz;
	tau_zx_i=tau_xz_i;
	tau_zy_i=tau_yz_i;
	tau_zz_i=Pa13[i].Szz;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	mi8=0.08/mi;
	mri=(mi/rhoi);

	visi=viscosity(tempi,ptypei)+Pa13[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	tmpx=tmpy=tmpz=0.0;
	tmpn=0.0;
	tmpd=1.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist; //rr
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

						if(tdist>0&&tdist<search_range){
							int_t ptypej;
							Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj;
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

							if(k_kgc_solve==1){
								tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								tdwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}

							ptypej=Pa11[j].p_type;
							uxj=Pa11[j].ux;
							uyj=Pa11[j].uy;
							uzj=Pa11[j].uz;
							mj=Pa11[j].m;
							tempj=Pa11[j].temp;
							rhoj=Pa11[j].rho;
							pj=Pa11[j].pres;
							hj=Pa11[j].h;


							if(k_fp_solve){
								Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
								tmpx+=C_p*tdwx;
								tmpy+=C_p*tdwy;
								tmpz+=C_p*tdwz;
							}
							if(k_fv_solve){
								Real visj,C_v,C_vx,C_vy,C_vz;
								Real tau_xx_j,tau_xy_j,tau_xz_j,tau_yx_j,tau_yy_j,tau_yz_j,tau_zx_j,tau_zy_j,tau_zz_j;
								tau_xx_j=Pa13[j].Sxx;
								tau_xy_j=Pa13[j].Sxy;
								tau_xz_j=Pa13[j].Sxz;
								tau_yx_j=tau_xy_j;
								tau_yy_j=Pa13[j].Syy;
								tau_yz_j=Pa13[j].Syz;
								tau_zx_j=tau_xz_j;
								tau_zy_j=tau_yz_j;
								tau_zz_j=Pa13[j].Szz;

								visj=viscosity(tempj,ptypej);

								//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz;
								C_v*=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj));
								C_v/=tdist*tdist;

								C_vx=(tau_xx_i/(rhoi*rhoi)+tau_xx_j/(rhoj*rhoj)*tdwx);
								C_vx+=(tau_xy_i/(rhoi*rhoi)+tau_xy_j/(rhoj*rhoj)*tdwy);
								C_vx+=(tau_xz_i/(rhoi*rhoi)+tau_xz_j/(rhoj*rhoj)*tdwz);
								C_vx*=mj;
								C_vx+=C_v*(uxi-uxj);

								C_vy=(tau_yx_i/(rhoi*rhoi)+tau_yx_j/(rhoj*rhoj)*tdwx);
								C_vy+=(tau_yy_i/(rhoi*rhoi)+tau_yy_j/(rhoj*rhoj)*tdwy);
								C_vy+=(tau_yz_i/(rhoi*rhoi)+tau_yz_j/(rhoj*rhoj)*tdwz);
								C_vy*=mj;
								C_vy+=C_v*(uyi-uyj);

								C_vz=(tau_zx_i/(rhoi*rhoi)+tau_zx_j/(rhoj*rhoj)*tdwx);
								C_vz+=(tau_zy_i/(rhoi*rhoi)+tau_zy_j/(rhoj*rhoj)*tdwy);
								C_vz+=(tau_zz_i/(rhoi*rhoi)+tau_zz_j/(rhoj*rhoj)*tdwz);
								C_vz*=mj;
								C_vz+=C_v*(uzi-uzj);

								tmpx+=C_vx;
								tmpy+=C_vy;
								tmpz+=C_vz;
								//tmpx+=C_v*(uxi-uxj)+mj*((tau_xx_i/(rhoi*rhoi)+tau_xx_j/(rhoj*rhoj)*tdwx)+(tau_xy_i/(rhoi*rhoi)+tau_xy_j/(rhoj*rhoj)*tdwy)+(tau_xz_i/(rhoi*rhoi)+tau_xz_j/(rhoj*rhoj)*tdwz));
								//tmpy+=C_v*(uyi-uyj)+mj*((tau_yx_i/(rhoi*rhoi)+tau_yx_j/(rhoj*rhoj)*tdwx)+(tau_yy_i/(rhoi*rhoi)+tau_yy_j/(rhoj*rhoj)*tdwy)+(tau_yz_i/(rhoi*rhoi)+tau_yz_j/(rhoj*rhoj)*tdwz));
								//tmpz+=C_v*(uzi-uzj)+mj*((tau_zx_i/(rhoi*rhoi)+tau_zx_j/(rhoj*rhoj)*tdwx)+(tau_zy_i/(rhoi*rhoi)+tau_zy_j/(rhoj*rhoj)*tdwy)+(tau_zz_i/(rhoi*rhoi)+tau_zz_j/(rhoj*rhoj)*tdwz));
							}

							if(k_fva_solve){
								Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
								if(uij_xij<0){
									Real h_ij,phi_ij,P_ij;
									//
									h_ij=(hi+hj)*0.5;
									phi_ij=h_ij*uij_xij;
									phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
									P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
									P_ij/=(rhoi+rhoj)*0.5;
									//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
									tmpx+=-(P_ij)*tdwx;
									tmpy+=-(P_ij)*tdwy;
									tmpz+=-(P_ij)*tdwz;
								}
							}

							if(k_interface_solve){
								int_t flag;
								Real mrj,C_i;
								//
								flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
								mrj=mj/rhoj;
								C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
								C_i*=mi8*tdwij/tdist;

								//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
								// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
								tmpx+=C_i*(xj-xi);
								tmpy+=C_i*(yj-yi);
								tmpz+=C_i*(zj-zi);
							}
							if(k_fb_solve){
								if((ptypei==FLUID)&(ptypej!=FLUID)){
									Real twij=calc_kernel_wij(tmp_A,hi,tdist);
									Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));

									tmpx+=fb_ij*(xi-xj);
									tmpy+=fb_ij*(yi-yj);
									tmpz+=fb_ij*(zi-zj);
								}
							}
						}
					}
				}
			}
		}
	}
	// z-directional gravitational force
	if(k_fg_solve) tmpz+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpz+=Gravitational_CONST*(alpha_T*(tempi-T_ref0)-alpha_S*(concni-S_ref0));

	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
	Pa11[i].ftotalz=tmpz;
}
////////////////////////////////////////////////////////////////////////
__global__ void coco_HB_viscous_force3D(int_t*g_str,int_t*g_end,part11*Pa11,part12*Pa12,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_number_of_particles) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real search_range,tmp_A;
	Real tmpx,tmpy,tmpz,tmpn,tmpd;
	Real concni;

	ptypei=Pa11[i].p_type;
	icell=Pa11[i].I_cell;
	jcell=Pa11[i].J_cell;
	kcell=Pa11[i].K_cell;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	hi=Pa11[i].h;
	tempi=Pa11[i].temp;
	pi=Pa11[i].pres;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	concni=Pa12[i].concn;
	visi=Pa13[i].vis_t;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	mi8=0.08/mi;
	mri=(mi/rhoi);

	betai=thermal_expansion(tempi,ptypei);

	tmpx=tmpy=tmpz=0.0;
	tmpn=0.0;
	tmpd=1.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				if(k<0) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						Real xj,yj,zj,tdist; //rr
						xj=Pa11[j].x;
						yj=Pa11[j].y;
						zj=Pa11[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));

						if(tdist>0&&tdist<search_range){
							int_t ptypej;
							Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj;
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

							if(k_kgc_solve==1){
								tdwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								tdwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								tdwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}

							ptypej=Pa11[j].p_type;
							uxj=Pa11[j].ux;
							uyj=Pa11[j].uy;
							uzj=Pa11[j].uz;
							mj=Pa11[j].m;
							tempj=Pa11[j].temp;
							rhoj=Pa11[j].rho;
							pj=Pa11[j].pres;
							hj=Pa11[j].h;


							if(k_fp_solve){
								Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
								tmpx+=C_p*tdwx;
								tmpy+=C_p*tdwy;
								tmpz+=C_p*tdwz;
							}
							if(k_fv_solve){
								Real visj,C_v;
								visj=Pa13[j].vis_t;
								//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
								//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj+1e-20))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;

								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz;
								C_v*=(visi*visj)/(visi+visj+1e-20);
								C_v*=4*(mj/(rhoi*rhoj));
								C_v/=tdist*tdist;

								tmpx+=C_v*(uxi-uxj);
								tmpy+=C_v*(uyi-uyj);
								tmpz+=C_v*(uzi-uzj);
							}

							if(k_fva_solve){
								Real uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
								if(uij_xij<0){
									Real h_ij,phi_ij,P_ij;
									//
									h_ij=(hi+hj)*0.5;
									phi_ij=h_ij*uij_xij;
									phi_ij/=(tdist*tdist+0.01*h_ij*h_ij);
									P_ij=mi*phi_ij*(-Alpha*k_soundspeed+Beta*phi_ij);
									P_ij/=(rhoi+rhoj)*0.5;
									//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
									tmpx+=-(P_ij)*tdwx;
									tmpy+=-(P_ij)*tdwy;
									tmpz+=-(P_ij)*tdwz;
								}
							}

							if(k_interface_solve){
								int_t flag;
								Real mrj,C_i;
								//
								flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
								mrj=mj/rhoj;
								C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
								C_i*=mi8*tdwij/tdist;

								//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
								// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
								tmpx+=C_i*(xj-xi);
								tmpy+=C_i*(yj-yi);
								tmpz+=C_i*(zj-zi);
							}
							if(k_fb_solve){
								if((ptypei==FLUID)&(ptypej!=FLUID)){
									Real twij=calc_kernel_wij(tmp_A,hi,tdist);
									Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));

									tmpx+=fb_ij*(xi-xj);
									tmpy+=fb_ij*(yi-yj);
									tmpz+=fb_ij*(zi-zj);
								}
							}
						}
					}
				}
			}
		}
	}
	// z-directional gravitational force
	if(k_fg_solve) tmpz+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpz+=Gravitational_CONST*(alpha_T*(tempi-T_ref0)-alpha_S*(concni-S_ref0));

	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
	Pa11[i].ftotalz=tmpz;
}
////////////////////////////////////////////////////////////////////////
void calculate_force(int_t*vii,int_t*g_str,int_t*g_end,part11*Pa11,part12*Pa12,part13*Pa13){
	dim3 b,t;
	t.x=128;
	b.x=(number_of_particles-1)/t.x+1;

	if(dim==2){
		switch(turbulence_model){
			case Laminar:
				coco_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			case K_LM:
				coco_turbulence_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			case K_E:
				coco_turbulence_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			case SPS:
				coco_SPS_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			case HB:
				coco_HB_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			default:
				coco_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
		}
		if(fs_solve==1){
			// surface tension force calculation function (2017.04.20 jyb)
			KERNEL_clc_color_field2D<<<b,t>>>(g_str,g_end,Pa11);
			cudaDeviceSynchronize();
			KERNEL_clc_normal_gradient_c2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
			cudaDeviceSynchronize();
			KERNEL_clc_normal_gradient2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
			cudaDeviceSynchronize();
			KERNEL_add_surface_tension2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
			cudaDeviceSynchronize();
		}
	}else if(dim==3){
		switch(turbulence_model){
			case Laminar:
				coco_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			case K_LM:
				coco_turbulence_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			case K_E:
				coco_turbulence_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			case SPS:
				coco_SPS_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			case HB:
				coco_HB_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
			default:
				coco_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa12,Pa13);
				cudaDeviceSynchronize();
				break;
		}

		if(fs_solve==1){
			// surface tension force calculation function (2017.04.20 jyb)
			KERNEL_clc_color_field3D<<<b,t>>>(g_str,g_end,Pa11);
			cudaDeviceSynchronize();
			KERNEL_clc_normal_gradient_c3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
			cudaDeviceSynchronize();
			KERNEL_clc_normal_gradient3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
			cudaDeviceSynchronize();
			KERNEL_add_surface_tension3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
			cudaDeviceSynchronize();
		}
	}

	// sum up forces
	KERNEL_clc_ftotal<<<b,t>>>(Pa11);
	cudaDeviceSynchronize();

	/*
	if(fp_solve==1){
		// pressure force calculation function
		if(dim==2) KERNEL_clc_pressure_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		if(dim==3) KERNEL_clc_pressure_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		cudaDeviceSynchronize();
	}
	if(fv_solve==1){
		// viscous force calculation function
		switch(turbulence_model){
			case Laminar:
				if(dim==2) KERNEL_add_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				if(dim==3) KERNEL_add_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				cudaDeviceSynchronize();
				break;
			case K_LM:
				if(dim==2) KERNEL_add_turbulence_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				if(dim==3) KERNEL_add_turbulence_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				cudaDeviceSynchronize();
				break;
			case K_E:
				if(dim==2) KERNEL_add_turbulence_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				if(dim==3) KERNEL_add_turbulence_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				cudaDeviceSynchronize();
				break;
			case SPS:
				if(dim==2) KERNEL_add_SPS_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				if(dim==3) KERNEL_add_SPS_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				cudaDeviceSynchronize();
				break;
			case HB:
				if(dim==2) KERNEL_add_HB_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				if(dim==3) KERNEL_add_HB_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				cudaDeviceSynchronize();
				break;
			default:
				if(dim==2) KERNEL_add_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				if(dim==3) KERNEL_add_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
				cudaDeviceSynchronize();
				break;
		}
	}
	if(fva_solve==1){
		// artificial viscous force calculation function
		if(dim==2) KERNEL_add_artificial_viscous_force2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		if(dim==3) KERNEL_add_artificial_viscous_force3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		cudaDeviceSynchronize();
	}
	if(fg_solve==1){
		// gravitational force calculation function
		KERNEL_add_gravity_force<<<b,t>>>(Pa11);
		cudaDeviceSynchronize();
	}
	if(fs_solve==1){
		// surface tension force calculation function (2017.04.20 jyb)
		if(dim==2) KERNEL_clc_color_field2D<<<b,t>>>(g_str,g_end,Pa11);
		if(dim==3) KERNEL_clc_color_field3D<<<b,t>>>(g_str,g_end,Pa11);
		cudaDeviceSynchronize();
		if(dim==2) KERNEL_clc_normal_gradient_c2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		if(dim==3) KERNEL_clc_normal_gradient_c3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		cudaDeviceSynchronize();
		if(dim==2) KERNEL_clc_normal_gradient2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		if(dim==3) KERNEL_clc_normal_gradient3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		cudaDeviceSynchronize();
		if(dim==2) KERNEL_add_surface_tension2D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		if(dim==3) KERNEL_add_surface_tension3D<<<b,t>>>(g_str,g_end,Pa11,Pa13);
		cudaDeviceSynchronize();
	}
	if(interface_solve==1){
		// interface sharpness force calculation function (2017.05.02 jyb)
		if(dim==2) KERNEL_add_interface_sharpness2D<<<b,t>>>(g_str,g_end,Pa11);
		if(dim==3) KERNEL_add_interface_sharpness3D<<<b,t>>>(g_str,g_end,Pa11);
		cudaDeviceSynchronize();
	}
	if(fb_solve==1){
		// boundary force calculation function
		if(dim==2) KERNEL_add_boundary_force2D<<<b,t>>>(g_str,g_end,Pa11);
		if(dim==3) KERNEL_add_boundary_force3D<<<b,t>>>(g_str,g_end,Pa11);
		cudaDeviceSynchronize();
	}
	if(boussinesq_solve==1){
		// natural convection (boussinesq approximation) force calculation function
		if(dim==2) KERNEL_add_boussinesq_force2D<<<b,t>>>(g_str,g_end,Pa11);
		if(dim==3) KERNEL_add_boussinesq_force3D<<<b,t>>>(g_str,g_end,Pa11);
		cudaDeviceSynchronize();
	}
	// sum up forces
	KERNEL_clc_ftotal<<<b,t>>>(Pa11);
	cudaDeviceSynchronize();
	//*/
}

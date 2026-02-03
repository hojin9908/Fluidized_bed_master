/*  by Y.W.Shim
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_turb_viscosity(part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	if(P1[i].e_turb>0) P3[i].vis_t=C_mu*P1[i].rho*P1[i].k_turb*P1[i].k_turb/P1[i].e_turb;
	else P3[i].vis_t=0;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_HB_viscosity(part1*P1,part2*P2,part3*P3)
{
	// calculation of Herschel-Bulkley Viscosity (Visco-plastic)
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real SR=P2[i].SR;	//strain-rate
	Real vis_a;

	//check equation
	vis_a=TAU0_HB/SR+K0_HB*pow(SR,Real(N0_HB-1));
	vis_a=min(NU0_HB,vis_a);

	P3[i].vis_t=vis_a;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_update_turbulence(Real tdt,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t p_type_i=P1[i].p_type;

	P1[i].k_turb+=P3[i].dk_turb*tdt*(p_type_i>0);
	P1[i].e_turb+=P3[i].de_turb*tdt*(p_type_i>0);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_SPS_stress_tensor(part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real tvis_t;
	Real trho,th;
	Real tSxx,tSxy,tSxz,tSyy,tSyz,tSzz; //tSzx,tSzy,tSyx
	Real tau_xx,tau_xy,tau_xz,tau_yy,tau_yz,tau_zz;

	trho=P1[i].rho;
	th=P1[i].h*L_SPS;

	tvis_t=P3[i].vis_t;
	tSxx=P3[i].Sxx;
	tSxy=P3[i].Sxy;
	tSxz=P3[i].Sxz;
	tSyy=P3[i].Syy;
	tSyz=P3[i].Syz;
	tSzz=P3[i].Szz;

	// please check equations (by esk)!!!
	tau_xx=trho*(2*tvis_t*tSxx-2/3*(tSxx+tSyy+tSzz))-2/3*trho*CI_SPS*th*th;
	tau_xy=trho*(2*tvis_t*tSxy);
	tau_xz=trho*(2*tvis_t*tSxz);
	tau_yy=trho*(2*tvis_t*tSyy-2/3*(tSxx+tSyy+tSzz))-2/3*trho*CI_SPS*th*th;
	tau_yz=trho*(2*tvis_t*tSyz);
	tau_zz=trho*(2*tvis_t*tSzz-2/3*(tSxx+tSyy+tSzz))-2/3*trho*CI_SPS*th*th;
	P3[i].Sxx=tau_xx;
	P3[i].Sxy=tau_xy;
	P3[i].Sxz=tau_xz;
	P3[i].Syy=tau_yy;
	P3[i].Syz=tau_yz;
	P3[i].Szz=tau_zz;
}
//*/
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_strain_rate2D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	int_t icell,jcell;
	Real xi,yi;
	Real uxi,uyi;
	Real rhoi,Sa;
	Real tmpx,tmpy;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	rhoi=P1[i].rho;

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
						Real tdwx,tdwy,tdwij,uxj,uyj,mj,rhoj,uij2,tmpval;
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

						if(k_kgc_solve==1){
							// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						mj=P1[j].m;
						rhoj=P1[j].rho;

						uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj);
						tmpval=mj*(rhoi+rhoj)*uij2;
						tmpval/=(rhoi*rhoj*tdist*tdist);
						tmpx+=tmpval*(xi-xj)*tdwx;
						tmpy+=tmpval*(yi-yj)*tdwy;
					}
				}
			}
		}
	}
	Sa=-0.5*(tmpx+tmpy);
	Sa=max(1e-20,Sa);
	Sa=sqrt(Sa);
	P2[i].SR=Sa;
	//KERNEL_HB_viscosity
	if(k_turbulence_model==4){
		Real vis_a;
		//check equation
		vis_a=TAU0_HB/Sa+K0_HB*pow(Sa,Real(N0_HB-1));
		vis_a=min(NU0_HB,vis_a);
		P3[i].vis_t=vis_a;
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_strain_rate3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	Real rhoi,Sa;
	Real tmpx,tmpy,tmpz;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	rhoi=P1[i].rho;

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
							Real tdwx,tdwy,tdwz,tdwij;
							Real uxj,uyj,uzj,mj,rhoj,uij2,tmpval;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							if(k_kgc_solve==1){
								// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								// tdwz=((P3[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							rhoj=P1[j].rho;

							uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);
							tmpval=mj*(rhoi+rhoj)*uij2;
							tmpval/=(rhoi*rhoj*tdist*tdist);
							tmpx+=tmpval*(xi-xj)*tdwx;
							tmpy+=tmpval*(yi-yj)*tdwy;
							tmpz+=tmpval*(zi-zj)*tdwz;
						}
					}
				}
			}
		}
	}
	Sa=-0.5*(tmpx+tmpy+tmpz);
	Sa=max(1e-20,Sa);
	Sa=sqrt(Sa);
	P2[i].SR=Sa;
	//KERNEL_HB_viscosity
	if(k_turbulence_model==4){
		Real vis_a;
		//check equation
		vis_a=TAU0_HB/Sa+K0_HB*pow(Sa,Real(N0_HB-1));
		vis_a=min(NU0_HB,vis_a);
		P3[i].vis_t=vis_a;
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_klm_turb2D(int_t inout,Real tdt,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real visi,tempi;
	Real tPi,tPi2,tSR;
	Real vis_ti,k_turbi,e_turbi;
	Real tmpx,tmpy;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	tempi=P1[i].temp;
	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	tSR=P2[i].SR;
	vis_ti=P3[i].vis_t;
	k_turbi=P1[i].k_turb;
	e_turbi=P1[i].e_turb;
	visi=viscosity(tempi,ptypei);

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
						Real tdwx,tdwy,tdwij;
						int_t ptypej;
						Real mj,tempj,rhoj,visj,vis_tj,k_turbj,tmpval;
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

						if(k_kgc_solve==1){
							// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}
						ptypej=P1[j].p_type;
						mj=P1[j].m;
						tempj=P1[j].temp;
						rhoj=P1[j].rho;
						vis_tj=P3[j].vis_t;
						k_turbj=P1[j].k_turb;
						visj=viscosity(tempj,ptypej);
						if(e_turbi==0){tPi=0;}
						else{
							tPi=C_mu*k_turbi*k_turbi/(e_turbi)*tSR;
							tPi2=0.3*k_turbi*sqrt(tSR);
							tPi=min(tPi,tPi2);
						}
						tmpval=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k);
						tmpval*=(k_turbi-k_turbj)/(tdist*tdist+0.0000001);
						tmpx+=tmpval*(xi-xj)*tdwx;
						tmpy+=tmpval*(yi-yj)*tdwy;
					}
				}
			}
		}
	}
	Real dk_turbi=tPi-e_turbi+(tmpx+tmpy);
	Real ne_turbi=pow(C_mu,0.75)*pow(k_turbi,Real(1.5))/Lm;		// (by esk)xxx => fast math
	//P1[i].e_turb=ne_turbi;
	P3[i].dk_turb=dk_turbi;

	//KERNEL_turb_viscosity ----------------------------------------
	if(ne_turbi>0) P3[i].vis_t=C_mu*P1[i].rho*k_turbi*k_turbi/ne_turbi;
	else P3[i].vis_t=0;
	//--------------------------------------------------------------
	//KERNEL_update_turbulence--------------------------------------
	P1[i].k_turb=k_turbi+dk_turbi*tdt*(ptypei>0);
	P1[i].e_turb=ne_turbi+P3[i].de_turb*tdt*(ptypei>0);
	//--------------------------------------------------------------
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_klm_turb3D(int_t inout,Real tdt,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real visi,tempi;
	Real tPi,tPi2,tSR;
	Real vis_ti,k_turbi,e_turbi;
	Real tmpx,tmpy,tmpz;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	tempi=P1[i].temp;
	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	tSR=P2[i].SR;
	vis_ti=P3[i].vis_t;
	k_turbi=P1[i].k_turb;
	e_turbi=P1[i].e_turb;
	visi=viscosity(tempi,ptypei);

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
							Real tdwx,tdwy,tdwz,tdwij;
							int_t ptypej;
							Real mj,tempj,rhoj,visj,vis_tj,k_turbj,tmpval;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							if(k_kgc_solve==1){
								// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								// tdwz=((P3[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}
							ptypej=P1[j].p_type;
							mj=P1[j].m;
							tempj=P1[j].temp;
							rhoj=P1[j].rho;
							vis_tj=P3[j].vis_t;
							k_turbj=P1[j].k_turb;
							visj=viscosity(tempj,ptypej);
							if(e_turbi==0){tPi=0;}
							else{
								tPi=C_mu*k_turbi*k_turbi/(e_turbi)*tSR;
								tPi2=0.3*k_turbi*sqrt(tSR);
								tPi=min(tPi,tPi2);
							}
							tmpval=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k);
							tmpval*=(k_turbi-k_turbj)/(tdist*tdist+0.0000001);
							tmpx+=tmpval*(xi-xj)*tdwx;
							tmpy+=tmpval*(yi-yj)*tdwy;
							tmpz+=tmpval*(zi-zj)*tdwz;
						}
					}
				}
			}
		}
	}
	Real dk_turbi=tPi-e_turbi+(tmpx+tmpy+tmpz);
	Real ne_turbi=pow(C_mu,0.75)*pow(k_turbi,Real(1.5))/Lm;		// (by esk)xxx => fast math
	//P1[i].e_turb=ne_turbi;
	P3[i].dk_turb=dk_turbi;

	//KERNEL_turb_viscosity ----------------------------------------
	if(ne_turbi>0) P3[i].vis_t=C_mu*P1[i].rho*k_turbi*k_turbi/ne_turbi;
	else P3[i].vis_t=0;
	//--------------------------------------------------------------
	//KERNEL_update_turbulence--------------------------------------
	P1[i].k_turb=k_turbi+dk_turbi*tdt*(ptypei>0);
	P1[i].e_turb=ne_turbi+P3[i].de_turb*tdt*(ptypei>0);
	//--------------------------------------------------------------
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_ke_turb2D(int_t inout,Real tdt,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real visi,tempi;
	Real tPi,tPi2,tSR;
	Real vis_ti,k_turbi,e_turbi;
	Real search_range,tmp_h,tmp_A;
	Real tmpk,tmpe;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	tempi=P1[i].temp;
	tSR=P2[i].SR;
	vis_ti=P3[i].vis_t;
	k_turbi=P1[i].k_turb;
	e_turbi=P1[i].e_turb;
	visi=viscosity(tempi,ptypei);

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpk=tmpe=0.0;
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
						Real tdwx,tdwy,tdwij;
						int_t ptypej;
						Real mj,tempj,rhoj,vis_tj,k_turbj,e_turbj,visj,tmprr,tmpkk,tmpee;
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						if(k_kgc_solve==1){
							// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}
						ptypej=P1[j].p_type;
						mj=P1[j].m;
						tempj=P1[j].temp;
						rhoj=P1[j].rho;
						vis_tj=P3[j].vis_t;
						k_turbj=P1[j].k_turb;
						e_turbj=P1[j].e_turb;
						visj=viscosity(tempj,ptypej);
						if(e_turbi==0){tPi=0;}
						else{
							tPi=C_mu*k_turbi*k_turbi/(e_turbi)*tSR;
							tPi2=0.3*k_turbi*sqrt(tSR);
							tPi=min(tPi,tPi2);
						}
						tmprr=1.0/(tdist*tdist+0.0000001)*((xi-xj)*tdwx+(yi-yj)*tdwy);
						tmpkk=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k);
						tmpkk*=(k_turbi-k_turbj);
						tmpee=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_e);
						tmpee*=(e_turbi-e_turbj);
						tmpk+=tmpkk*tmprr;
						tmpe+=tmpee*tmprr;
					}
				}
			}
		}
	}
	Real dk_turbi=tPi-e_turbi+tmpk;
	Real de_turbi=e_turbi/(k_turbi+1e-10)*(C_e1*tPi-C_e2*e_turbi)+tmpe;

	//KERNEL_turb_viscosity ------------------------------------------------
	if(e_turbi>0) P3[i].vis_t=C_mu*P1[i].rho*k_turbi*k_turbi/e_turbi;
	else P3[i].vis_t=0;
	//----------------------------------------------------------------------
	P3[i].dk_turb=dk_turbi;
	P3[i].de_turb=de_turbi;
	//KERNEL_update_turbulence----------------------------------------------
	P1[i].k_turb=k_turbi+dk_turbi*tdt*(ptypei>0);
	P1[i].e_turb=e_turbi+de_turbi*tdt*(ptypei>0);
	//----------------------------------------------------------------------
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_ke_turb3D(int_t inout,Real tdt,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real visi,tempi;
	Real tPi,tPi2,tSR;
	Real vis_ti,k_turbi,e_turbi;
	Real search_range,tmp_h,tmp_A;
	Real tmpk,tmpe;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	tempi=P1[i].temp;
	tSR=P2[i].SR;
	vis_ti=P3[i].vis_t;
	k_turbi=P1[i].k_turb;
	e_turbi=P1[i].e_turb;
	visi=viscosity(tempi,ptypei);

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpk=tmpe=0.0;
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
							Real tdwx,tdwy,tdwz,tdwij;
							int_t ptypej;
							Real mj,tempj,rhoj,vis_tj,k_turbj,e_turbj,visj,tmprr,tmpkk,tmpee;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							if(k_kgc_solve==1){
								// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								// tdwz=((P3[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}
							ptypej=P1[j].p_type;
							mj=P1[j].m;
							tempj=P1[j].temp;
							rhoj=P1[j].rho;
							vis_tj=P3[j].vis_t;
							k_turbj=P1[j].k_turb;
							e_turbj=P1[j].e_turb;
							visj=viscosity(tempj,ptypej);

							if(e_turbi==0){
								tPi=0;
							}else{
								tPi=C_mu*k_turbi*k_turbi/(e_turbi)*tSR;
								tPi2=0.3*k_turbi*sqrt(tSR);
								tPi=min(tPi,tPi2);
							}
							tmprr=1.0/(tdist*tdist+0.0000001)*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz);
							tmpkk=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k);
							tmpkk*=(k_turbi-k_turbj);
							tmpee=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_e);
							tmpee*=(e_turbi-e_turbj);
							tmpk+=tmpkk*tmprr;
							tmpe+=tmpee*tmprr;
						}
					}
				}
			}
		}
	}
	Real dk_turbi=tPi-e_turbi+tmpk;
	Real de_turbi=e_turbi/(k_turbi+1e-10)*(C_e1*tPi-C_e2*e_turbi)+tmpe;

	//KERNEL_turb_viscosity ------------------------------------------------
	if(e_turbi>0) P3[i].vis_t=C_mu*P1[i].rho*k_turbi*k_turbi/e_turbi;
	else P3[i].vis_t=0;
	//----------------------------------------------------------------------
	P3[i].dk_turb=dk_turbi;
	P3[i].de_turb=de_turbi;
	//KERNEL_update_turbulence----------------------------------------------
	P1[i].k_turb=k_turbi+dk_turbi*tdt*(ptypei>0);
	P1[i].e_turb=e_turbi+de_turbi*tdt*(ptypei>0);
	//----------------------------------------------------------------------
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_SPS_strain_tensor2D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	int_t icell,jcell;
	Real xi,yi;
	Real S,th;
	Real search_range,tmp_h,tmp_A;
	Real tmpxx,tmpxy,tmpyy;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	xi=P1[i].x;
	yi=P1[i].y;
	th=tmp_h*L_SPS;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpxx=tmpxy=tmpyy=0.0;
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
						Real tdwx,tdwy,tdwij;
						Real uxj,uyj,mj,rhoj;
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						if(k_kgc_solve==1){
							// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						mj=P1[j].m;
						rhoj=P1[j].rho;

						tmpxx+=-(mj/rhoj)*(uxj)*tdwx;
						tmpxy+=-0.5*(mj/rhoj)*(uxj*tdwy+uyj*tdwx);
						tmpyy+=-(mj/rhoj)*(uyj)*tdwy;
					}
				}
			}
		}
	}
	//KERNEL_clc_SPS_stress_tensor
	S=sqrt((2*tmpxx*tmpxx+4*tmpxy*tmpxy+2*tmpyy*tmpyy));

	Real tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*S;
	Real trho=P1[i].rho;

	P3[i].vis_t=tvis_t;

	// please check equations (by esk)!!!
	Real tau_xx,tau_xy,tau_yy;
	tau_xx=trho*(2*tvis_t*tmpxx-2/3*(tmpxx+tmpyy))-2/3*trho*CI_SPS*th*th;
	tau_xy=trho*(2*tvis_t*tmpxy);
	tau_yy=trho*(2*tvis_t*tmpyy-2/3*(tmpxx+tmpyy))-2/3*trho*CI_SPS*th*th;

	P3[i].Sxx=tau_xx;
	P3[i].Sxy=tau_xy;
	P3[i].Syy=tau_yy;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_SPS_strain_tensor3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real S,th;
	Real tmpxx,tmpxy,tmpxz,tmpyy,tmpyz,tmpzz;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	th=tmp_h*L_SPS;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpxx=tmpxy=tmpxz=tmpyy=tmpyz=tmpzz=0.0;
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
							Real tdwx,tdwy,tdwz,tdwij;
							Real uxj,uyj,uzj,mj,rhoj;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							if(k_kgc_solve==1){
								// tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								// tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								// tdwz=((P3[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							rhoj=P1[j].rho;

							tmpxx+=-(mj/rhoj)*(uxj)*tdwx;
							tmpxy+=-0.5*(mj/rhoj)*(uxj*tdwy+uyj*tdwx);
							tmpxz+=-0.5*(mj/rhoj)*(uxj*tdwz+uzj*tdwx);
							tmpyy+=-(mj/rhoj)*(uyj)*tdwy;
							tmpyz+=-0.5*(mj/rhoj)*(uyj*tdwz+uzj*tdwy);
							tmpzz+=-(mj/rhoj)*(uzj)*tdwz;
						}
					}
				}
			}
		}
	}
	//KERNEL_clc_SPS_stress_tensor
	S=sqrt((2*tmpxx*tmpxx+4*tmpxy*tmpxy+4*tmpxz*tmpxz+2*tmpyy*tmpyy+4*tmpyz*tmpyz+2*tmpzz*tmpzz));

	Real tvis_t=(Cs_SPS*th)*(Cs_SPS*th)*S;
	Real trho=P1[i].rho;

	P3[i].vis_t=tvis_t;
	// please check equations (by esk)!!!
	Real tau_xx,tau_xy,tau_xz,tau_yy,tau_yz,tau_zz;
	tau_xx=trho*(2*tvis_t*tmpxx-2/3*(tmpxx+tmpyy+tmpzz))-2/3*trho*CI_SPS*th*th;
	tau_xy=trho*(2*tvis_t*tmpxy);
	tau_xz=trho*(2*tvis_t*tmpxz);
	tau_yy=trho*(2*tvis_t*tmpyy-2/3*(tmpxx+tmpyy+tmpzz))-2/3*trho*CI_SPS*th*th;
	tau_yz=trho*(2*tvis_t*tmpyz);
	tau_zz=trho*(2*tvis_t*tmpzz-2/3*(tmpxx+tmpyy+tmpzz))-2/3*trho*CI_SPS*th*th;

	P3[i].Sxx=tau_xx;
	P3[i].Sxy=tau_xy;
	P3[i].Sxz=tau_xz;
	P3[i].Syy=tau_yy;
	P3[i].Syz=tau_yz;
	P3[i].Szz=tau_zz;
}
/*  by Y.W.Shim
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_SPS_viscous_force2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real uxi,uyi;
	Real rhoi,visi,tempi;
	Real tau_xx_i,tau_xy_i,tau_yx_i,tau_yy_i;
	Real search_range,tmp_h,tmp_A;
	Real tmpx,tmpy;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;

	ptypei=P1[i].p_type;
	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	tempi=P1[i].temp;
	rhoi=P1[i].rho;
	tau_xx_i=P3[i].Sxx;
	tau_xy_i=P3[i].Sxy;
	tau_yx_i=tau_xy_i;
	tau_yy_i=P3[i].Syy;
	visi=viscosity(tempi,ptypei);

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
						Real tdwx,tdwy,tdwij;
						int_t ptypej;
						Real uxj,uyj,mj,tempj,rhoj,visj,C_v,txx,tyy;
						Real tau_xx_j,tau_xy_j,tau_yx_j,tau_yy_j;
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

						if(k_kgc_solve==1){
							tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}
						mj=P1[j].m;
						rhoj=P1[j].rho;
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						tempj=P1[j].temp;
						ptypej=P1[j].p_type;
						tau_xx_j=P3[j].Sxx;
						tau_xy_j=P3[j].Sxy;
						tau_yx_j=tau_xy_j;
						tau_yy_j=P3[j].Syy;

						visj=viscosity(tempj,ptypej);

						//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy)/tdist/tdist;
						C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj));
						C_v*=((xi-xj)*tdwx+(yi-yj)*tdwy)/tdist/tdist;

						txx=(tau_xx_i/(rhoi*rhoi)+tau_xx_j/(rhoj*rhoj)*tdwx);
						txx+=(tau_xy_i/(rhoi*rhoi)+tau_xy_j/(rhoj*rhoj)*tdwy);
						txx*=mj;
						txx+=C_v*(uxi-uxj);
						tyy=(tau_yx_i/(rhoi*rhoi)+tau_yx_j/(rhoj*rhoj)*tdwx);
						tyy+=(tau_yy_i/(rhoi*rhoi)+tau_yy_j/(rhoj*rhoj)*tdwy);
						tyy*=mj;
						tyy+=C_v*(uyi-uyj);

						tmpx+=txx;
						tmpy+=tyy;
					}
				}
			}
		}
	}
	P3[i].ftotalx+=tmpx;
	P3[i].ftotaly+=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_SPS_viscous_force3D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	Real rhoi,visi,tempi;
	Real tau_xx_i,tau_xy_i,tau_xz_i,tau_yx_i,tau_yy_i,tau_yz_i,tau_zx_i,tau_zy_i,tau_zz_i;
	Real tmpx,tmpy,tmpz;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	tempi=P1[i].temp;
	rhoi=P1[i].rho;
	tau_xx_i=P3[i].Sxx;
	tau_xy_i=P3[i].Sxy;
	tau_xz_i=P3[i].Sxz;
	tau_yx_i=tau_xy_i;
	tau_yy_i=P3[i].Syy;
	tau_yz_i=P3[i].Syz;
	tau_zx_i=tau_xz_i;
	tau_zy_i=tau_yz_i;
	tau_zz_i=P3[i].Szz;
	visi=viscosity(tempi,ptypei);

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
							Real tdwx,tdwy,tdwz,tdwij;
							int_t ptypej;
							Real uxj,uyj,uzj,mj,tempj,rhoj,visj,C_v,tmpxx,tmpyy,tmpzz;
							Real tau_xx_j,tau_xy_j,tau_xz_j,tau_yx_j,tau_yy_j,tau_yz_j,tau_zx_j,tau_zy_j,tau_zz_j;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);

							if(k_kgc_solve==1){
								tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								tdwz=((P3[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}
							ptypej=P1[j].p_type;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							tempj=P1[j].temp;
							rhoj=P1[j].rho;
							tau_xx_j=P3[j].Sxx;
							tau_xy_j=P3[j].Sxy;
							tau_xz_j=P3[j].Sxz;
							tau_yx_j=tau_xy_j;
							tau_yy_j=P3[j].Syy;
							tau_yz_j=P3[j].Syz;
							tau_zx_j=tau_xz_j;
							tau_zy_j=tau_yz_j;
							tau_zz_j=P3[j].Szz;

							visj=viscosity(tempj,ptypej);

							C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj));
							C_v*=((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;

							tmpxx=(tau_xx_i/(rhoi*rhoi)+tau_xx_j/(rhoj*rhoj)*tdwx);
							tmpxx+=(tau_xy_i/(rhoi*rhoi)+tau_xy_j/(rhoj*rhoj)*tdwy);
							tmpxx+=(tau_xz_i/(rhoi*rhoi)+tau_xz_j/(rhoj*rhoj)*tdwz);
							tmpxx*=mj;
							tmpxx+=C_v*(uxi-uxj);

							tmpyy=(tau_yx_i/(rhoi*rhoi)+tau_yx_j/(rhoj*rhoj)*tdwx);
							tmpyy+=(tau_yy_i/(rhoi*rhoi)+tau_yy_j/(rhoj*rhoj)*tdwy);
							tmpyy+=(tau_yz_i/(rhoi*rhoi)+tau_yz_j/(rhoj*rhoj)*tdwz);
							tmpyy*=mj;
							tmpyy+=C_v*(uyi-uyj);

							tmpzz=(tau_zx_i/(rhoi*rhoi)+tau_zx_j/(rhoj*rhoj)*tdwx);
							tmpzz=(tau_zy_i/(rhoi*rhoi)+tau_zy_j/(rhoj*rhoj)*tdwy);
							tmpzz=(tau_zz_i/(rhoi*rhoi)+tau_zz_j/(rhoj*rhoj)*tdwz);
							tmpzz*=mj;
							tmpzz+=C_v*(uzi-uzj);

							tmpx+=tmpxx;
							tmpy+=tmpyy;
							tmpz+=tmpzz;
						}
					}
				}
			}
		}
	}
	P3[i].ftotalx+=tmpx;
	P3[i].ftotaly+=tmpy;
	P3[i].ftotalz+=tmpz;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_turbulence_viscous_force2D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi;
	Real uxi,uyi;
	Real rhoi,visi,tempi;
	Real tmpx,tmpy;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	tempi=P1[i].temp;
	rhoi=P1[i].rho;
	visi=viscosity(tempi,ptypei)+P3[i].vis_t;

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
						Real tdwx,tdwy,tdwij;
						int_t ptypej;
						Real uxj,uyj,mj,tempj,rhoj,visj,C_v;
						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
						if(k_kgc_solve==1){
							tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist));
							tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist));
						}else{
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
						}
						ptypej=P1[j].p_type;
						uxj=P1[j].ux;
						uyj=P1[j].uy;
						mj=P1[j].m;
						tempj=P1[j].temp;
						rhoj=P1[j].rho;

						visj=viscosity(tempj,ptypej)+P3[j].vis_t;
						//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy)/tdist/tdist;
						C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj));
						C_v*=((xi-xj)*tdwx+(yi-yj)*tdwy)/tdist/tdist;

						tmpx+=C_v*(uxi-uxj);
						tmpy+=C_v*(uyi-uyj);
					}
				}
			}
		}
	}
	P3[i].ftotalx+=tmpx;
	P3[i].ftotaly+=tmpy;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_turbulence_viscous_force3D(int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	Real rhoi,visi,tempi;
	Real tmpx,tmpy,tmpz;
	Real search_range,tmp_h,tmp_A;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;	// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	rhoi=P1[i].rho;
	tempi=P1[i].temp;
	visi=viscosity(tempi,ptypei)+P3[i].vis_t;

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
							Real tdwx,tdwy,tdwz,tdwij;
							int_t ptypej;
							Real uxj,uyj,uzj,mj,tempj,rhoj,visj,C_v;
							tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
							if(k_kgc_solve==1){
								tdwx=((P3[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
								tdwy=((P3[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
								tdwz=((P3[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(P3[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(P3[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
							}else{
								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;
							}
							ptypej=P1[j].p_type;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							tempj=P1[j].temp;
							rhoj=P1[j].rho;

							visj=viscosity(tempj,ptypej)+P3[j].vis_t;
							//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
							C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj));
							C_v*=((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;

							tmpx+=C_v*(uxi-uxj);
							tmpy+=C_v*(uyi-uyj);
							tmpz+=C_v*(uzi-uzj);
						}
					}
				}
			}
		}
	}
	P3[i].ftotalx+=tmpx;
	P3[i].ftotaly+=tmpy;
	P3[i].ftotalz+=tmpz;
}
//*/

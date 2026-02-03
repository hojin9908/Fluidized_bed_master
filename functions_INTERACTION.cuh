__device__ Real calc_kernel_wij_ipf(Real tH,Real rr){

	Real tR,wij_ipf,tA;
	tR=wij_ipf=0.0;
	tA=1.0;
	Real eps;
	eps=tH/3.5;

	// if(k_IPF_kernel_type==Cosine){
	// 	if(k_kappa==1){
	// 	tR=rr/tH;
	// 	wij_ipf=-(tR<1)*tA*cos(3*PI/2*tR);}
	// 	else	if(k_kappa==2){
		tR=rr/tH;
		wij_ipf=-(tR<2)*tA*cos(3*PI/4*tR);
	// }else	if(k_IPF_kernel_type==Gauss){
	// 	tR=rr/tH;
	// 	wij_ipf=(tR<1)*tA*exp(-pow(rr,2)/2/eps/eps);
	// }else	if(k_IPF_kernel_type==Modified_Gaussian){
	// 	tR=rr/tH;
	// 	wij_ipf=(tR<1)*tA*rr*exp(-pow(rr,2)/2/eps/eps);
	// }else	if(k_IPF_kernel_type==Cubic){
	// 	tR=rr/tH;
	// 	if(0<=tR&&tR<1/3)  wij_ipf=(tR<1)*tA*((3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)-6*(2-3*tR)*(2-3*tR)*(2-3*tR)*(2-3*tR)*(2-3*tR)+15*(1-3*tR)*(1-3*tR)*(1-3*tR)*(1-3*tR)*(1-3*tR));
	// 	if(1/3<=tR&&tR<2/3)  wij_ipf=(tR<1)*tA*((3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)-6*(2-3*tR)*(2-3*tR)*(2-3*tR)*(2-3*tR)*(2-3*tR));
	// 	if(2/3<=tR&&tR<1)  wij_ipf=(tR<1)*tA*((3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR)*(3-3*tR));
	// }else	if(k_IPF_kernel_type==Wend2){
	// 	tR=rr/tH*0.5;
	// 	wij_ipf=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR);
	// }
	return wij_ipf;
}

__device__ Real calc_kernel_wij_half(Real tH,Real rr){

	Real tR,wij_half,tA;
	tR=wij_half=0.0;
	tA=1.0;
	Real eps0;
	eps0=tH/3.5*0.4;

	// if(k_IPF_kernel_type==Cosine){
		tR=rr/tH;
		wij_half=0.0;
	// }else	if(k_IPF_kernel_type==Gauss){
	// 	tR=rr/tH;
	// 	wij_half=(tR<1)*tA*exp(-pow(rr,2)/2/eps0/eps0);
	// }else	if(k_IPF_kernel_type==Modified_Gaussian){
	// 	tR=rr/tH;
	// 	wij_half=(tR<1)*tA*rr*exp(-pow(rr,2)/2/eps0/eps0);
	// }else	if(k_IPF_kernel_type==Cubic){
	// 	tR=rr/tH;
	// 	if(0<=(tR*2)&&(tR*2)<1/3)  wij_half=((tR*2)<1)*tA*((3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))-6*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))+15*(1-3*(tR*2))*(1-3*(tR*2))*(1-3*(tR*2))*(1-3*(tR*2))*(1-3*(tR*2)));
	// 	if(1/3<=(tR*2)&&(tR*2)<2/3)  wij_half=((tR*2)<1)*tA*((3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))-6*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2))*(2-3*(tR*2)));
	// 	if(2/3<=(tR*2)&&(tR*2)<1)  wij_half=((tR*2)<1)*tA*((3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2))*(3-3*(tR*2)));
	// }else	if(k_IPF_kernel_type==Wend2){
	// 	tR=rr/tH*0.5;
	// 	wij_half=((tR*2)<1)*tA*(1-(tR*2))*(1-(tR*2))*(1-(tR*2))*(1-(tR*2))*(1+4*(tR*2));
	// }
	return wij_half;
}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_interaction2D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;

	int_t ptypei;
	int_t icell,jcell;
	Real xi,yi,uxi,uyi,kci,cpi,eta;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real diffi,concni;
	Real pori;
	Real nxi,nyi,nmagi,sigmai;			// for surface tension
	Real nx_ci,ny_ci,nmag_ci,curvi; 	// for surface tension
	Real search_range,tmp_A,tmp_Rc,tmp_Rd;
	Real tmpx,tmpy,tmpn,tmpd;
	Real tmppx, tmppy, tmppz;//yhs	for pressure force
	Real tmpsx, tmpsy;
	Real tmp_fsn, tmp_fsd;
	Real virial1, virial2;

	Real tmp_pgf_x,tmp_pgf_y;		// pressure gradient force term (for SPH-DEM Coupling)



	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	hi=P1[i].h;
	tempi=P1[i].temp;
	pi=P1[i].pres;
	mi=P1[i].m;
	rhoi=P1[i].rho;
	pori=P1[i].DEMpor;


	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range
	mi8=0.08/mi;
	mri=(mi/rhoi);

	visi=viscosity(tempi,ptypei)+P3[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

	if((k_fs_solve)&(k_surf_model==2)){

		nxi=P3[i].nx;
		nyi=P3[i].ny;
		nmagi=P3[i].nmag;
		nx_ci=P3[i].nx_c;
		ny_ci=P3[i].ny_c;
		nmag_ci=P3[i].nmag_c;

		sigmai=sigma(tempi,ptypei);
	}

	if(k_con_solve){
		kci=conductivity(tempi,ptypei);
		cpi=heat_capacity(tempi,ptypei);
		eta=0.001*hi;
	}

	if(k_concn_solve){
		concni=P1[i].concn;
		diffi=diffusion_coefficient(tempi,ptypei);
	}


	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;

	tmpx=tmpy=0.0;
	tmpn=0.0;
	tmpd=1.0;
	tmp_Rc=0.0;
	tmp_Rd=0.0;
	tmp_fsn=0.0;
	tmp_fsd=0.0;
	tmppx=tmppy=0.0;
	tmpsx=tmpsy=0.0;
	virial1=virial2=0.0;
	tmp_pgf_x=tmp_pgf_y=0.0;

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

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))+1e-20;
						if(tdist<search_range){
							int_t ptypej;
							Real tdwx,tdwy,uxj,uyj,mj,tempj,rhoj,pj,hj,kcj,sum_con_H, diffj, concnj,porj,tmprd;
							Real nx_cj,ny_cj,nmag_cj,Phi_s,tmpnt;	// for surface tension
							
							Real twij=calc_kernel_wij(tmp_A,hi,tdist);
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

							Real wij_ipf=calc_kernel_wij_ipf(hi,tdist);
							Real wij_half=calc_kernel_wij_half(hi,tdist);


							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;

							if(k_kgc_solve>0){
								apply_gradient_correction_2D(P3[i].Cm,twij,tdwx,tdwy,&tdwx,&tdwy);
							}

							ptypej=P1[j].p_type;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							mj=P1[j].m;
							tempj=P1[j].temp;
							rhoj=P1[j].rho;
							pj=P1[j].pres;
							hj=P1[j].h;
							porj=P1[j].DEMpor;

							if(k_fp_solve){
								Real C_p=-mj*(pi+pj)/(pori*rhoi*porj*rhoj);
								tmppx=C_p*tdwx;
								tmppy=C_p*tdwy;

								tmpx+=tmppx;
								tmpy+=tmppy;

								tmp_pgf_x+=rhoi*pori*tmppx;				// force per volume
								tmp_pgf_y+=rhoi*pori*tmppy;

								if((ptypei==1) & (ptypej==1))	virial1+=(tmppx*(xi-xj)+tmppy*(yi-yj))*mi; //virial
								if((ptypei==2) & (ptypej==2))	virial2+=(tmppx*(xi-xj)+tmppy*(yi-yj))*mi; //virial
							}
							if(k_fv_solve){
								Real visj,C_v;
								visj=viscosity(tempj,ptypej)+P3[j].vis_t;
								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy;
								C_v*=4*(mj/(rhoi*rhoj));
								C_v*=(visi*visj)/(visi+visj);
								C_v/=tdist;
								C_v/=tdist;	

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
									P_ij/=(rhoi+rhoj);
									P_ij*=0.5;
									//P_ij=mi*(-Alpha*k_soundspeed*phi_ij+Beta*phi_ij*phi_ij)/(rhoi+rhoj)*0.5;
									tmpx+=-(P_ij)*tdwx;
									tmpy+=-(P_ij)*tdwy;
								}
							}
							if(k_interface_solve){
								int_t flag;
								Real mrj,C_i;
								//flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej);
								flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypei!=ptypej);
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
									Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
									tmpx+=fb_ij*(xi-xj);
									tmpy+=fb_ij*(yi-yj);
								}
							}
							if(k_fs_solve){
								if(k_surf_model==2){
									Real Cs;

									nx_cj=P3[j].nx_c;
									ny_cj=P3[j].ny_c;	

									nmag_cj=P3[j].nmag_c;
									Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);

									tmpnt=((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi);
									tmpnt+=((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi);

									tmpnt*=k_dim*(mj/rhoj)*tdwij/tdist;
									tmp_fsn+=tmpnt;
									tmp_fsd+=(mj/rhoj)*tdist*abs(tdwij);

									if (((ptypei == 1) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 1))
									{
										Cs=s_f1f2 * (A_f1f2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_f1f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 0) & (ptypej == 1)) || (ptypei == 1) & (ptypej == 0))
									{
										Cs=s_sf1 * (A_sf1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_sf1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else
									{
										Cs=0.0;
									}
									tmpsx += Cs*(xi - xj);
									tmpsy += Cs*(yi - yj);
									tmpx+= Cs*(xi - xj);
									tmpy+= Cs*(yi - yj);
								}
								else	if(k_surf_model==1){
									Real Cs;
									if ((ptypei == 1) & (ptypej == 1))
									{
										Cs=s_ff1 * (A_ff1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_ff1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if ((ptypei == 2) & (ptypej == 2))
									{
										Cs=s_ff2 * (A_ff2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_ff2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 1) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 1))
									{
										Cs=s_f1f2 * (A_f1f2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_f1f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 0) & (ptypej == 1)) || (ptypei == 1) & (ptypej == 0))
									{
										Cs=s_sf1 * (A_sf1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_sf1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 0) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 0))
									{
										Cs=s_sf2 * (A_sf2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_sf2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 9) & (ptypej == 1)) || (ptypei == 1) & (ptypej == 9))
									{
										Cs=s_s2f1 * (A_s2f1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_s2f1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 9) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 9))
									{
										Cs=s_s2f2 * (A_s2f2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_s2f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else
									{
										Cs=0.0;
									}
									Real tmpssx, tmpssy;
									tmpssx=Cs*(xi - xj);
									tmpssy=Cs*(yi - yj);
									tmpsx+= tmpssx;
									tmpsy+= tmpssy;
									tmpx+= tmpssx;
									tmpy+= tmpssy;
									if((ptypei==1) & (ptypej==1))	virial1+=(tmpssx*(xi-xj)+tmpssy*(yi-yj))*mi; //virial
									if((ptypei==2) & (ptypej==2))	virial2+=(tmpssx*(xi-xj)+tmpssy*(yi-yj))*mi; //virial
								}
							}
							if(k_boussinesq_solve){
								if((ptypei!=BOUNDARY)&(ptypei!=MOVING)){
									tmpn+=mj*(tempj-tempi)*twij*(ptypei==ptypej)/rhoj;
									tmpd+=mj*twij*(ptypei==ptypej)/rhoj;
								}
							}
							if(k_con_solve){
								kcj=conductivity(tempj, ptypej);
								sum_con_H=4.0*mj*kcj*kci*(tempi-tempj)*tdwij;
								sum_con_H/=mi*cpi*(tdist+eta*eta)*rhoi*rhoj*(kci+kcj);		//denthalpy 아니고 dtemp로 바꿈 (08.29)
								tmp_Rc+=sum_con_H;
							}
							if(k_concn_solve){
								concnj=P1[j].concn;
								diffj=diffusion_coefficient(tempj,ptypej);
								tmprd=mi*(diffi*rhoi+diffj*rhoj);
								tmprd*=((xi-xj)*tdwx+(yi-yj)*tdwy);
								tmprd/=(rhoi*rhoj*(tdist*tdist+0.01*hi*hi));
								tmprd*=(concni-concnj)*(ptypei==ptypej);

								tmp_Rd+=tmprd;
							}
						}
					}				
				}
			}
		}
	}
	// y-directional gravitational force
	if(k_fg_solve) tmpy+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpy+=-betai*Gravitational_CONST*(tmpn/tmpd);
	if(k_fs_solve){
		if((nmagi>0.1/hi)) curvi=tmp_fsn/tmp_fsd;
		else curvi=0;

		tmpx+=sigmai*curvi*nxi/rhoi;
		tmpy+=sigmai*curvi*nyi/rhoi;

		tmpsx += sigmai*curvi*nxi/rhoi;
		tmpsy += sigmai*curvi*nyi/rhoi;
	}

	P3[i].ftotalx=tmpx;
	P3[i].ftotaly=tmpy;

	P3[i].fsx=tmpsx;
	P3[i].fsy=tmpsy;

	P1[i].pgf_x=tmp_pgf_x;
	P1[i].pgf_y=tmp_pgf_y;

	if(ptypei==1)	P1[i].pres_ipp=virial1;
	if(ptypei==2)	P1[i].pres_ipp=virial2;

	P3[i].ftotal=sqrt(tmpx*tmpx+tmpy*tmpy);

	if(k_con_solve) P3[i].dtemp=tmp_Rc;
	if(k_concn_solve)	P3[i].dconcn=tmp_Rd;

}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_interaction3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi,kci,cpi,eta;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real diffi,concni;
	Real pori;
	Real nxi,nyi,nzi,nmagi,sigmai;			// for surface tension
	Real nx_ci,ny_ci,nz_ci,nmag_ci,curvi; 	// for surface tension
	Real search_range,tmp_A,tmp_Rc,tmp_Rd;
	Real tmpx,tmpy,tmpz,tmpn,tmpd;
	Real tmppx, tmppy, tmppz;
	Real tmpsx, tmpsy, tmpsz;
	Real tmp_fsn, tmp_fsd;
	Real virial1, virial2;
	Real tmp_pgf_x,tmp_pgf_y,tmp_pgf_z;							// pressure gradient force term (for SPH-DEM Coupling)
	Real eulerx,eulery,eulerz,eulert;

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	hi=P1[i].h;
	tempi=P1[i].temp;
	pi=P1[i].pres;
	mi=P1[i].m;
	rhoi=P1[i].rho;
	pori=P1[i].DEMpor;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	//if(abs(uzi>4.2)) P1[i].uz=0.42;

	if((k_fs_solve)&(k_surf_model==2)){

		nxi=P3[i].nx;
		nyi=P3[i].ny;
		nzi=P3[i].nz;

		nmagi=P3[i].nmag;
		nx_ci=P3[i].nx_c;
		ny_ci=P3[i].ny_c;
		nz_ci=P3[i].nz_c;

		nmag_ci=P3[i].nmag_c;

		sigmai=sigma(tempi,ptypei);
	}

	if(k_con_solve){
		eta=0.001*hi;
		kci=conductivity(tempi,ptypei);
		cpi=heat_capacity(tempi,ptypei);
	}
	if(k_concn_solve){
		concni=P1[i].concn;
		diffi=diffusion_coefficient(tempi,ptypei);
	}

	mi8=0.06/(mi); // .. interface force
	mri=mi/rhoi;

	visi=viscosity(tempi,ptypei)+P3[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);

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
	tmpn=0.0;
	tmpd=1.0;
	tmp_Rc=0.0;
	tmp_Rd=0.0;
	tmp_fsn=0.0;
	tmp_fsd=0.0;
	tmpsx=tmpsy=tmpsz=0.0;
	tmppx=tmppy=tmppz=0.0;
	virial1=virial2=0.0;
	tmp_pgf_x=tmp_pgf_y=tmp_pgf_z=0.0;
	eulerx=eulery=eulerz=eulert=0.0;

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

							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
							if(tdist<search_range){
								int_t ptypej;
								Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj,kcj,sum_con_H,diffj,concnj,porj,tmprd;
								Real nx_cj,ny_cj,nz_cj,nmag_cj,Phi_s,tmpnt;	// for surface tension
								Real volj;

								Real twij=calc_kernel_wij(tmp_A,hi,tdist);
								Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);

								Real wij_ipf=calc_kernel_wij_ipf(hi,tdist);
								Real wij_half=calc_kernel_wij_half(hi,tdist);

								tdwx=tdwij*(xi-xj)/tdist;
								tdwy=tdwij*(yi-yj)/tdist;
								tdwz=tdwij*(zi-zj)/tdist;


								if(k_kgc_solve>0){
									apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
								}

								ptypej=P1[j].p_type;
								uxj=P1[j].ux;
								uyj=P1[j].uy;
								uzj=P1[j].uz;
								mj=P1[j].m;
								tempj=P1[j].temp;
								rhoj=P1[j].rho;
								pj=P1[j].pres;
								hj=P1[j].h;
								porj=P1[j].DEMpor;
								volj=P1[j].vol;


								if(k_fp_solve){
									Real C_p=-pori*(mj)*(pi+pj)/(rhoi*rhoj);
									//Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
									//Real C_p_pgf=-mj*(pi+pj)/(porhoi*rhoj);
									
									tmppx=C_p*tdwx;
									tmppy=C_p*tdwy;
									tmppz=C_p*tdwz;

									tmpx+=tmppx;
									tmpy+=tmppy;
									tmpz+=tmppz;

									tmp_pgf_x+=rhoi/pori*C_p*tdwx;			// force per unit volume
									tmp_pgf_y+=rhoi/pori*C_p*tdwy;
									tmp_pgf_z+=rhoi/pori*C_p*tdwz;

									if((ptypei==1) & (ptypej==1))	virial1+=(tmppx*(xi-xj)+tmppy*(yi-yj)+tmppz*(zi-zj))*mi; //virial
									if((ptypei==2) & (ptypej==2))	virial2+=(tmppx*(xi-xj)+tmppy*(yi-yj)+tmppz*(zi-zj))*mi; //virial
								}
								if(k_fv_solve){
									Real visj,C_v;
									visj=viscosity(tempj,ptypej)+P3[j].vis_t;
									//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
									C_v=(xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz;
									C_v*=(visi*visj)/(visi+visj);
									C_v*=4*pori*(mj/(rhoi*rhoj));
									C_v/=tdist;
									C_v/=tdist;

									if(P1[i].p_type<=0)	C_v=0.0;
									//if(P1[j].p_type<=0)	C_v=0.0;

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
										P_ij/=(rhoi+rhoj);
										P_ij*=0.5;
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
									flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)&&(ptypei!=-3)&&(ptypej!=-3);
									//flag=(ptypei!=BOUNDARY)&&(ptypei!=MOVING)&&(ptypei!=ptypej);
									mrj=mj/(rhoj);
									C_i=abs(pi)*mri*mri+abs(pj)*mrj*mrj*(flag);
									C_i*=pori*mi8*tdwij/tdist;	

									//C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
									// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
									tmpx+=C_i*(xj-xi);
									tmpy+=C_i*(yj-yi);
									tmpz+=C_i*(zj-zi);
								}
								if(k_fb_solve){
									if((ptypei==FLUID)&(ptypej!=FLUID)){
										Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));	

										tmpx+=fb_ij*(xi-xj);
										tmpy+=fb_ij*(yi-yj);
										tmpz+=fb_ij*(zi-zj);
									}
								}
								if(k_fs_solve){
									if(k_surf_model==2){

										nx_cj=P3[j].nx_c;
										ny_cj=P3[j].ny_c;
										nz_cj=P3[j].nz_c;

										nmag_cj=P3[j].nmag_c;
										Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);

										tmpnt=((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi);
										tmpnt+=((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi);
										tmpnt+=((nz_ci/nmag_ci)-Phi_s*(nz_cj/nmag_cj))*(zj-zi);

										// tmpnt=((nx_ci)-Phi_s*(nx_cj))*(xj-xi);
										// tmpnt+=((ny_ci)-Phi_s*(ny_cj))*(yj-yi);
										// tmpnt+=((nz_ci)-Phi_s*(nz_cj))*(zj-zi);

										tmpnt*=k_dim*(mj/rhoj)*tdwij/tdist;
										tmp_fsn+=tmpnt;
										tmp_fsd+=(mj/rhoj)*tdist*abs(tdwij);

										Real Cs;

										if (((ptypei == 0) & (ptypej == 1)) || (ptypei == 1) & (ptypej == 0))
										{
											Cs=s_sf1 * (A_sf1 * wij_half/(mi*(tdist + 1.0e-10)));
											Cs-=s_sf1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
										}
										else if (((ptypei == 0) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 0))
										{
											Cs=s_sf2 * (A_sf2 * wij_half/(mi*(tdist + 1.0e-10)));
											Cs-=s_sf2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
										}
										else
										{
											Cs=0.0;
										}
										tmpsx += Cs*(xi - xj);
										tmpsy += Cs*(yi - yj);
										tmpsz += Cs*(zi - zj);

										tmpx+= Cs*(xi - xj);
										tmpy+= Cs*(yi - yj);
										tmpz+= Cs*(zi - zj);

									}else	if(k_surf_model==1){
										Real Cs;
										if ((ptypei == 1) & (ptypej == 1))
										{
											Cs=s_ff1 * (A_ff1 * wij_half/(mi*(tdist + 1.0e-10)));
											Cs-=s_ff1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
										}
										else if ((ptypei == 2) & (ptypej == 2))
										{
											Cs=s_ff2 * (A_ff2 * wij_half/(mi*(tdist + 1.0e-10)));
											Cs-=s_ff2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
										}
										else if (((ptypei == 1) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 1))
										{
											Cs=s_f1f2 * (A_f1f2 * wij_half/(mi*(tdist + 1.0e-10)));
											Cs-=s_f1f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
										}
										else if (((ptypei == 0) & (ptypej == 1)) || (ptypei == 1) & (ptypej == 0))
										{
											Cs=s_sf1 * (A_sf1 * wij_half/(mi*(tdist + 1.0e-10)));
											Cs-=s_sf1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
										}
										else if (((ptypei == 0) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 0))
										{
											Cs=s_sf2 * (A_sf2 * wij_half/(mi*(tdist + 1.0e-10)));
											Cs-=s_sf2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
										}
										else if (((ptypei == 9) & (ptypej == 1)) || (ptypei == 1) & (ptypej == 9))
										{
											Cs=s_s2f1 * (A_s2f1 * wij_half/(mi*(tdist + 1.0e-10)));
											Cs-=s_s2f1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
										}
										else if (((ptypei == 9) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 9))
										{
											Cs=s_s2f2 * (A_s2f2 * wij_half/(mi*(tdist + 1.0e-10)));
											Cs-=s_s2f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
										}
										else
										{
											Cs=0.0;
										}
										Real tmpssx, tmpssy, tmpssz;
										tmpssx=Cs*(xi - xj);
										tmpssy=Cs*(yi - yj);
										tmpssz=Cs*(zi - zj);
										tmpsx+= tmpssx;
										tmpsy+= tmpssy;
										tmpsz+= tmpssz;
										tmpx+= tmpssx;
										tmpy+= tmpssy;
										tmpz+= tmpssz;
										if((ptypei==1) & (ptypej==1))	virial1+=(tmpssx*(xi-xj)+tmpssy*(yi-yj)+tmpssz*(zi-zj))*mi; //virial
										if((ptypei==2) & (ptypej==2))	virial2+=(tmpssx*(xi-xj)+tmpssy*(yi-yj)+tmpssz*(zi-zj))*mi; //virial
									}
								}
								if(k_boussinesq_solve){
									if((ptypei!=BOUNDARY)&(ptypei!=MOVING)){
										tmpn+=mj*(tempj-tempi)*twij*(ptypei==ptypej)/rhoj;
										tmpd+=mj*twij*(ptypei==ptypej)/rhoj;
									}
								}
								if(k_con_solve){
									kcj=conductivity(tempj, ptypej);
									sum_con_H=4.0*mj*kcj*kci*(tempi-tempj)*tdwij;
									sum_con_H/=cpi*(tdist+eta*eta)*rhoi*rhoj*(kci+kcj);	// denthalpy에서 dtemp로 바꿈 (2023.08.25)
									tmp_Rc+=sum_con_H;
								}
								if(k_concn_solve){
									concnj=P1[j].concn;
									diffj=diffusion_coefficient(tempj,ptypej);
									tmprd=mi*(diffi*rhoi+diffj*rhoj);
									tmprd*=((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz);
									tmprd/=(rhoi*rhoj*(tdist*tdist+0.01*hi*hi));
									tmprd*=(concni-concnj)*(ptypei==ptypej);

									tmp_Rd+=tmprd;
								}
								if((P1[i].elix<1.0)||(P1[i].eliy<1.0)){
									eulerx += (uxi*(uxj-uxi)*tdwx+uyi*(uxj-uxi)*tdwy+uzi*(uxj-uxi)*tdwz)*mj/rhoj*(1-P1[i].elix);
									eulery += (uxi*(uyj-uyi)*tdwx+uyi*(uyj-uyi)*tdwy+uzi*(uyj-uyi)*tdwz)*mj/rhoj*(1-P1[i].eliy);
									eulerz += (uxi*(uzj-uzi)*tdwx+uyi*(uzj-uzi)*tdwy+uzi*(uzj-uzi)*tdwz)*mj/rhoj*(1-P1[i].eliz);
	
									if(k_con_solve){
										eulert += (uxi*(tempj-tempi)*tdwx+uyi*(tempj-tempi)*tdwy+uzi*(tempj-tempi)*tdwz)*mj/rhoj*(1-P1[i].elix);
									}
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
	if(k_boussinesq_solve) tmpz+=-betai*Gravitational_CONST*(tmpn/tmpd);
	if((k_fs_solve)&(k_surf_model==2)){
		if((nmagi>0.1/hi)&((tmp_fsn>0)&(ptypei==1)||(tmp_fsn<0)&(ptypei==2))) curvi=tmp_fsn/tmp_fsd;
		else curvi=0;

		tmpsx+=sigmai*curvi*nxi/rhoi;
		tmpsy+=sigmai*curvi*nyi/rhoi;
		tmpsz+=sigmai*curvi*nzi/rhoi;

		tmpx+=sigmai*curvi*nxi/rhoi;
		tmpy+=sigmai*curvi*nyi/rhoi;
		tmpz+=sigmai*curvi*nzi/rhoi;

		P3[i].curv=curvi;
	}

	P3[i].fsx=tmpsx;
	P3[i].fsy=tmpsy;
	P3[i].fsz=tmpsz;

	P3[i].ftotalx=tmpx-eulerx;
	P3[i].ftotaly=tmpy-eulery;
	P3[i].ftotalz=tmpz-eulerz;

	//P3[i].ftotalx=tmpx;
	//P3[i].ftotaly=tmpy;
	//P3[i].ftotalz=tmpz;

	P1[i].test2=tmpz;
	P1[i].test3=-eulerz;
	//P1[i].test3=0.0;
	
	P1[i].pgf_x=tmp_pgf_x;
	P1[i].pgf_y=tmp_pgf_y;
	P1[i].pgf_z=tmp_pgf_z;

	if(ptypei==1)	P1[i].pres_ipp=virial1;
	if(ptypei==2)	P1[i].pres_ipp=virial2;

	P3[i].ftotal=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);

	if(k_con_solve)	P3[i].dtemp=tmp_Rc*(ptypei!=-3)-eulert;
	if(k_concn_solve) P3[i].dconcn=tmp_Rd;

}

////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_interaction3D_1(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi,uxi,uyi,uzi,kci,eta,cpi;
	Real pi,hi,mi,mi8,mri,rhoi,tempi,visi,betai;
	Real diffi,concni;
	Real nxi,nyi,nzi,nmagi,sigmai;			// for surface tension
	Real nx_ci,ny_ci,nz_ci,nmag_ci,curvi; 	// for surface tension
	Real search_range,tmp_A,tmp_Rc,tmp_Rd;
	Real tmpx,tmpy,tmpz,tmpn,tmpd;
	Real tmp_fsn, tmp_fsd;
	Real virialx, virialy, virialz;
	Real eulerx, eulery, eulerz, eulert;

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	hi=P1[i].h;
	tempi=P1[i].temp;
	pi=P1[i].pres;
	mi=P1[i].m;
	rhoi=P1[i].rho;

	tmp_A=calc_tmpA(hi);
	search_range=k_search_kappa*hi;	// search range

	if(k_fs_solve){

		nxi=P3[i].nx;
		nyi=P3[i].ny;
		nzi=P3[i].nz;

		nmagi=P3[i].nmag;
		nx_ci=P3[i].nx_c;
		ny_ci=P3[i].ny_c;
		nz_ci=P3[i].nz_c;

		nmag_ci=P3[i].nmag_c;

		sigmai=sigma(tempi,ptypei);
	}

	if(k_con_solve){
		kci=conductivity(tempi,ptypei);
		cpi=300.0;
		eta=0.001*hi;
	}
	if(k_concn_solve){
		concni=P1[i].concn;
		diffi=diffusion_coefficient(tempi,ptypei);
	}

	mi8=0.08/mi; // .. interface force
	mri=(mi/rhoi);

	visi=viscosity(tempi,ptypei)+P3[i].vis_t;
	betai=thermal_expansion(tempi,ptypei);




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
	tmpn=0.0;
	tmpd=1.0;
	tmp_Rc=0.0;
	tmp_Rd=0.0;
	tmp_fsn=0.0;
	tmp_fsd=0.0;
	virialx=virialy=virialz=0.0;
	eulerx=eulery=eulerz=eulert=0.0;

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
						xj=P1[j].x;
						yj=P1[j].y;
						zj=P1[j].z;

						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))+1e-20;
						if(tdist<search_range){
							int_t ptypej;
							Real tdwx,tdwy,tdwz,uxj,uyj,uzj,mj,tempj,rhoj,pj,hj,kcj,sum_con_H,diffj,concnj,tmprd;
							Real nx_cj,ny_cj,nz_cj,nmag_cj,Phi_s,tmpnt;	// for surface tension

							Real twij=calc_kernel_wij(tmp_A,hi,tdist);
							Real tdwij=calc_kernel_dwij(tmp_A,hi,tdist);
							Real wij_ipf=calc_kernel_wij_ipf(hi,tdist);
							Real wij_half=calc_kernel_wij_half(hi,tdist);
	
							tdwx=tdwij*(xi-xj)/tdist;
							tdwy=tdwij*(yi-yj)/tdist;
							tdwz=tdwij*(zi-zj)/tdist;


							if(k_kgc_solve>0){
								apply_gradient_correction_3D(P3[i].Cm,twij,tdwx,tdwy,tdwz,&tdwx,&tdwy,&tdwz);
							}

							ptypej=P1[j].p_type;
							uxj=P1[j].ux;
							uyj=P1[j].uy;
							uzj=P1[j].uz;
							mj=P1[j].m;
							tempj=P1[j].temp;
							rhoj=P1[j].rho;
							pj=P1[j].pres;
							hj=P1[j].h;


							if(k_fp_solve){
								Real C_p=-mj*(pi+pj)/(rhoi*rhoj);
								tmpx+=C_p*tdwx;
								tmpy+=C_p*tdwy;
								tmpz+=C_p*tdwz;
																
								virialx+=mi*C_p*tdwx*(xi-xj);
								virialy+=mi*C_p*tdwy*(yi-yj);
								virialz+=mi*C_p*tdwz*(zi-zj);
							}
							if(k_fv_solve){
								Real visj,C_v;
								visj=viscosity(tempj,ptypej)+P3[j].vis_t;
								//C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
								C_v=(xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz;
								C_v*=(visi*visj)/(visi+visj);
								C_v*=4*(mj/(rhoi*rhoj));
								C_v/=tdist;
								C_v/=tdist;

								if(P1[i].p_type<=0)	C_v=0.0;
								if(P1[j].p_type<=0)	C_v=0.0;


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
									P_ij/=(rhoi+rhoj);
									P_ij*=0.5;
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
									Real fb_ij=k_c_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));

									tmpx+=fb_ij*(xi-xj);
									tmpy+=fb_ij*(yi-yj);
									tmpz+=fb_ij*(zi-zj);
								}
							}
							if(k_fs_solve){
								if(k_surf_model==2){

								nx_cj=P3[j].nx_c;
								ny_cj=P3[j].ny_c;
								nz_cj=P3[j].nz_c;

								nmag_cj=P3[j].nmag_c;
								Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);

								tmpnt=((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi);
								tmpnt+=((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi);
								tmpnt+=((nz_ci/nmag_ci)-Phi_s*(nz_cj/nmag_cj))*(zj-zi);

								tmpnt*=k_dim*(mj/rhoj)*tdwij/tdist;
								tmp_fsn+=tmpnt;
								tmp_fsd+=(mj/rhoj)*tdist*abs(tdwij);
							}else	if(k_surf_model==1){
									Real Cs;
									if ((ptypei == 1) & (ptypej == 1))
									{
										Cs=s_ff1 * (A_ff1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_ff1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if ((ptypei == 2) & (ptypej == 2))
									{
										Cs=s_ff2 * (A_ff2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_ff2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 1) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 1))
									{
										Cs=s_f1f2 * (A_f1f2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_f1f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 0) & (ptypej == 1)) || (ptypei == 1) & (ptypej == 0))
									{
										Cs=s_sf1 * (A_sf1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_sf1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == 0) & (ptypej == 2)) || (ptypei == 2) & (ptypej == 0))
									{
										Cs=s_sf2 * (A_sf2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_sf2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == -1) & (ptypej == 1)) || (ptypei == 1) & (ptypej == -1))
									{
										Cs=s_s2f1 * (A_s2f1 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_s2f1 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else if (((ptypei == -1) & (ptypej == 2)) || (ptypei == 2) & (ptypej == -1))
									{
										Cs=s_s2f2 * (A_s2f2 * wij_half/(mi*(tdist + 1.0e-10)));
										Cs-=s_s2f2 * (wij_ipf/(mi*(tdist + 1.0e-10)));
									}
									else
									{
										Cs=0.0;
									}
									tmpx+=Cs*(xi - xj);
									tmpy+=Cs*(yi - yj);
									tmpz+=Cs*(zi - zj);
									virialx+=mi*Cs*(xi - xj)*(xi-xj);
									virialy+=mi*Cs*(yi - yj)*(yi-yj);
									virialz+=mi*Cs*(zi - zj)*(zi-zj);
								}
							}
							if((P1[i].elix<1.0)||(P1[i].eliy<1.0)){
								eulerx += (uxi*(uxj-uxi)*tdwx+uyi*(uxj-uxi)*tdwy+uzi*(uxj-uxi)*tdwz)*mj/rhoj*(1-P1[i].elix);
								eulery += (uxi*(uyj-uyi)*tdwx+uyi*(uyj-uyi)*tdwy+uzi*(uyj-uyi)*tdwz)*mj/rhoj*(1-P1[i].eliy);
								eulerz += (uxi*(uzj-uzi)*tdwx+uyi*(uzj-uzi)*tdwy+uzi*(uzj-uzi)*tdwz)*mj/rhoj*(1-P1[i].eliz);

								if(k_con_solve){
									eulert += (uxi*(tempj-tempi)*tdwx+uyi*(tempj-tempi)*tdwy+uzi*(tempj-tempi)*tdwz)*mj/rhoj*(1-P1[i].elix);
								}
							}
							if(k_boussinesq_solve){
								if((ptypei!=BOUNDARY)&(ptypei!=MOVING)){
									tmpn+=mj*(tempj-tempi)*twij*(ptypei==ptypej)/rhoj;
									tmpd+=mj*twij*(ptypei==ptypej)/rhoj;
								}
							}
							if(k_concn_solve){
								concnj=P1[j].concn;
								diffj=diffusion_coefficient(tempj,ptypej);
								tmprd=mi*(diffi*rhoi+diffj*rhoj);
								tmprd*=((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz);
								tmprd/=(rhoi*rhoj*(tdist*tdist+0.01*hi*hi));
								tmprd*=(concni-concnj)*(ptypei==ptypej);

								tmp_Rd+=tmprd;
							}
						}
					}
				}
			}
		}
	}
	// z-directional gravitational force
	// if(k_fg_solve) tmpz+=-Gravitational_CONST;
	if(k_fg_solve) tmpz+=-Gravitational_CONST;
	if(k_boussinesq_solve) tmpz+=-betai*Gravitational_CONST*(tmpn/tmpd);
	if(k_fs_solve){
		if((nmagi>0.1/hi)&(tmp_fsn>0)) curvi=tmp_fsn/tmp_fsd;
		else curvi=0;

		tmpx+=sigmai*curvi*nxi/rhoi;
		tmpy+=sigmai*curvi*nyi/rhoi;
		tmpz+=sigmai*curvi*nzi/rhoi;
	}

	P3[i].ftotalx=tmpx-eulerx;
	P3[i].ftotaly=tmpy-eulery;
	P3[i].ftotalz=tmpz-eulerz;

	//if(k_con_solve){
	//	if (enthalpy_eqn) P3[i].denthalpy=tmp_Rc*(ptypei!=-1);
	//	else{
			P3[i].dtemp=tmp_Rc/cpi*(ptypei!=-1)-eulert;
	//	}
	//}

	//P1[i].virialx = virialx;
	//P1[i].virialy = virialy;
	//P1[i].virialz = virialz;

	P3[i].ftotal=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);

	if(k_concn_solve) P3[i].dconcn=tmp_Rd;
	if(abs(uzi<4.2)) P1[i].uz=0.42;
}

////////////////////////////////////////////////////////////////////////
void CSF_validation(part1*P1,part3*P3)
{
	char FileName_vtk[512];
		sprintf(FileName_vtk,"./validation/CSF_pressurevalid_%dstp.txt",count);
		// If the file already exists, its contents are discarded and create the new one.
		FILE*outFile_vtk;
		outFile_vtk=fopen(FileName_vtk,"w");

		Real xcm=0;
		Real ycm=0;
		Real zcm=0;
		int nop1=0;
		Real radius=0;
		int nop_edge=0;
		int nop=num_part2;
		Real space = P1[0].h/1.5;

		for(int i=0; i<num_part2; i++)
		{
			if(P1[i].p_type==1){
			nop1+=1;}
		}

		for(int i=0; i<nop; i++)
		{
			if(P1[i].p_type==1){
			xcm+=P1[i].x/nop1;
			ycm+=P1[i].y/nop1;
			zcm+=P1[i].z/nop1;}
		}

		for(int i=0; i<nop; i++)
		{
			if((P1[i].p_type==1)&(P3[i].lbl_surf==1)){
			radius+=sqrt((P1[i].x-xcm)*(P1[i].x-xcm)+(P1[i].y-ycm)*(P1[i].y-ycm)+(P1[i].z-zcm)*(P1[i].z-zcm));
			nop_edge+=1;}
		}

		radius/=nop_edge;
		// radius = 0.0015;

		//행렬 생성 및 초기화
		int_t N=30;
		Real basis = 0.1*radius;
		Real gap=(1.5*radius-basis)/N;

		Real avg_pres[N],simpleavg_pres[N];
		int number1[N],number2[N];
		for(int i=0; i<N; i++){
			avg_pres[i]=0;
			simpleavg_pres[i]=0;
			number1[i]=0;
			number2[i]=0;
		}

		Real dist=0.0;

		for(int j=0; j<N-1; j++)
		{
		for(int i=0; i<nop; i++)
		{
			dist=sqrt((P1[i].x-xcm)*(P1[i].x-xcm)+(P1[i].y-ycm)*(P1[i].y-ycm)+(P1[i].z-zcm)*(P1[i].z-zcm));

			Real d1 = basis+j*gap;
			Real d2 = basis+(j+1)*gap;

			if(dim==2)
			{
					if((dist>=d1)&&(dist<d2)){
						avg_pres[j]+=P1[i].pres*space*space/(PI*(d2*d2-d1*d1));
						simpleavg_pres[j]+=P1[i].pres;
						if(P1[i].p_type==1) number1[j]+=1;
						if(P1[i].p_type==2) number2[j]+=1;
					}
			}
			else if(dim==3)
			{
				if((dist>=d1)&&(dist<d2)){
					avg_pres[j]+=P1[i].pres*space*space*space/(4.0/3.0*PI*(d2*d2*d2-d1*d1*d1));
					simpleavg_pres[j]+=P1[i].pres;
					if(P1[i].p_type==1) number1[j]+=1;
					if(P1[i].p_type==2) number2[j]+=1;
				}
			}
		}
		simpleavg_pres[j]=simpleavg_pres[j]/(number1[j]+number2[j]);
	}

	fprintf(outFile_vtk,"CSF validation\n\n");
	fprintf(outFile_vtk, "number of fluid\t%d\n",nop1);
	fprintf(outFile_vtk, "radius of droplet\t%f\n",radius);
	fprintf(outFile_vtk, "laplace_pressure (theory)\t%f\n",(1+(dim==3))*0.073/radius);
	fprintf(outFile_vtk, "laplace_pressure (simulation)\t%f\t%f\n",avg_pres[0]-avg_pres[N-2],simpleavg_pres[0]-simpleavg_pres[N-2]);
	fprintf(outFile_vtk, "number\tradius\tnop1\tnop2\tavg_pres\tsimpleavg_pres\t\n");

	for (int i=0; i<N-1; i++){
		fprintf(outFile_vtk, "%d\t%f\t%d\t%d\t%f\t%f\t\n",(i+1), (basis+(2.0*i+1.0)/2.0*gap),number1[i],number2[i], avg_pres[i], simpleavg_pres[i]);
	}

	fclose(outFile_vtk);
}

////////////////////////////////////////////////////////////////////////
void IPF_validation(part1*P1,part3*P3)
{
	char FileName_vtk[512];
		sprintf(FileName_vtk,"./validation/IPF_pressurevalid_%dstp.txt",count);
		// If the file already exists, its contents are discarded and create the new one.
		FILE*outFile_vtk;
		outFile_vtk=fopen(FileName_vtk,"w");

		Real xcm=0;
		Real ycm=0;
		Real zcm=0;
		int nop1=0;
		Real radius=0;
		int nop_edge=0;
		int nop=num_part2;
		Real space = P1[0].h/1.5;

		for(int i=0; i<num_part2; i++)
		{
			if(P1[i].p_type==1){
			nop1+=1;}
		}

		for(int i=0; i<nop; i++)
		{
			if(P1[i].p_type==1){
			xcm+=P1[i].x/nop1;
			ycm+=P1[i].y/nop1;
			zcm+=P1[i].z/nop1;}
		}

		for(int i=0; i<nop; i++)
		{
			if((P1[i].p_type==1)&(P3[i].lbl_surf==1)){
			radius+=sqrt((P1[i].x-xcm)*(P1[i].x-xcm)+(P1[i].y-ycm)*(P1[i].y-ycm)+(P1[i].z-zcm)*(P1[i].z-zcm));
			nop_edge+=1;}
		}

		radius/=nop_edge;
		// radius = 0.0015;

		//행렬 생성 및 초기화
		int_t N=40;
		Real basis = 0.1*radius;
		Real gap=(1.5*radius-basis)/N;

		Real virial_pres[N];
		int number1[N],number2[N];
		for(int i=0; i<N; i++){
			virial_pres[i]=0;
			number1[i]=0;
			number2[i]=0;
		}

		Real dist=0.0;

		for(int j=0; j<N; j++)
		{
		for(int i=0; i<nop; i++)
		{
			dist=sqrt((P1[i].x-xcm)*(P1[i].x-xcm)+(P1[i].y-ycm)*(P1[i].y-ycm)+(P1[i].z-zcm)*(P1[i].z-zcm));

			Real d1 = basis+j*gap;
			// Real d2 = basis+(j+1)*gap;

			// PHASE 1
			if(P1[i].p_type==1){
			if(dim==2)
			{
					if(dist<=d1){
						virial_pres[j]+=P1[i].pres_ipp/(2*dim*PI*d1*d1);
						number1[j]+=1;
					}
			}
			else if(dim==3)
			{
				if(dist<=d1){
					virial_pres[j]+=P1[i].pres_ipp/(2*dim*4/3*PI*d1*d1*d1);
					number1[j]+=1;
				}
			}
			}
		}
	}

	fprintf(outFile_vtk,"IPF validation\n\n");
	fprintf(outFile_vtk, "number of fluid\t%d\n",nop1);
	fprintf(outFile_vtk, "radius of droplet\t%f\n",radius);
	fprintf(outFile_vtk, "laplace_pressure (theory)\t%f\n",(1+(dim==3))*0.073/radius);
	fprintf(outFile_vtk, "laplace_pressure (simulation)\t%f\n",virial_pres[0]);
	fprintf(outFile_vtk, "number\tradius\tnop1\tnop2\tvirial_pres\t\n");

	for (int i=0; i<N; i++){
		fprintf(outFile_vtk, "%d\t%f\t%d\t%d\t%f\t\n",(i+1), (basis+i*gap),number1[i],number2[i], virial_pres[i]);
	}

	fclose(outFile_vtk);
}

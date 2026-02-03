//////////////////////////////////////////////////////////////////////
__global__ void KERNEL_DEM_SINKED_VOLUME3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;		// leave if it is not DEM particle

	
	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;			
	Real radi,voli, hi, sp;	
	Real tmp_A;
	
	
	//Real search_range;
	Real tmp_vsph;			// tmptq : temporary value for torque calculation
	Real search_range;



	// position
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	radi=P1[i].rad;


	// radius & diameter
	 radi=P1[i].rad;
	 sp=radi*0.2;
	 voli=(4.0/3.0)*3.14159*(radi+sp*2.0)*(radi+sp*2.0)*(radi+sp*2.0);

	//search_range=k_search_kappa*hi;	// search range
	//hi=P1[i].h;
	//tmp_A=calc_tmpA(hi);
	//search_range=k_search_kappa*hi;	// search range

	//voli=(4.0/3.0)*3.14159*radi*radi*radi;



	//search_range=DEM_radius;	

	search_range=radi+sp;	// search range
	
	
	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;


	tmp_vsph=0.0;

	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;

				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						

						if (P1[j].p_type<=1000){
							Real xj,yj,zj,tdist; 
						
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
							
			
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist<search_range){

								Real mj,rhoj,dem_porj;
								//Real twij;

								//twij=calc_kernel_wij(tmp_A,hi,tdist);
									
								mj=P1[j].m;
								rhoj=P1[j].rho;					
								dem_porj=1-P1[j].DEMpor;

								tmp_vsph+=mj/rhoj;





							}
								
	
						}


					}			
				}
			}
		}
	}
	


	// Save Sinked Volume Fraction of DEM Particles
	P1[i].DEMvf=tmp_vsph/voli;
	//P1[i].DEMvf=1.0;

}

////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_DEM_coupling3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;

	if((i>=k_num_part2) || (P1[i].i_type!=inout) || (P1[i].p_type<=1000)) return;
	// if(i>=k_num_part2) return;
	// if(P1[i].i_type!=inout) return;
	// if(P1[i].p_type<=1000) return;		// leave if it is not DEM particle

	
	int_t ptypei;
	int_t icell,jcell,kcell;
	int_t demidxi;
	Real xi,yi,zi,uxi,uyi,uzi;	
	Real tempi, tempj,ki,alph_i,cpi;
	Real mi,radi,di,voli,rhoi,hi;	
	Real flt_air,flt_fluid;											
	Real ftotalx,ftotaly,ftotalz;
	Real vfi;
	
	Real ufxf, ufyf, ufzf, uxijf,uyijf,uzijf,mag_uijf;
	Real ufxa, ufya, ufza, uxija,uyija,uzija,mag_uija;
	Real ufxfa, ufyfa, ufzfa, uxijfa,uyijfa,uzijfa,mag_uijfa;
	Real ufx_visf, ufy_visf, ufz_visf;
	Real ufx_visa, ufy_visa, ufz_visa;
	Real Ref,Cdf,betaf,pori;
	Real Rea,Cda,betaa;
	Real Refa,Cdfa,betafa;
	Real vis_kif,vis_k_invf,rho_f;
	Real vis_kia,vis_k_inva,rho_a;
	Real vis_kifa,rho_fa,vis_fa;
	Real dfxf,dfyf,dfzf;
	Real dfxa,dfya,dfza;

	Real temp_f, temp_a, temp_fa, temp_ijf, temp_ija, temp_ijfa;
	Real dtempi;
	Real Pr_f, Pr_a, Pr_fa, a_ht;
	Real dq_vol;

	Real Nu_f, Nu_a, Nu_fa;
	Real h_conv_f, h_conv_a, h_conv_fa;

	
	
	//Real search_range;
	Real tmpx,tmpy,tmpz;			// tmptq : temporary value for torque calculation
	Real tmp_ufxf,tmp_ufyf,tmp_ufzf,tmp_por;
	Real tmp_ufxa,tmp_ufya,tmp_ufza;
	Real tmp_temp_f, tmp_temp_a;
	Real f_fluid,f_air;
	Real search_range,tmp_A;


	ptypei = P1[i].p_type;

	// position
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	// velocity
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	

	// radius & diameter
	radi=P1[i].rad;
	di=2*radi;
	voli=(4.0/3.0)*3.14159*radi*radi*radi;

	// mass,density
	mi=P1[i].m;
	rhoi=P1[i].rho;

	// temperature
	tempi = P1[i].temp;

	demidxi=P1[i].dem_idx;

	if(radi<0.0012){
		ki=conductivity_1(tempi,ptypei,1);
		alph_i=diffusivity_1(tempi,ptypei,1);
		cpi=(ki)/(alph_i*rhoi);

	}
	else{
		ki=conductivity_1(tempi,ptypei,0);
		alph_i=diffusivity_1(tempi,ptypei,0);
		cpi=(ki)/(alph_i*rhoi);
	}

	// filter (SPH particles around DEM particle i)
	flt_air=P1[i].flt_s;
	flt_fluid=P1[i].flt_sd;

	//if (flt_air<0.01) return;
	//if (flt_fluid<0.01) return;

	// ftotal
	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	// dtemp
	dtempi = P3[i].dtemp;
	
	





	//search_range=k_search_kappa*hi;	// search range
	hi=1*P1[i].h;
	tmp_A=calc_tmpA(1.0*hi);
	search_range=k_search_kappa*1.0*hi;	// search range


	// Sinked Volume Fraction of DEM Particle
	vfi=P1[i].DEMvf;





	
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
	tmp_ufxf=tmp_ufyf=tmp_ufzf=0.0;
	tmp_ufxa=tmp_ufya=tmp_ufza=0.0;
	tmp_por=0.0;
	tmp_temp_f = tmp_temp_a = 0.0;


	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;

				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){


						
						

						

						if ((P1[j].p_type<=1000) && (P1[j].DEMpor<1.0)   ){
							Real xj,yj,zj,tdist; 
						
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
							
			
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist<search_range){

								Real twij;
								Real uxj,uyj,uzj;
								Real mj,porj,rhoj,rho_refj;
								Real pgf_xj,pgf_yj,pgf_zj;

								Real tempj,visj;
								int_t ptypej;

								twij=calc_kernel_wij(tmp_A,1.0*hi,tdist);
								

								uxj=P1[j].ux;
								uyj=P1[j].uy;
								uzj=P1[j].uz;



								mj=P1[j].m;
								porj=P1[j].DEMpor;
								rhoj=P1[j].rho;
								tempj = P1[j].temp;
								rho_refj=P2[j].rho_ref;
								

								pgf_xj=P1[j].pgf_x;
								pgf_yj=P1[j].pgf_y;
								pgf_zj=P1[j].pgf_z;

								ptypej=P1[j].p_type;
								tempj=P1[j].temp;
								visj=viscosity(tempj,ptypej);

								




								if (ptypej==1){

									if(flt_fluid>0.000001){

										vis_kif=VISCOSITY_AIR/DENSITY_AIR;
										rho_f=DENSITY_AIR;
										vis_k_invf=rho_f/visj;
	
										tmp_ufxf+=uxj*mj*twij/rhoj;
										tmp_ufyf+=uyj*mj*twij/rhoj;
										tmp_ufzf+=uzj*mj*twij/rhoj;
	
										tmp_por+=porj*mj*twij/rhoj;

										tmp_temp_f+=tempj* mj * twij / rhoj;

									}



								}

								if (ptypej==3){

									if(flt_air>0.000001){
										vis_kia=VISCOSITY_AIR/DENSITY_AIR;
										rho_a=DENSITY_AIR;
										vis_k_inva=rho_a/visj;
	
										tmp_ufxa+=uxj*mj*twij/rhoj;
										tmp_ufya+=uyj*mj*twij/rhoj;
										tmp_ufza+=uzj*mj*twij/rhoj;
	
										tmp_por+=porj*mj*twij/rhoj;

										tmp_temp_a += tempj * mj * twij / rhoj;



									}


								}
								
								if ((ptypej!=0) && (ptypej!=MOVING) && (ptypej!=-3)&& (ptypej!=3)){
								//if ((ptypej!=0) | (P1[j].flt_s>0.8)){++

									if(flt_fluid>0.01){

										tmpx+=pgf_xj*(mj/rhoj)*twij*voli/(mi*flt_fluid+1.0e-15);		// pressure gradient force
										tmpy+=pgf_yj*(mj/rhoj)*twij*voli/(mi*flt_fluid+1.0e-15);
										tmpz+=pgf_zj*(mj/rhoj)*twij*voli/(mi*flt_fluid+1.0e-15);


									}


									
								

								}


							}
								
	
						}


					}			
				}
			}
		}
	}

	pori=tmp_por/(P1[i].flt_sd+1.0e-15);
	//pori=P1[i].DEMpor;
	
	ufxf=tmp_ufxf/(flt_fluid+1.0e-15);	// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
	ufyf=tmp_ufyf/(flt_fluid+1.0e-15);
	ufzf=tmp_ufzf/(flt_fluid+1.0e-15);
	temp_f=tmp_temp_f/ (flt_fluid + 1.0e-15);

	ufxa=tmp_ufxa/(flt_air+1.0e-15);	// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
	ufya=tmp_ufya/(flt_air+1.0e-15);
	ufza=tmp_ufza/(flt_air+1.0e-15);
	temp_a = tmp_temp_a / (flt_air + 1.0e-15);

	//P1[i].test2=temp_a;
	

	ufxfa=(tmp_ufxf+tmp_ufxa)/(flt_fluid+flt_air+1.0e-15);	// �ֺ� ��ü�� ��ü+��ü
	ufyfa=(tmp_ufyf+tmp_ufya)/(flt_fluid+flt_air+1.0e-15);
	ufzfa=(tmp_ufzf+tmp_ufza)/(flt_fluid+flt_air+1.0e-15);
	temp_fa = (tmp_temp_f + tmp_temp_a) / (flt_fluid + flt_air + 1.0e-15);

	

	uxijf=ufxf-uxi;		// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
	uyijf=ufyf-uyi;
	uzijf=ufzf-uzi;
	temp_ijf = temp_f - tempi;

	uxija=ufxa-uxi;		// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
	uyija=ufya-uyi;
	uzija=ufza-uzi;
	temp_ija = temp_a - tempi;

	//P1[i].test3=uzija;
	//P1[i].test4=temp_ija;

	uxijfa=ufxfa-uxi;	// �ֺ� ��ü�� ��ü+��ü
	uyijfa=ufyfa-uyi;
	uzijfa=ufzfa-uzi;
	temp_ijfa = temp_fa - tempi;

	f_fluid=flt_fluid/(flt_fluid+flt_air+1.0e-15);
	f_air=flt_air/(flt_fluid+flt_air+1.0e-15);


	mag_uijf=sqrt(uxijf*uxijf+uyijf*uyijf+uzijf*uzijf);
	mag_uija=sqrt(uxija*uxija+uyija*uyija+uzija*uzija);
	mag_uijfa=sqrt(uxijfa*uxijfa+uyijfa*uyijfa+uzijfa*uzijfa);

	P1[i].test6=mag_uijf;

	dfxf=0.0;
	dfyf=0.0;
	dfzf=0.0;

	dfxa=0.0;
	dfya=0.0;
	dfza=0.0;

	//pori=tmp_por/(flt_fluid+flt_air+1.0e-15);	//original
	
	
	//pori=0.38;
	a_ht = 6.0 / di;
		
	P1[i].DEMpor=pori;

	if((mag_uijf>1.0e-8) && ((flt_fluid+flt_air)>0.00001) && (f_fluid>0.65)){		// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
		

		Ref=di*mag_uijf*pori/(vis_kif+1.0e-15);
		Cdf=(0.63+4.8/sqrt(Ref))*(0.63+4.8/sqrt(Ref));
		//betaf=3.7-0.65*exp(-(1.5-log10(Ref))*(1.5-log10(Ref))/2.0);
		betaf=2.65*(1.0+pori)-(5.3-3.5*pori)*pori*pori*exp(-(1.5-log10(Ref))*(1.5-log10(Ref))/2.0);


		Real SPH_filter=flt_fluid + flt_air;


		dfxf=(1.0/8.0)*Cdf*rho_f*3.14159*di*di*uxijf*mag_uijf*pow(pori,-1-betaf)/mi;
		dfyf=(1.0/8.0)*Cdf*rho_f*3.14159*di*di*uyijf*mag_uijf*pow(pori,-1-betaf)/mi;	
		dfzf=(1.0/8.0)*Cdf*rho_f*3.14159*di*di*uzijf*mag_uijf*pow(pori,-1-betaf)/mi;	

		
		
		Pr_f = 0.7;
		Nu_f = 2.0 + 0.6 * pow(Ref, 0.5) * pow(Pr_f, 1 / 3);
		h_conv_f = Nu_f * 0.0518 / di;
		dq_vol = h_conv_f * a_ht * temp_ijf;

		 P1[i].Fdx_df=dfxf;
		 P1[i].Fdy_df=dfyf;
		 P1[i].Fdz_df=dfzf;




	}

	if((mag_uija>1.0e-8) && ((flt_fluid+flt_air)>0.00001) && (f_air>0.65)){		// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
		
		// if(flti>0.9) pori=tmp_por/flti;
		// else pori=tmp_por;

		Rea=di*pori*mag_uija/(vis_kia+1.0e-15);
		//Re=di*pori*mag_uij_vis;

		//if(Re<=1000.0)	Cd=24.0*(1.0+0.15*pow(Re,0.687))/(Re+1.0e-15);
		//else			Cd=0.44;
		Cda=(0.63+4.8/sqrt(Rea))*(0.63+4.8/sqrt(Rea));
		betaa=2.65*(1.0+pori)-(5.3-3.5*pori)*pori*pori*exp(-(1.5-log10(Rea))*(1.5-log10(Rea))/2.0);
		//betaa=3.7-0.65*exp(-(1.5-log10(Rea))*(1.5-log10(Rea))/2.0);

		//if(pori<=0.8)	beta=150*vis_ki*rho_f*(1.0-pori)*(1.0-pori)/(pori*di*di+1.0e-15)+1.75*(1.0-pori)*rho_f*mag_uij/(di+1.0e-15);
		//else			beta=(3.0/4.0)*Cd*pori*(1.0-pori)*rho_f*mag_uij*pow(pori,-2.65)/(di+1.0e-15);

	
		//dfx=beta*(voli/mi)*uxij/((1-pori)+1.0e-15);		// Drag Force [acceleration]
		//dfy=beta*(voli/mi)*uyij/((1-pori)+1.0e-15);
		//dfz=beta*(voli/mi)*uzij/((1-pori)+1.0e-15);


		//dfxf=(1.0/8.0)*Cdf*rho_eff*3.14159*di*di*uxijf*mag_uijf*pow(pori,-1-betaf)/mi;
		dfxa=(1.0/8.0)*Cda*rho_a*3.14159*di*di*uxija*mag_uija*pow(pori,-1-betaa)/mi;
		dfya=(1.0/8.0)*Cda*rho_a*3.14159*di*di*uyija*mag_uija*pow(pori,-1-betaa)/mi;
		dfza=(1.0/8.0)*Cda*rho_a*3.14159*di*di*uzija*mag_uija*pow(pori,-1-betaa)/mi;

		

		Pr_a = 1.0;
		Nu_a = 2.0 + 0.6 * pow(Rea, 0.5) * pow(Pr_a, 1 / 3);
		h_conv_a = Nu_a * ki / di;
		dq_vol = h_conv_a * a_ht * temp_ija;
	}

	
	if(((mag_uija>1.0e-8)||(mag_uijf>1.0e-8)) && ((flt_fluid+flt_air)>0.00001) && (f_air<=0.65)&&(f_fluid<=0.65)){	// �ֺ� ��ü�� ��ü+��ü
		
		// if(flti>0.9) pori=tmp_por/flti;
		// else pori=tmp_por;

		rho_fa=DENSITY_AIR*f_air+DENSITY_WATER*f_fluid;
		vis_fa=VISCOSITY_AIR*f_air+VISCOSITY_WATER*f_fluid;
		vis_kifa=vis_fa/(rho_fa+1.0e-15);

		Refa=di*pori*mag_uijfa/(vis_kifa+1.0e-15);
		//Re=di*pori*mag_uij_vis;

		//if(Re<=1000.0)	Cd=24.0*(1.0+0.15*pow(Re,0.687))/(Re+1.0e-15);
		//else			Cd=0.44;
		Cdfa=(0.63+4.8/sqrt(Refa))*(0.63+4.8/sqrt(Refa));
		betafa=2.65*(1.0+pori)-(5.3-3.5*pori)*pori*pori*exp(-(1.5-log10(Refa))*(1.5-log10(Refa))/2.0);
		//betafa=3.7-0.65*exp(-(1.5-log10(Refa))*(1.5-log10(Refa))/2.0);

		//if(pori<=0.8)	beta=150*vis_ki*rho_f*(1.0-pori)*(1.0-pori)/(pori*di*di+1.0e-15)+1.75*(1.0-pori)*rho_f*mag_uij/(di+1.0e-15);
		//else			beta=(3.0/4.0)*Cd*pori*(1.0-pori)*rho_f*mag_uij*pow(pori,-2.65)/(di+1.0e-15);

	
		//dfx=beta*(voli/mi)*uxij/((1-pori)+1.0e-15);		// Drag Force [acceleration]
		//dfy=beta*(voli/mi)*uyij/((1-pori)+1.0e-15);
		//dfz=beta*(voli/mi)*uzij/((1-pori)+1.0e-15);



		dfxa=(1.0/8.0)*Cdfa*rho_fa*3.14159*di*di*uxijfa*mag_uijfa*pow(pori,-1-betafa)/mi;
		dfya=(1.0/8.0)*Cdfa*rho_fa*3.14159*di*di*uyijfa*mag_uijfa*pow(pori,-1-betafa)/mi;
		dfza=(1.0/8.0)*Cdfa*rho_fa*3.14159*di*di*uzijfa*mag_uijfa*pow(pori,-1-betafa)/mi;


		Pr_fa = 1.0;
		Nu_fa = 2.0 + 0.6 * pow(Refa, 0.5) * pow(Pr_fa, 1 / 3);
		h_conv_fa = Nu_fa * ki / di;
		dq_vol = h_conv_fa * a_ht * temp_ijfa;


		

	}


	// if abs(dfxf)>1.0e5 dfxf=dfxf*1.0e5/abs(dfxf);
	// if abs(dfyf)>1.0e5 dfyf=dfyf*1.0e5/abs(dfyf);
	// if abs(dfzf)>1.0e5 dfzf=dfzf*1.0e5/abs(dfzf);



	// Add SPH-DEM coupling force calculation (acting on DEM particles)
	P3[i].ftotalx=ftotalx+dfxf+tmpx;		
	P3[i].ftotaly=ftotaly+dfyf+tmpy;		
	P3[i].ftotalz=ftotalz+dfzf+tmpz;		


	if (k_con_solve==1){
		P3[i].dtemp = dq_vol / (rhoi * cpi);	//[K/s]
		P1[i].Q_sd=dq_vol*mi/rhoi;		//[J/s]	
	
		//if (abs(temp_ija)>10) 
		//P3[i].dtemp=-10.0;
	}
	

	// Save SPH-DEM coupling force acting on DEM particles ()


	P1[i].Fdx_da=dfxa;		// 0 (2021.07.20) : change the RE number estimation model
	P1[i].Fdy_da=dfya;		// 0 (2021.07.20) : change the RE number estimation model	
	P1[i].Fdz_da=dfza;		// 0 (2021.07.20) : change the RE number estimation model
	

	// Save SPH-DEM coupling force acting on DEM particles ()
	P1[i].Fdx_b=tmpx;
	P1[i].Fdy_b=tmpy;
	P1[i].Fdz_b=tmpz;	




}

__global__ void KERNEL_DEM_coupling3D_dem(int_t inout,int_t*g_str_dem,int_t*g_end_dem,part1*P1_dem,part2*P2_dem,part3*P3_dem,
											int_t*g_str_sph,int_t*g_end_sph,part1*P1_sph,part2*P2_sph,part3*P3_sph)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;

	if((i>=k_num_part2_dem) || (P1_dem[i].i_type!=inout) || (P1_dem[i].p_type<=1000)) return;
	// if(i>=k_num_part2) return;
	// if(P1[i].i_type!=inout) return;
	// if(P1[i].p_type<=1000) return;		// leave if it is not DEM particle

	
	int_t ptypei;
	int_t icell,jcell,kcell;
	int_t demidxi;
	Real xi,yi,zi,uxi,uyi,uzi;	
	Real tempi, tempj,ki,alph_i,cpi;
	Real mi,radi,di,voli,rhoi,hi;	
	Real flt_air,flt_fluid;											
	Real ftotalx,ftotaly,ftotalz;
	Real vfi;
	
	Real ufxf, ufyf, ufzf, uxijf,uyijf,uzijf,mag_uijf;
	Real ufxa, ufya, ufza, uxija,uyija,uzija,mag_uija;
	Real ufxfa, ufyfa, ufzfa, uxijfa,uyijfa,uzijfa,mag_uijfa;
	Real ufx_visf, ufy_visf, ufz_visf;
	Real ufx_visa, ufy_visa, ufz_visa;
	Real Ref,Cdf,betaf,pori;
	Real Rea,Cda,betaa;
	Real Refa,Cdfa,betafa;
	Real vis_kif,vis_k_invf,rho_f;
	Real vis_kia,vis_k_inva,rho_a;
	Real vis_kifa,rho_fa,vis_fa;
	Real dfxf,dfyf,dfzf;
	Real dfxa,dfya,dfza;

	Real temp_f, temp_a, temp_fa, temp_ijf, temp_ija, temp_ijfa;
	Real dtempi;
	Real Pr_f, Pr_a, Pr_fa, a_ht;
	Real dq_vol;

	Real Nu_f, Nu_a, Nu_fa;
	Real h_conv_f, h_conv_a, h_conv_fa;

	
	
	//Real search_range;
	Real tmpx,tmpy,tmpz;			// tmptq : temporary value for torque calculation
	Real tmp_ufxf,tmp_ufyf,tmp_ufzf,tmp_por;
	Real tmp_ufxa,tmp_ufya,tmp_ufza;
	Real tmp_temp_f, tmp_temp_a;
	Real f_fluid,f_air;
	Real search_range,tmp_A;


	ptypei = P1_dem[i].p_type;

	// position
	xi=P1_dem[i].x;
	yi=P1_dem[i].y;
	zi=P1_dem[i].z;

	// velocity
	uxi=P1_dem[i].ux;
	uyi=P1_dem[i].uy;
	uzi=P1_dem[i].uz;
	

	// radius & diameter
	radi=P1_dem[i].rad;
	di=2*radi;
	voli=(4.0/3.0)*3.14159*radi*radi*radi;

	// mass,density
	mi=P1_dem[i].m;
	rhoi=P1_dem[i].rho;

	// temperature
	tempi = P1_dem[i].temp;

	demidxi=P1_dem[i].dem_idx;

	if(radi<0.0012){
		ki=conductivity_1(tempi,ptypei,1);
		alph_i=diffusivity_1(tempi,ptypei,1);
		cpi=(ki)/(alph_i*rhoi);

	}
	else{
		ki=conductivity_1(tempi,ptypei,0);
		alph_i=diffusivity_1(tempi,ptypei,0);
		cpi=(ki)/(alph_i*rhoi);
	}

	// filter (SPH particles around DEM particle i)
	flt_air=P1_dem[i].flt_s;
	flt_fluid=P1_dem[i].flt_sd;

	//if (flt_air<0.01) return;
	//if (flt_fluid<0.01) return;

	// ftotal
	ftotalx=P3_dem[i].ftotalx;
	ftotaly=P3_dem[i].ftotaly;
	ftotalz=P3_dem[i].ftotalz;

	// dtemp
	dtempi = P3_dem[i].dtemp;
	
	





	//search_range=k_search_kappa*hi;	// search range
	hi=1*P1_dem[i].h;
	tmp_A=calc_tmpA(1.0*hi);
	search_range=k_search_kappa*1.0*hi;	// search range


	// Sinked Volume Fraction of DEM Particle
	vfi=P1_dem[i].DEMvf;





	
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
	tmp_ufxf=tmp_ufyf=tmp_ufzf=0.0;
	tmp_ufxa=tmp_ufya=tmp_ufza=0.0;
	tmp_por=0.0;
	tmp_temp_f = tmp_temp_a = 0.0;


	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;

				if(g_str_sph[k]!=cu_memset){
					int_t fend=g_end_sph[k];
					for(int_t j=g_str_sph[k];j<fend;j++){


						
						

						

						if ((P1_sph[j].p_type<=1000) && (P1_sph[j].DEMpor<1.0)   ){
							Real xj,yj,zj,tdist; 
						
							xj=P1_sph[j].x;
							yj=P1_sph[j].y;
							zj=P1_sph[j].z;
							
			
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist<search_range){

								Real twij;
								Real uxj,uyj,uzj;
								Real mj,porj,rhoj,rho_refj;
								Real pgf_xj,pgf_yj,pgf_zj;

								Real tempj,visj;
								int_t ptypej;

								twij=calc_kernel_wij(tmp_A,1.0*hi,tdist);
								

								uxj=P1_sph[j].ux;
								uyj=P1_sph[j].uy;
								uzj=P1_sph[j].uz;



								mj=P1_sph[j].m;
								porj=P1_sph[j].DEMpor;
								rhoj=P1_sph[j].rho;
								tempj = P1_sph[j].temp;
								rho_refj=P2_sph[j].rho_ref;
								

								pgf_xj=P1_sph[j].pgf_x;
								pgf_yj=P1_sph[j].pgf_y;
								pgf_zj=P1_sph[j].pgf_z;

								ptypej=P1_sph[j].p_type;
								tempj=P1_sph[j].temp;
								visj=viscosity(tempj,ptypej);

								




								if (ptypej==1){

									if(flt_fluid>0.000001){

										vis_kif=VISCOSITY_AIR/DENSITY_AIR;
										rho_f=DENSITY_AIR;
										vis_k_invf=rho_f/visj;
	
										tmp_ufxf+=uxj*mj*twij/rhoj;
										tmp_ufyf+=uyj*mj*twij/rhoj;
										tmp_ufzf+=uzj*mj*twij/rhoj;
	
										tmp_por+=porj*mj*twij/rhoj;

										tmp_temp_f+=tempj* mj * twij / rhoj;

									}



								}

								if (ptypej==3){

									if(flt_air>0.000001){
										vis_kia=VISCOSITY_AIR/DENSITY_AIR;
										rho_a=DENSITY_AIR;
										vis_k_inva=rho_a/visj;
	
										tmp_ufxa+=uxj*mj*twij/rhoj;
										tmp_ufya+=uyj*mj*twij/rhoj;
										tmp_ufza+=uzj*mj*twij/rhoj;
	
										tmp_por+=porj*mj*twij/rhoj;

										tmp_temp_a += tempj * mj * twij / rhoj;



									}


								}
								
								if ((ptypej!=0) && (ptypej!=MOVING) && (ptypej!=-3)&& (ptypej!=3)){
								//if ((ptypej!=0) | (P1[j].flt_s>0.8)){++

									if(flt_fluid>0.01){

										tmpx+=pgf_xj*(mj/rhoj)*twij*voli/(mi*flt_fluid+1.0e-15);		// pressure gradient force
										tmpy+=pgf_yj*(mj/rhoj)*twij*voli/(mi*flt_fluid+1.0e-15);
										tmpz+=pgf_zj*(mj/rhoj)*twij*voli/(mi*flt_fluid+1.0e-15);


									}


									
								

								}


							}
								
	
						}


					}			
				}
			}
		}
	}

	pori=tmp_por/(P1_dem[i].flt_sd+1.0e-15);
	//pori=P1[i].DEMpor;
	
	ufxf=tmp_ufxf/(flt_fluid+1.0e-15);	// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
	ufyf=tmp_ufyf/(flt_fluid+1.0e-15);
	ufzf=tmp_ufzf/(flt_fluid+1.0e-15);
	temp_f=tmp_temp_f/ (flt_fluid + 1.0e-15);

	ufxa=tmp_ufxa/(flt_air+1.0e-15);	// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
	ufya=tmp_ufya/(flt_air+1.0e-15);
	ufza=tmp_ufza/(flt_air+1.0e-15);
	temp_a = tmp_temp_a / (flt_air + 1.0e-15);

	//P1[i].test2=temp_a;
	

	ufxfa=(tmp_ufxf+tmp_ufxa)/(flt_fluid+flt_air+1.0e-15);	// �ֺ� ��ü�� ��ü+��ü
	ufyfa=(tmp_ufyf+tmp_ufya)/(flt_fluid+flt_air+1.0e-15);
	ufzfa=(tmp_ufzf+tmp_ufza)/(flt_fluid+flt_air+1.0e-15);
	temp_fa = (tmp_temp_f + tmp_temp_a) / (flt_fluid + flt_air + 1.0e-15);

	

	uxijf=ufxf-uxi;		// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
	uyijf=ufyf-uyi;
	uzijf=ufzf-uzi;
	temp_ijf = temp_f - tempi;

	uxija=ufxa-uxi;		// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
	uyija=ufya-uyi;
	uzija=ufza-uzi;
	temp_ija = temp_a - tempi;

	//P1[i].test3=uzija;
	//P1[i].test4=temp_ija;

	uxijfa=ufxfa-uxi;	// �ֺ� ��ü�� ��ü+��ü
	uyijfa=ufyfa-uyi;
	uzijfa=ufzfa-uzi;
	temp_ijfa = temp_fa - tempi;

	f_fluid=flt_fluid/(flt_fluid+flt_air+1.0e-15);
	f_air=flt_air/(flt_fluid+flt_air+1.0e-15);


	mag_uijf=sqrt(uxijf*uxijf+uyijf*uyijf+uzijf*uzijf);
	mag_uija=sqrt(uxija*uxija+uyija*uyija+uzija*uzija);
	mag_uijfa=sqrt(uxijfa*uxijfa+uyijfa*uyijfa+uzijfa*uzijfa);

	P1_dem[i].test6=mag_uijf;

	dfxf=0.0;
	dfyf=0.0;
	dfzf=0.0;

	dfxa=0.0;
	dfya=0.0;
	dfza=0.0;

	//pori=tmp_por/(flt_fluid+flt_air+1.0e-15);	//original
	
	
	//pori=0.38;
	a_ht = 6.0 / di;
		
	P1_dem[i].DEMpor=pori;

	if((mag_uijf>1.0e-8) && ((flt_fluid+flt_air)>0.00001) && (f_fluid>0.65)){		// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
		

		Ref=di*mag_uijf*pori/(vis_kif+1.0e-15);
		Cdf=(0.63+4.8/sqrt(Ref))*(0.63+4.8/sqrt(Ref));
		//betaf=3.7-0.65*exp(-(1.5-log10(Ref))*(1.5-log10(Ref))/2.0);
		betaf=2.65*(1.0+pori)-(5.3-3.5*pori)*pori*pori*exp(-(1.5-log10(Ref))*(1.5-log10(Ref))/2.0);


		Real SPH_filter=flt_fluid + flt_air;


		dfxf=(1.0/8.0)*Cdf*rho_f*3.14159*di*di*uxijf*mag_uijf*pow(pori,-1-betaf)/mi;
		dfyf=(1.0/8.0)*Cdf*rho_f*3.14159*di*di*uyijf*mag_uijf*pow(pori,-1-betaf)/mi;	
		dfzf=(1.0/8.0)*Cdf*rho_f*3.14159*di*di*uzijf*mag_uijf*pow(pori,-1-betaf)/mi;	

		
		
		Pr_f = 0.7;
		Nu_f = 2.0 + 0.6 * pow(Ref, 0.5) * pow(Pr_f, 1 / 3);
		h_conv_f = Nu_f * 0.0518 / di;
		dq_vol = h_conv_f * a_ht * temp_ijf;

		 P1_dem[i].Fdx_df=dfxf;
		 P1_dem[i].Fdy_df=dfyf;
		 P1_dem[i].Fdz_df=dfzf;




	}

	if((mag_uija>1.0e-8) && ((flt_fluid+flt_air)>0.00001) && (f_air>0.65)){		// �ֺ� ��ü ���ڰ� ��κ� ��ü�� ��
		
		// if(flti>0.9) pori=tmp_por/flti;
		// else pori=tmp_por;

		Rea=di*pori*mag_uija/(vis_kia+1.0e-15);
		//Re=di*pori*mag_uij_vis;

		//if(Re<=1000.0)	Cd=24.0*(1.0+0.15*pow(Re,0.687))/(Re+1.0e-15);
		//else			Cd=0.44;
		Cda=(0.63+4.8/sqrt(Rea))*(0.63+4.8/sqrt(Rea));
		betaa=2.65*(1.0+pori)-(5.3-3.5*pori)*pori*pori*exp(-(1.5-log10(Rea))*(1.5-log10(Rea))/2.0);
		//betaa=3.7-0.65*exp(-(1.5-log10(Rea))*(1.5-log10(Rea))/2.0);

		//if(pori<=0.8)	beta=150*vis_ki*rho_f*(1.0-pori)*(1.0-pori)/(pori*di*di+1.0e-15)+1.75*(1.0-pori)*rho_f*mag_uij/(di+1.0e-15);
		//else			beta=(3.0/4.0)*Cd*pori*(1.0-pori)*rho_f*mag_uij*pow(pori,-2.65)/(di+1.0e-15);

	
		//dfx=beta*(voli/mi)*uxij/((1-pori)+1.0e-15);		// Drag Force [acceleration]
		//dfy=beta*(voli/mi)*uyij/((1-pori)+1.0e-15);
		//dfz=beta*(voli/mi)*uzij/((1-pori)+1.0e-15);


		//dfxf=(1.0/8.0)*Cdf*rho_eff*3.14159*di*di*uxijf*mag_uijf*pow(pori,-1-betaf)/mi;
		dfxa=(1.0/8.0)*Cda*rho_a*3.14159*di*di*uxija*mag_uija*pow(pori,-1-betaa)/mi;
		dfya=(1.0/8.0)*Cda*rho_a*3.14159*di*di*uyija*mag_uija*pow(pori,-1-betaa)/mi;
		dfza=(1.0/8.0)*Cda*rho_a*3.14159*di*di*uzija*mag_uija*pow(pori,-1-betaa)/mi;

		

		Pr_a = 1.0;
		Nu_a = 2.0 + 0.6 * pow(Rea, 0.5) * pow(Pr_a, 1 / 3);
		h_conv_a = Nu_a * ki / di;
		dq_vol = h_conv_a * a_ht * temp_ija;
	}

	
	if(((mag_uija>1.0e-8)||(mag_uijf>1.0e-8)) && ((flt_fluid+flt_air)>0.00001) && (f_air<=0.65)&&(f_fluid<=0.65)){	// �ֺ� ��ü�� ��ü+��ü
		
		// if(flti>0.9) pori=tmp_por/flti;
		// else pori=tmp_por;

		rho_fa=DENSITY_AIR*f_air+DENSITY_WATER*f_fluid;
		vis_fa=VISCOSITY_AIR*f_air+VISCOSITY_WATER*f_fluid;
		vis_kifa=vis_fa/(rho_fa+1.0e-15);

		Refa=di*pori*mag_uijfa/(vis_kifa+1.0e-15);
		//Re=di*pori*mag_uij_vis;

		//if(Re<=1000.0)	Cd=24.0*(1.0+0.15*pow(Re,0.687))/(Re+1.0e-15);
		//else			Cd=0.44;
		Cdfa=(0.63+4.8/sqrt(Refa))*(0.63+4.8/sqrt(Refa));
		betafa=2.65*(1.0+pori)-(5.3-3.5*pori)*pori*pori*exp(-(1.5-log10(Refa))*(1.5-log10(Refa))/2.0);
		//betafa=3.7-0.65*exp(-(1.5-log10(Refa))*(1.5-log10(Refa))/2.0);

		//if(pori<=0.8)	beta=150*vis_ki*rho_f*(1.0-pori)*(1.0-pori)/(pori*di*di+1.0e-15)+1.75*(1.0-pori)*rho_f*mag_uij/(di+1.0e-15);
		//else			beta=(3.0/4.0)*Cd*pori*(1.0-pori)*rho_f*mag_uij*pow(pori,-2.65)/(di+1.0e-15);

	
		//dfx=beta*(voli/mi)*uxij/((1-pori)+1.0e-15);		// Drag Force [acceleration]
		//dfy=beta*(voli/mi)*uyij/((1-pori)+1.0e-15);
		//dfz=beta*(voli/mi)*uzij/((1-pori)+1.0e-15);



		dfxa=(1.0/8.0)*Cdfa*rho_fa*3.14159*di*di*uxijfa*mag_uijfa*pow(pori,-1-betafa)/mi;
		dfya=(1.0/8.0)*Cdfa*rho_fa*3.14159*di*di*uyijfa*mag_uijfa*pow(pori,-1-betafa)/mi;
		dfza=(1.0/8.0)*Cdfa*rho_fa*3.14159*di*di*uzijfa*mag_uijfa*pow(pori,-1-betafa)/mi;


		Pr_fa = 1.0;
		Nu_fa = 2.0 + 0.6 * pow(Refa, 0.5) * pow(Pr_fa, 1 / 3);
		h_conv_fa = Nu_fa * ki / di;
		dq_vol = h_conv_fa * a_ht * temp_ijfa;


		

	}


	// if abs(dfxf)>1.0e5 dfxf=dfxf*1.0e5/abs(dfxf);
	// if abs(dfyf)>1.0e5 dfyf=dfyf*1.0e5/abs(dfyf);
	// if abs(dfzf)>1.0e5 dfzf=dfzf*1.0e5/abs(dfzf);



	// Add SPH-DEM coupling force calculation (acting on DEM particles)
	P3_dem[i].ftotalx=ftotalx+dfxf+tmpx;		
	P3_dem[i].ftotaly=ftotaly+dfyf+tmpy;		
	P3_dem[i].ftotalz=ftotalz+dfzf+tmpz;		


	if (k_con_solve==1){
		P3_dem[i].dtemp = dq_vol / (rhoi * cpi);	//[K/s]
		P1_dem[i].Q_sd=dq_vol*mi/rhoi;		//[J/s]	
	
		//if (abs(temp_ija)>10) 
		//P3[i].dtemp=-10.0;
	}
	

	// Save SPH-DEM coupling force acting on DEM particles ()


	P1_dem[i].Fdx_da=dfxa;		// 0 (2021.07.20) : change the RE number estimation model
	P1_dem[i].Fdy_da=dfya;		// 0 (2021.07.20) : change the RE number estimation model	
	P1_dem[i].Fdz_da=dfza;		// 0 (2021.07.20) : change the RE number estimation model
	

	// Save SPH-DEM coupling force acting on DEM particles ()
	P1_dem[i].Fdx_b=tmpx;
	P1_dem[i].Fdy_b=tmpy;
	P1_dem[i].Fdz_b=tmpz;	




}
// ////////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_DEM_coupling3D_b(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
// {
// 	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type!=inout) return;
// 	if(P1[i].p_type<=1000) return;		// leave if it is not DEM particle

	
// 	int_t ptypei;
// 	int_t icell,jcell,kcell;
// 	Real xi,yi,zi,uxi,uyi,uzi;			
// 	Real mi,radi,di,voli,rhoi,hi;	
// 	Real flti;											
// 	Real ftotalx,ftotaly,ftotalz;
// 	Real vfi;
	
	
// 	//Real search_range;
// 	Real tmpx,tmpy,tmpz;			// tmptq : temporary value for torque calculation
	
// 	Real search_range,tmp_A;



// 	// position
// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;

// 	// velocity
// 	uxi=P1[i].ux;
// 	uyi=P1[i].uy;
// 	uzi=P1[i].uz;
	

// 	// radius & diameter
// 	radi=P1[i].rad;
// 	di=2*radi;
// 	voli=(4.0/3.0)*3.14159*radi*radi*radi;

// 	// mass,density
// 	mi=P1[i].m;
// 	rhoi=P1[i].rho;

// 	// filter (SPH particles around DEM particle i)
// 	flti=P1[i].flt_s;

// 	if (flti<0.01) return;

// 	// ftotal
// 	ftotalx=P3[i].ftotalx;
// 	ftotaly=P3[i].ftotaly;
// 	ftotalz=P3[i].ftotalz;




// 	//search_range=k_search_kappa*hi;	// search range
// 	hi=P1[i].h;
// 	tmp_A=calc_tmpA(hi);
// 	search_range=k_search_kappa*hi;	// search range


// 	// Sinked Volume Fraction of DEM Particle
// 	vfi=P1[i].DEMvf;





	
// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;


// 	tmpx=tmpy=tmpz=0.0;


// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;

// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
						

// 						if ((P1[j].p_type<=1000) && (P1[j].DEMpor<1.0)  && (P1[j].DEMpor>0.05) && (P1[j].p_type!=-3) ){
// 							Real xj,yj,zj,tdist; 
						
// 							xj=P1[j].x;
// 							yj=P1[j].y;
// 							zj=P1[j].z;
							
			
// 							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
// 							if(tdist<search_range){

// 								Real twij;
// 								Real uxj,uyj,uzj;
// 								Real mj,rhoj,rho_refj;
// 								Real pgf_xj,pgf_yj,pgf_zj;

// 								int_t ptypej;

// 								twij=calc_kernel_wij(tmp_A,hi,tdist);
								

// 								uxj=P1[j].ux;
// 								uyj=P1[j].uy;
// 								uzj=P1[j].uz;



// 								mj=P1[j].m;
// 								rhoj=P1[j].rho;
// 								rho_refj=P2[j].rho_ref;
								

// 								pgf_xj=P1[j].pgf_x;
// 								pgf_yj=P1[j].pgf_y;
// 								pgf_zj=P1[j].pgf_z;

// 								ptypej=P1[j].p_type;
								
// 								if ((ptypej!=0) && (ptypej!=MOVING) && (ptypej!=-3)){
// 								//if ((ptypej!=0) | (P1[j].flt_s>0.8)){


									

// 									if((rho_refj>rhoi) && (flti<0.85)){

// 										// if((rho_refj>rhoi) && (vfi>0.1)){
// 										// 	//tmpx+=(rhoi/rho_refj)*pgf_xj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);		// pressure gradient force
// 										// 	//tmpy+=(rhoi/rho_refj)*pgf_yj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);
// 										// 	//tmpz+=(rhoi/rho_refj)*pgf_zj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);
// 										// 	tmpx+=0.0;
// 										// 	tmpy+=0.0;
// 										// 	tmpz+=9
// 										// }
// 										// else if((rho_refj>rhoi) && (vfi<=0.1)){
// 										// 	tmpx+=0.0;
// 										// 	tmpy+=0.0;
// 										// 	tmpz+=0.0;
// 										// }
// 										// else{
// 										// 	tmpx+=vfi*pgf_xj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);		// pressure gradient force
// 										// 	tmpy+=vfi*pgf_yj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);
// 										// 	tmpz+=vfi*pgf_zj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);

// 										// }

// 										if (flti>0.3){

// 											tmpx+=pgf_xj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);		// pressure gradient force
// 											tmpy+=pgf_yj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);
// 											tmpz+=pgf_zj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);
// 										}


										
// 									}
// 									else{
// 										tmpx+=pgf_xj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);		// pressure gradient force
// 										tmpy+=pgf_yj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);
// 										tmpz+=pgf_zj*(mj/rhoj)*twij*voli/(mi*flti+1.0e-15);

// 									}
									
								

// 								}


// 							}
								
	
// 						}


// 					}			
// 				}
// 			}
// 		}
// 	}

// 	// Add SPH-DEM coupling force calculation (acting on DEM particles)
// 	P3[i].ftotalx=ftotalx+tmpx;
// 	P3[i].ftotaly=ftotaly+tmpy;
// 	P3[i].ftotalz=ftotalz+tmpz;
	
// 	// Save SPH-DEM coupling force acting on DEM particles ()
// 	P1[i].Fdx_b=tmpx;
// 	P1[i].Fdy_b=tmpy;
// 	P1[i].Fdz_b=tmpz;	




// }


// ////////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_DEM_coupling3D_d(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part2*P2,part3*P3)
// {
// 	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type!=inout) return;
// 	if(P1[i].p_type<=1000) return;		// leave if it is not DEM particle

	
// 	int_t ptypei;
// 	int_t icell,jcell,kcell;
// 	Real xi,yi,zi,uxi,uyi,uzi;			
// 	Real mi,radi,di,voli,rhoi,hi;	
// 	Real flti;											
// 	Real ftotalx,ftotaly,ftotalz;
// 	Real vfi;

// 	Real ufx, ufy, ufz, uxij,uyij,uzij,mag_uij;
// 	Real ufx_vis, ufy_vis, ufz_vis, mag_uij_vis;
// 	Real Re,Cd,beta,pori;
// 	Real vis_ki,vis_k_inv,rho_f;
// 	Real dfx,dfy,dfz;
	
	
// 	//Real search_range;
	
// 	Real tmp_ufx,tmp_ufy,tmp_ufz,tmp_por;
// 	Real tmp_ufx_vis,tmp_ufy_vis,tmp_ufz_vis;
// 	Real search_range,tmp_A;



// 	// position
// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;

// 	// velocity
// 	uxi=P1[i].ux;
// 	uyi=P1[i].uy;
// 	uzi=P1[i].uz;
	

// 	// radius & diameter
// 	radi=P1[i].rad;
// 	di=2*radi;
// 	voli=(4.0/3.0)*3.14159*radi*radi*radi;

// 	// mass,density
// 	mi=P1[i].m;
// 	rhoi=P1[i].rho;

// 	// filter (SPH particles around DEM particle i)
// 	flti=P1[i].flt_sd;

// 	if (flti<0.01) return;

// 	// ftotal
// 	ftotalx=P3[i].ftotalx;
// 	ftotaly=P3[i].ftotaly;
// 	ftotalz=P3[i].ftotalz;




// 	//search_range=k_search_kappa*hi;	// search range
// 	hi=P1[i].h;
// 	tmp_A=calc_tmpA(hi);
// 	search_range=k_search_kappa*hi;	// search range


// 	// Sinked Volume Fraction of DEM Particle
// 	vfi=P1[i].DEMvf;





	
// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;


// 	tmp_ufx=tmp_ufy=tmp_ufz=0.0;
// 	tmp_ufx_vis=tmp_ufy_vis=tmp_ufz_vis=0.0;
// 	tmp_por=0.0;

// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;

// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
						

// 						if ((P1[j].p_type<=1000) && (P1[j].DEMpor<1.0)  && (P1[j].DEMpor>0.05) && (P1[j].p_type==1)){
// 							Real xj,yj,zj,tdist; 
						
// 							xj=P1[j].x;
// 							yj=P1[j].y;
// 							zj=P1[j].z;
							
			
// 							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
// 							if(tdist<search_range){

// 								Real twij;
// 								Real uxj,uyj,uzj;
// 								Real mj,porj,rhoj,rho_refj;
// 								Real pgf_xj,pgf_yj,pgf_zj;

// 								Real tempj,visj;
// 								int_t ptypej;

// 								twij=calc_kernel_wij(tmp_A,hi,tdist);
								

// 								uxj=P1[j].ux;
// 								uyj=P1[j].uy;
// 								uzj=P1[j].uz;



// 								mj=P1[j].m;
// 								porj=P1[j].DEMpor;
// 								rhoj=P1[j].rho;
// 								rho_refj=P2[j].rho_ref;
								


// 								ptypej=P1[j].p_type;
// 								tempj=P1[j].temp;
// 								visj=viscosity(tempj,ptypej);


								
// 								if ((ptypej!=0) && (ptypej!=MOVING)){
// 								//if ((ptypej!=0) | (P1[j].flt_s>0.8)){

// 									vis_ki=visj/DENSITY_WATER;
// 									rho_f=DENSITY_WATER;
// 									vis_k_inv=rho_f/visj;

// 									tmp_ufx+=uxj*mj*twij/rhoj;
// 									tmp_ufy+=uyj*mj*twij/rhoj;
// 									tmp_ufz+=uzj*mj*twij/rhoj;

// 									tmp_ufx_vis+=vis_k_inv*(uxj-uxi)*mj*twij/rhoj;
// 									tmp_ufy_vis+=vis_k_inv*(uyj-uyi)*mj*twij/rhoj;
// 									tmp_ufz_vis+=vis_k_inv*(uzj-uzi)*mj*twij/rhoj;

// 									tmp_por+=porj*mj*twij/rhoj;


						
// 								}


// 							}
								
	
// 						}


// 					}			
// 				}
// 			}
// 		}
// 	}

// 	if (flti>0.02)
// 	{


// 		ufx=tmp_ufx/(flti+1.0e-15);
// 		ufy=tmp_ufy/(flti+1.0e-15);
// 		ufz=tmp_ufz/(flti+1.0e-15);

// 		ufx_vis=tmp_ufx_vis/(flti+1.0e-15);
// 		ufy_vis=tmp_ufy_vis/(flti+1.0e-15);
// 		ufz_vis=tmp_ufz_vis/(flti+1.0e-15);


	
// 		uxij=ufx-uxi;
// 		uyij=ufy-uyi;
// 		uzij=ufz-uzi;



// 		mag_uij=sqrt(uxij*uxij+uyij*uyij+uzij*uzij);
// 		mag_uij_vis=sqrt(ufx_vis*ufx_vis+ufy_vis*ufy_vis+ufz_vis*ufz_vis);

// 		dfx=0.0;
// 		dfy=0.0;
// 		dfz=0.0;
	
// 		if(mag_uij>1.0e-8){
			
// 			// if(flti>0.9) pori=tmp_por/flti;
// 			// else pori=tmp_por;

// 			pori=tmp_por/(flti+1.0e-15);
			
// 			//pori=P1[i].DEMpor;
		
	
// 			Re=di*pori*mag_uij/(vis_ki+1.0e-15);
// 			//Re=di*pori*mag_uij_vis;
	
// 			//if(Re<=1000.0)	Cd=24.0*(1.0+0.15*pow(Re,0.687))/(Re+1.0e-15);
// 			//else			Cd=0.44;
// 			Cd=(0.63+4.8/sqrt(Re))*(0.63+4.8/sqrt(Re));
// 			beta=3.7-0.65*exp(-(1.5-log10(Re))*(1.5-log10(Re))/2.0);
	
// 			//if(pori<=0.8)	beta=150*vis_ki*rho_f*(1.0-pori)*(1.0-pori)/(pori*di*di+1.0e-15)+1.75*(1.0-pori)*rho_f*mag_uij/(di+1.0e-15);
// 			//else			beta=(3.0/4.0)*Cd*pori*(1.0-pori)*rho_f*mag_uij*pow(pori,-2.65)/(di+1.0e-15);
	
		
// 			//dfx=beta*(voli/mi)*uxij/((1-pori)+1.0e-15);		// Drag Force [acceleration]
// 			//dfy=beta*(voli/mi)*uyij/((1-pori)+1.0e-15);
// 			//dfz=beta*(voli/mi)*uzij/((1-pori)+1.0e-15);

	
	
// 			dfx=(1.0/8.0)*Cd*rho_f*3.14159*di*di*uxij*mag_uij*pow(pori,1-beta)/mi;
// 			dfy=(1.0/8.0)*Cd*rho_f*3.14159*di*di*uyij*mag_uij*pow(pori,1-beta)/mi;
// 			dfz=(1.0/8.0)*Cd*rho_f*3.14159*di*di*uzij*mag_uij*pow(pori,1-beta)/mi;

// 		}

		


// 		// Add SPH-DEM coupling force calculation (acting on DEM particles)
// 		P3[i].ftotalx=ftotalx+dfx;
// 		P3[i].ftotaly=ftotaly+dfy;
// 		P3[i].ftotalz=ftotalz+dfz;
	
// 		// Save SPH-DEM coupling force acting on DEM particles ()
// 		P1[i].Fdx_d=dfx;
// 		P1[i].Fdy_d=dfy;
// 		P1[i].Fdz_d=dfz;
// 	}
	
	




// }



////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_SPH_coupling3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;		// leave if it is not SPH particle

	
	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;			
	Real mi,rhoi,pori,hi,flti,cpi,tempi;	
	Real ftotalx,ftotaly,ftotalz;	
	Real dtempi;										
	
	
	
	//Real search_range;
	Real tmpx,tmpy,tmpz,tmpt;			// tmptq : temporary value for torque calculation
	Real search_range,tmp_A;
	


	// position
	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;

	ptypei=P1[i].p_type;


	// mass,density
	mi=P1[i].m;
	rhoi=P1[i].rho;
	pori=P1[i].DEMpor;

	tempi=P1[i].temp;

	// ftotal
	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;
	
	dtempi=P3[i].dtemp;
	cpi=heat_capacity(ptypei,tempi);

	// filter
	flti=P1[i].flt_s;



	//search_range=k_search_kappa*hi;	// search range
	hi=1*P1[i].h;
	tmp_A=calc_tmpA(1.0*hi);
	search_range=k_search_kappa*1.0*hi;	// search range

	
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
	tmpt=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;

				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						

						if (P1[j].p_type>1000){
							Real xj,yj,zj,tdist; 

						
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
							
			
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist<search_range){

								if ((ptypei!=MOVING)&&(ptypei!=-3)&&(ptypei!=0)){

									
									Real twij;
									Real Fdx_bj,Fdy_bj,Fdz_bj,Fdx_dj,Fdy_dj,Fdz_dj,mj,flt_fluidj,flt_airj,flt_fj,Q_sdj;					// SPH-DEM Coupling Force acting on DEM Particle
					
							
									twij=calc_kernel_wij(tmp_A,1.0*hi,tdist);
								
									mj=P1[j].m;
									flt_fluidj=P1[j].flt_sd_2;
									flt_airj=P1[j].flt_s;

									flt_fj=flt_fluidj+flt_airj;

									if((ptypei==1)||(ptypei==3)){


										Fdx_dj=P1[j].Fdx_df;
										Fdy_dj=P1[j].Fdy_df;
										Fdz_dj=P1[j].Fdz_df;

										Q_sdj=P1[j].Q_sd;

										if (flt_fj>0.0){

											
											// tmpx+=-pori*(mj/(rhoi))*Fdx_dj*twij/(flt_fj+1.0e-20);
											// tmpy+=-pori*(mj/(rhoi))*Fdy_dj*twij/(flt_fj+1.0e-20);
											// tmpz+=-pori*(mj/(rhoi))*Fdz_dj*twij/(flt_fj+1.0e-20);

											tmpx+=-(mj*pori/(rhoi))*Fdx_dj*twij/(flt_fj+1.0e-20);
											tmpy+=-(mj*pori/(rhoi))*Fdy_dj*twij/(flt_fj+1.0e-20);
											tmpz+=-(mj*pori/(rhoi))*Fdz_dj*twij/(flt_fj+1.0e-20);

											//tmpx+=-(mj/(rhoi))*Fdx_dj*twij;
											//tmpy+=-(mj/(rhoi))*Fdy_dj*twij;
											//tmpz+=-(mj/(rhoi))*Fdz_dj*twij;
//

											

											tmpt+=-Q_sdj/(rhoi*cpi)*twij/(flt_fj+1.0e-20);
											//tmpt+=(mi/(rhoi*cpi)*twij*Q_sdj/(flt_fj+1.0e-20);



	
											}


									}

								
								



								}



								

							
												

							}
								
	
						}


					}			
				}
			}
		}
	}

	// Add SPH-DEM coupling force calculation (acting on SPH particles)


	
	// tmpx=tmpy=tmpz=0.0;
	// if (P1[i].DEMpor<0.9) tmpz=-100.0;

	// if abs(tmpx)>1.0e5 tmpx=tmpx*1.0e5/abs(tmpx);
	// if abs(tmpy)>1.0e5 tmpy=tmpy*1.0e5/abs(tmpy);
	// if abs(tmpz)>1.0e5 tmpz=tmpz*1.0e5/abs(tmpz);

	P3[i].ftotalx=ftotalx+tmpx;
	P3[i].ftotaly=ftotaly+tmpy;
	P3[i].ftotalz=ftotalz+tmpz;
	

	P3[i].ftotal=sqrt((ftotalx+tmpx)*(ftotalx+tmpx)+(ftotaly+tmpy)*(ftotaly+tmpy)+(ftotalz+tmpz)*(ftotalz+tmpz));
	//P3[i].ftotal=sqrt((ftotalx)*(ftotalx)+(ftotaly)*(ftotaly)+(ftotalz-200.0)*(ftotalz-200.0));

	P1[i].test4=sqrt((ftotalx+tmpx)*(ftotalx+tmpx)+(ftotaly+tmpy)*(ftotaly+tmpy)+(ftotalz+tmpz)*(ftotalz+tmpz));

	P1[i].test7=tmpz;
	//if (zi>0.02 & zi<0.06) P3[i].ftotalz=ftotalz-500.0;
	

	// P3[i].ftotalx=ftotalx;
	// P3[i].ftotaly=ftotaly;
	// P3[i].ftotalz=ftotalz-100;
	

	if(k_con_solve==1)	P3[i].dtemp=dtempi+tmpt;

}


__global__ void KERNEL_SPH_coupling3D_sph(int_t inout,int_t*g_str_sph,int_t*g_end_sph,part1*P1_sph,part3*P3_sph,
											int_t*g_str_dem,int_t*g_end_dem,part1*P1_dem,part3*P3_dem)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1_sph[i].i_type!=inout) return;
	if(P1_sph[i].p_type>1000) return;		// leave if it is not SPH particle

	
	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;			
	Real mi,rhoi,pori,hi,flti,cpi,tempi;	
	Real ftotalx,ftotaly,ftotalz;	
	Real dtempi;										
	
	
	
	//Real search_range;
	Real tmpx,tmpy,tmpz,tmpt;			// tmptq : temporary value for torque calculation
	Real search_range,tmp_A;
	


	// position
	xi=P1_sph[i].x;
	yi=P1_sph[i].y;
	zi=P1_sph[i].z;

	ptypei=P1_sph[i].p_type;


	// mass,density
	mi=P1_sph[i].m;
	rhoi=P1_sph[i].rho;
	pori=P1_sph[i].DEMpor;

	tempi=P1_sph[i].temp;

	// ftotal
	ftotalx=P3_sph[i].ftotalx;
	ftotaly=P3_sph[i].ftotaly;
	ftotalz=P3_sph[i].ftotalz;
	
	dtempi=P3_sph[i].dtemp;
	cpi=heat_capacity(ptypei,tempi);

	// filter
	flti=P1_sph[i].flt_s;



	//search_range=k_search_kappa*hi;	// search range
	hi=1*P1_sph[i].h;
	tmp_A=calc_tmpA(1.0*hi);
	search_range=k_search_kappa*1.0*hi;	// search range

	
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
	tmpt=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);
				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;

				if(g_str_dem[k]!=cu_memset){
					int_t fend=g_end_dem[k];
					for(int_t j=g_str_dem[k];j<fend;j++){
						

						if (P1_dem[j].p_type>1000){
							Real xj,yj,zj,tdist; 

						
							xj=P1_dem[j].x;
							yj=P1_dem[j].y;
							zj=P1_dem[j].z;
							
			
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist<search_range){

								if ((ptypei!=MOVING)&&(ptypei!=-3)&&(ptypei!=0)){

									
									Real twij;
									Real Fdx_bj,Fdy_bj,Fdz_bj,Fdx_dj,Fdy_dj,Fdz_dj,mj,flt_fluidj,flt_airj,flt_fj,Q_sdj;					// SPH-DEM Coupling Force acting on DEM Particle
					
							
									twij=calc_kernel_wij(tmp_A,1.0*hi,tdist);
								
									mj=P1_dem[j].m;
									flt_fluidj=P1_dem[j].flt_sd_2;
									flt_airj=P1_dem[j].flt_s;

									flt_fj=flt_fluidj+flt_airj;

									if((ptypei==1)||(ptypei==3)){


										Fdx_dj=P1_dem[j].Fdx_df;
										Fdy_dj=P1_dem[j].Fdy_df;
										Fdz_dj=P1_dem[j].Fdz_df;

										Q_sdj=P1_dem[j].Q_sd;

										if (flt_fj>0.0){

											
											// tmpx+=-pori*(mj/(rhoi))*Fdx_dj*twij/(flt_fj+1.0e-20);
											// tmpy+=-pori*(mj/(rhoi))*Fdy_dj*twij/(flt_fj+1.0e-20);
											// tmpz+=-pori*(mj/(rhoi))*Fdz_dj*twij/(flt_fj+1.0e-20);

											tmpx+=-(mj*pori/(rhoi))*Fdx_dj*twij/(flt_fj+1.0e-20);
											tmpy+=-(mj*pori/(rhoi))*Fdy_dj*twij/(flt_fj+1.0e-20);
											tmpz+=-(mj*pori/(rhoi))*Fdz_dj*twij/(flt_fj+1.0e-20);

											//tmpx+=-(mj/(rhoi))*Fdx_dj*twij;
											//tmpy+=-(mj/(rhoi))*Fdy_dj*twij;
											//tmpz+=-(mj/(rhoi))*Fdz_dj*twij;
//

											

											tmpt+=-Q_sdj/(rhoi*cpi)*twij/(flt_fj+1.0e-20);
											//tmpt+=(mi/(rhoi*cpi)*twij*Q_sdj/(flt_fj+1.0e-20);



	
											}


									}

								
								



								}



								

							
												

							}
								
	
						}


					}			
				}
			}
		}
	}

	// Add SPH-DEM coupling force calculation (acting on SPH particles)


	
	// tmpx=tmpy=tmpz=0.0;
	// if (P1[i].DEMpor<0.9) tmpz=-100.0;

	// if abs(tmpx)>1.0e5 tmpx=tmpx*1.0e5/abs(tmpx);
	// if abs(tmpy)>1.0e5 tmpy=tmpy*1.0e5/abs(tmpy);
	// if abs(tmpz)>1.0e5 tmpz=tmpz*1.0e5/abs(tmpz);

	P3_sph[i].ftotalx=ftotalx+tmpx;
	P3_sph[i].ftotaly=ftotaly+tmpy;
	P3_sph[i].ftotalz=ftotalz+tmpz;
	

	P3_sph[i].ftotal=sqrt((ftotalx+tmpx)*(ftotalx+tmpx)+(ftotaly+tmpy)*(ftotaly+tmpy)+(ftotalz+tmpz)*(ftotalz+tmpz));
	//P3[i].ftotal=sqrt((ftotalx)*(ftotalx)+(ftotaly)*(ftotaly)+(ftotalz-200.0)*(ftotalz-200.0));

	P1_sph[i].test4=sqrt((ftotalx+tmpx)*(ftotalx+tmpx)+(ftotaly+tmpy)*(ftotaly+tmpy)+(ftotalz+tmpz)*(ftotalz+tmpz));

	P1_sph[i].test7=tmpz;
	//if (zi>0.02 & zi<0.06) P3[i].ftotalz=ftotalz-500.0;
	

	// P3[i].ftotalx=ftotalx;
	// P3[i].ftotaly=ftotaly;
	// P3[i].ftotalz=ftotalz-100;
	

	if(k_con_solve==1)	P3_sph[i].dtemp=dtempi+tmpt;

}
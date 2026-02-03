////////////////////////////////////////////////////////////////////////
__device__ Real calc_tmpA(Real tH)
{
	Real tA=0.0;
	//
	if(k_kernel_type==Gaussian){
		//if(k_dim==1) tA=1.0/(pow(PI,0.5)*tH);
		if(k_dim==2) tA=1.0/(PI*pow(tH,2));
		else if(k_dim==3) tA=1.0/(pow(PI,1.5)*pow(tH,3));
	}else	if(k_kernel_type==Quintic){
		//if(k_dim==1) tA=1.0;
		if(k_dim==2) tA=7.0/(478.0*PI*pow(tH,2));
		else if(k_dim==3) tA=3.0/(359.0*PI*pow(tH,3));
	}else	if(k_kernel_type==Quartic){
		//if(k_dim==1) tA=1.0/tH;
		if(k_dim==2) tA=15.0/(7.0*PI*pow(tH,2));
		else if(k_dim==3) tA=315.0/(208.0*PI*pow(tH,3));
	}else	if(k_kernel_type==Wendland2){
		//if(k_dim==1) tA=1.25/(2*tH); // 5.0/(4*(2h))
		if(k_dim==2) tA=2.228169203286535/(4*tH*tH);	// 7.0/(pi*(2h)^2)
		else if(k_dim==3) tA=3.342253804929802/(8*tH*tH*tH);	// 21.0/(2*pi*(2th)^3)
	}else	if(k_kernel_type==Wendland4){
		//if(k_dim==1) tA=1.5/(2*tH);	// 3.0/(2*(2h))
		if(k_dim==2) tA=2.864788975654116/(4*tH*tH);	// 9.0/(pi*(2h)^2)
		else if(k_dim==3) tA=4.923856051905513/(8*tH*tH*tH);	// 495.0/(32*pi*(2h)^3)
	}else	if(k_kernel_type==Wendland6){
		//if(k_dim==1) tA=1.71875/(2*tH);	// 55.0/(32*(2h))
		if(k_dim==2) tA=3.546881588905096/(4*tH*tH);	// 78.0/(7*pi*(2h)^2)
		else if(k_dim==3) tA=6.788953041263660/(8*tH*tH*tH);	// 1365.0/(64*pi*(2h)^3)
	}
	return tA;
}
////////////////////////////////////////////////////////////////////////
__device__ Real calc_tmP3D(Real tH)
{
	Real tA=0.0;
	//
	if(k_kernel_type==Gaussian)tA=1.0/(pow(PI,1.5)*pow(tH,3));
	else if(k_kernel_type==Quintic) tA=3.0/(359.0*PI*pow(tH,3));
	else if(k_kernel_type==Quartic) tA=315.0/(208.0*PI*pow(tH,3));
	else if(k_kernel_type==Wendland2) tA=3.342253804929802/(8*tH*tH*tH);	// 21.0/(2*pi*(2th)^3)
	else if(k_kernel_type==Wendland4) tA=4.923856051905513/(8*tH*tH*tH);	// 495.0/(32*pi*(2h)^3)
	else if(k_kernel_type==Wendland6) tA=6.788953041263660/(8*tH*tH*tH);	// 1365.0/(64*pi*(2h)^3)
	return tA;
}
////////////////////////////////////////////////////////////////////////
__device__ Real calc_kernel_wij(Real tA,Real tH,Real rr){

	Real tR,wij;
	tR=wij=0.0;

	if(k_kernel_type==Gaussian){
		tR=rr/tH;
		wij=tA*exp(-pow(tR,2));
	}else	if(k_kernel_type==Quintic){
		tR=rr/tH;
		if(tR<1) wij=tA*(pow(3.0-tR,5)-6.0*pow(2.0-tR,5)+15.0*pow(1.0-tR,5));
		else if(1<=tR&&tR<2) wij=tA*(pow(3.0-tR,5)-6.0*pow(2.0-tR,5));
		else if(2<=tR&&tR<3) wij=tA*(pow(3.0-tR,5));
	}else	if(k_kernel_type==Quartic){
		tR=rr/tH;
		wij=(tR<2)*tA*(2.0/3.0-9.0/8.0*pow(tR,2)+19.0/24.0*pow(tR,3)-5.0/32.0*pow(tR,4));
	}else	if(k_kernel_type==Wendland2){
		tR=rr/tH*0.5;
		//if(k_dim==1) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1+3*tR);
		if(k_dim==2) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR);
		else if(k_dim==3) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR);
	}else	if(k_kernel_type==Wendland4){
		tR=rr/tH*0.5;
		//if(k_dim==1) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+5*tR+8*tR*tR);
		if(k_dim==2) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+6*tR+11.666666666666666*tR*tR);
		else if(k_dim==3) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+6*tR+11.666666666666666*tR*tR);
	}else	if(k_kernel_type==Wendland6){
		tR=rr/tH*0.5;
		//if(k_dim==1) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+7*tR+19*tR*tR+21*tR*tR*tR);
		if(k_dim==2) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
		else if(k_dim==3) wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
	}
	return wij;
}
////////////////////////////////////////////////////////////////////////
__device__ Real calc_kernel_wij3D(Real tA,Real tH,Real rr){

	Real tR,wij;
	tR=wij=0.0;

	if(k_kernel_type==Gaussian){
		tR=rr/tH;
		wij=tA*exp(-pow(tR,2));
	}else	if(k_kernel_type==Quintic){
		tR=rr/tH;
		if(tR<1) wij=tA*(pow(3.0-tR,5)-6.0*pow(2.0-tR,5)+15.0*pow(1.0-tR,5));
		else if(1<=tR&&tR<2) wij=tA*(pow(3.0-tR,5)-6.0*pow(2.0-tR,5));
		else if(2<=tR&&tR<3) wij=tA*(pow(3.0-tR,5));
	}else	if(k_kernel_type==Quartic){
		tR=rr/tH;
		wij=(tR<2)*tA*(2.0/3.0-9.0/8.0*pow(tR,2)+19.0/24.0*pow(tR,3)-5.0/32.0*pow(tR,4));
	}else	if(k_kernel_type==Wendland2){
		tR=rr/tH*0.5;
		wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR);
	}else	if(k_kernel_type==Wendland4){
		tR=rr/tH*0.5;
		wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+6*tR+11.666666666666666*tR*tR);
	}else	if(k_kernel_type==Wendland6){
		tR=rr/tH*0.5;
		wij=(tR<1)*tA*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+8*tR+25*tR*tR+32*tR*tR*tR);
	}
	return wij;
}
////////////////////////////////////////////////////////////////////////
__device__ Real calc_kernel_dwij(Real tA,Real tH,Real rr){

	Real tR,dwij;
	tR=dwij=0.0;

	if(k_kernel_type==Gaussian){
		tR=rr/tH;
		dwij=(1.0/tH)*tA*(-2.0)*tR*exp(-pow(tR,2));
	}else	if(k_kernel_type==Quintic){
		tR=rr/tH;
		if(tR<1) dwij=(1.0/tH)*tA*(pow(3.0-tR,5)+30.0*pow(2.0-tR,4)-75.0*pow(1.0-tR,4));
		else if(1<=tR&&tR<2) dwij=(1.0/tH)*tA*(-5.0*pow(3.0-tR,4)+30.0*pow(2.0-tR,4));
		else if(2<=tR&&tR<3) dwij=(1.0/tH)*tA*(-5.0*pow(3.0-tR,4));
	}else	if(k_kernel_type==Quartic){
		tR=rr/tH;
		dwij=(tR<2)*(1.0/tH)*tA*(-9.0/8.0*2*tR+19.0/24.0*3.0*pow(tR,2)-5.0/32.0*4.0*pow(tR,3));
	}else	if(k_kernel_type==Wendland2){
		tR=rr/tH*0.5;
		//if(k_dim==1) dwij=(tR<1)*(1/(2*tH))*tA*(-12*tR*(1-tR)*(1-tR));
		if(k_dim==2) dwij=(tR<1)*(1/(2*tH))*tA*(-20*tR*(1-tR)*(1-tR)*(1-tR));
		else if(k_dim==3) dwij=(tR<1)*(1/(2*tH))*tA*(-20*tR*(1-tR)*(1-tR)*(1-tR));
	}else	if(k_kernel_type==Wendland4){
		tR=rr/tH*0.5;
		//if(k_dim==1) dwij=(tR<1)*tA*(-14*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1+4*tR));
		if(k_dim==2) dwij=(tR<1)*(1/(2*tH))*tA*(-18.666666666666668*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(5*tR+1));
		else if(k_dim==3) dwij=(tR<1)*(1/(2*tH))*tA*(-18.666666666666668*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(5*tR+1));
	}else	if(k_kernel_type==Wendland6){
		tR=rr/tH*0.5;
		//if(k_dim==1) dwij=(tR<1)*(1/(2*tH))*tA*(-6*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(35*tR*tR+18*tR+3));
		if(k_dim==2) dwij=(tR<1)*(1/(2*tH))*tA*(-22*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(16*tR*tR+7*tR+1));
		else if(k_dim==3) dwij=(tR<1)*(1/(2*tH))*tA*(-22*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(16*tR*tR+7*tR+1));
	}
	return dwij;
}
////////////////////////////////////////////////////////////////////////
__device__ Real calc_kernel_dwij3D(Real tA,Real tH,Real rr){

	Real tR,dwij;
	tR=dwij=0.0;

	if(k_kernel_type==Gaussian){
		tR=rr/tH;
		dwij=(1.0/tH)*tA*(-2.0)*tR*exp(-pow(tR,2));
	}else	if(k_kernel_type==Quintic){
		tR=rr/tH;
		if(tR<1) dwij=(1.0/tH)*tA*(pow(3.0-tR,5)+30.0*pow(2.0-tR,4)-75.0*pow(1.0-tR,4));
		else if(1<=tR&&tR<2) dwij=(1.0/tH)*tA*(-5.0*pow(3.0-tR,4)+30.0*pow(2.0-tR,4));
		else if(2<=tR&&tR<3) dwij=(1.0/tH)*tA*(-5.0*pow(3.0-tR,4));
	}else	if(k_kernel_type==Quartic){
		tR=rr/tH;
		dwij=(tR<2)*(1.0/tH)*tA*(-9.0/8.0*2*tR+19.0/24.0*3.0*pow(tR,2)-5.0/32.0*4.0*pow(tR,3));
	}else	if(k_kernel_type==Wendland2){
		tR=rr/tH*0.5;
		dwij=(tR<1)*(1/(2*tH))*tA*(-20*tR*(1-tR)*(1-tR)*(1-tR));
	}else	if(k_kernel_type==Wendland4){
		tR=rr/tH*0.5;
		dwij=(tR<1)*(1/(2*tH))*tA*(-18.666666666666668*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(5*tR+1));
	}else	if(k_kernel_type==Wendland6){
		tR=rr/tH*0.5;
		dwij=(tR<1)*(1/(2*tH))*tA*(-22*tR*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(1-tR)*(16*tR*tR+7*tR+1));
	}
	return dwij;
}
////////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_clc_filter2D_init(int_t*g_str,int_t*g_end,part1*P1)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type==3) return;

// 	int_t icell,jcell;
// 	Real xi,yi;
// 	Real tmp_h,tmp_A,tmp_R,search_range;

// 	xi=P1[i].x;
// 	yi=P1[i].y;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	search_range=k_search_kappa*tmp_h;	// search range

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;

// 	tmp_R=0.0;
// 	for(int_t y=-1;y<=1;y++){
// 		for(int_t x=-1;x<=1;x++){
// 			int_t k=(icell+x)+k_NI*(jcell+y);
// 			if(k<0||k>=k_num_cells-1) continue;
// 			if(g_str[k]!=cu_memset){
// 				int_t fend=g_end[k];
// 				for(int_t j=g_str[k];j<fend;j++){
// 					Real xj,yj,tdist;
// 					xj=P1[j].x;
// 					yj=P1[j].y;

// 					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// 					if(tdist<search_range){
// 						Real tmp_wij,mj,rhoj;
// 						tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
// 						mj=P1[j].m;
// 						rhoj=P1[j].rho;

// 						tmp_R+=mj/rhoj*tmp_wij;
// 					}
// 				}
// 			}
// 		}
// 	}
// 	P1[i].flt_s=tmp_R;
// }
// ////////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_clc_filter3D_init(int_t*g_str,int_t*g_end,part1*P1)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type==3) return;

// 	int_t icell,jcell,kcell;
// 	Real xi,yi,zi;
// 	Real tmp_h,tmp_A,tmp_R,search_range;

// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	search_range=k_search_kappa*tmp_h;	// search range

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

// 	tmp_R=0.0;
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				if(k<0||k>=k_num_cells-1) continue;
// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
// 						Real xj,yj,zj,tdist;
// 						xj=P1[j].x;
// 						yj=P1[j].y;
// 						zj=P1[j].z;

// 						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
// 						if(tdist<search_range){
// 							Real tmp_wij,mj,rhoj;
// 							tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
// 							mj=P1[j].m;
// 							rhoj=P1[j].rho;
// 							tmp_R+=mj/rhoj*tmp_wij;
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	P1[i].flt_s=tmp_R;
// }
// ////////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_clc_filter2D(int_t inout,int_t*g_str,int_t*g_end,part1*P1)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type!=inout) return;

// 	int_t icell,jcell;
// 	Real xi,yi;
// 	Real tmp_h,tmp_A,tmp_R,search_range;

// 	xi=P1[i].x;
// 	yi=P1[i].y;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	search_range=k_search_kappa*tmp_h;	// search range

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;

// 	tmp_R=0.0;
// 	for(int_t y=-1;y<=1;y++){
// 		for(int_t x=-1;x<=1;x++){
// 			int_t k=(icell+x)+k_NI*(jcell+y);
// 			if(k<0||k>=k_num_cells-1) continue;
// 			if(g_str[k]!=cu_memset){
// 				int_t fend=g_end[k];
// 				for(int_t j=g_str[k];j<fend;j++){
// 					Real xj,yj,tdist;
// 					xj=P1[j].x;
// 					yj=P1[j].y;

// 					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// 					if(tdist<search_range){
// 						Real tmp_wij,mj,rhoj;
// 						tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
// 						mj=P1[j].m;
// 						rhoj=P1[j].rho;

// 						tmp_R+=mj/rhoj*tmp_wij;
// 					}
// 				}
// 			}
// 		}
// 	}
// 	P1[i].flt_s=tmp_R;
// }
// ////////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_clc_filter3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type!=inout) return;

// 	int_t icell,jcell,kcell;
// 	Real xi,yi,zi;
// 	Real tmp_h,tmp_A,tmp_R,search_range;

// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	search_range=k_search_kappa*tmp_h;	// search range

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

// 	tmp_R=0.0;
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				if(k<0||k>=k_num_cells-1) continue;
// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
// 						Real xj,yj,zj,tdist;
// 						xj=P1[j].x;
// 						yj=P1[j].y;
// 						zj=P1[j].z;

// 						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
// 						if(tdist<search_range){
// 							Real tmp_wij,mj,rhoj;
// 							tmp_wij=calc_kernel_wij(tmp_A,tmp_h,tdist);
// 							mj=P1[j].m;
// 							rhoj=P1[j].rho;
// 							tmp_R+=mj/rhoj*tmp_wij;
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// 	P1[i].flt_s=tmp_R;
// }
//////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_clc_gradient_correction001_2D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part3*P3)
// {
// 	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type!=inout) return;

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
// 			int_t k=(icell+x)+k_NI*(jcell+y);
// 			if(k<0||k>=k_num_cells-1) continue;
// 			if(g_str[k]!=cu_memset){
// 				int_t fend=g_end[k];
// 				for(int_t j=g_str[k];j<fend;j++){
// 					Real xj,yj,tdist;
// 					xj=P1[j].x;
// 					yj=P1[j].y;

// 					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// 					if(tdist>0&&tdist<search_range){
// 						Real tdwij,mj,rhoj,txx,txy,tyy,rtd,mtd;
// 						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
// 						mj=P1[j].m;
// 						rhoj=P1[j].rho;
// 						mtd=mj*tdwij;
// 						rtd=1.0/(rhoj*tdist);

// 						txx=-mtd*(xi-xj)*(xi-xj);
// 						txx*=rtd;
// 						txy=-mtd*(yi-yj)*(xi-xj);
// 						txy*=rtd;
// 						tyy=-mtd*(yi-yj)*(yi-yj);
// 						tyy*=rtd;

// 						tmpxx+=txx;
// 						tmpxy+=txy;
// 						tmpyy+=tyy;
// 					}
// 				}
// 			}
// 		}
// 	}
// 	// save values to particle array

// 	Real tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
// 	if(abs(tmpcmd)>0){
// 		Real rtcmd=1.0/tmpcmd;
// 		// P3[i].inv_cm_xx=tmpyy*rtcmd;		//P3[i].inv_cm_xx=tmpyy/tmpcmd;
// 		// P3[i].inv_cm_xy=-tmpxy*rtcmd;		//P3[i].inv_cm_xy=-tmpxy/tmpcmd;
// 		// P3[i].inv_cm_yy=tmpxx*rtcmd;		//P3[i].inv_cm_yy=tmpxx/tmpcmd;
// 	}else{
// 		// P3[i].inv_cm_xx=1;
// 		// P3[i].inv_cm_xy=0;
// 		// P3[i].inv_cm_yy=1;
// 	}
// }
// ////////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_clc_gradient_correction001_3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part3*P3)
// {
// 	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type!=inout) return;

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
// 				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				if(k<0||k>=k_num_cells-1) continue;
// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
// 						Real xj,yj,zj,tdist;
// 						xj=P1[j].x;
// 						yj=P1[j].y;
// 						zj=P1[j].z;

// 						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
// 						if(tdist>0&&tdist<search_range){
// 							Real tdwij,mj,rhoj,txx,txy,tyy,tzx,tyz,tzz,rtd,mtd;
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
// 							tzx=-mtd*(xi-xj)*(zi-zj);
// 							tzx*=rtd;
// 							tyz=-mtd*(yi-yj)*(zi-zj);
// 							tyz*=rtd;
// 							tzz=-mtd*(zi-zj)*(zi-zj);
// 							tzz*=rtd;
// 							tmpxx+=txx;
// 							tmpxy+=txy;
// 							tmpyy+=tyy;
// 							tmpzx+=tzx;
// 							tmpyz+=tyz;
// 							tmpzz+=tzz;
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

// 	if(abs(tmpcmd)>0){
// 		Real rtcmd=1.0/tmpcmd;
// 		// P3[i].inv_cm_xx=(tmpyy*tmpzz-tmpyz*tmpyz)*rtcmd;
// 		// P3[i].inv_cm_xy=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
// 		// P3[i].inv_cm_zx=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
// 		// P3[i].inv_cm_yy=(tmpxx*tmpzz-tmpzx*tmpzx)*rtcmd;
// 		// P3[i].inv_cm_yz=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
// 		// P3[i].inv_cm_zz=(tmpxx*tmpyy-tmpxy*tmpxy)*rtcmd;
// 	}
// }
// ////////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_clc_gradient_correction_density001_2D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part3*P3)
// {
// 	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type!=inout) return;

// 	//Real dw_cx,dw_cy,dw_cz,rhoi;
// 	int_t icell,jcell;
// 	Real xi,yi;
// 	Real search_range,tmp_h,tmp_A;
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
// 			int_t k=(icell+x)+k_NI*(jcell+y);
// 			if(k<0||k>=k_num_cells-1) continue;
// 			if(g_str[k]!=cu_memset){
// 				int_t fend=g_end[k];
// 				for(int_t j=g_str[k];j<fend;j++){
// 					Real xj,yj,tdist;
// 					xj=P1[j].x;
// 					yj=P1[j].y;

// 					tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj));
// 					if(tdist>0&&tdist<search_range){
// 						Real tdwij,mj,rhoj,txx,txy,tyy,mtd,rtd;
// 						tdwij=calc_kernel_dwij(tmp_A,tmp_h,tdist);
// 						mj=P1[j].m;
// 						rhoj=P1[j].rho;

// 						mtd=mj*tdwij;
// 						rtd=1.0/(rhoj*tdist);

// 						txx=-mtd*(xi-xj)*(xi-xj);
// 						txx*=rtd;
// 						txy=-mtd*(yi-yj)*(xi-xj);
// 						txy*=rtd;
// 						tyy=-mtd*(yi-yj)*(yi-yj);
// 						tyy*=rtd;
// 						tmpxx+=txx;
// 						tmpxy+=txy;
// 						tmpyy+=tyy;
// 					}
// 				}
// 			}
// 		}
// 	}
// 	// save values to particle array
// 	Real tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;

// 	if(abs(tmpcmd)>0){
// 		Real rtcmd=1.0/tmpcmd;
// 		// P3[i].inv_cm_xx=tmpyy*rtcmd;
// 		// P3[i].inv_cm_xy=-tmpxy*rtcmd;
// 		// P3[i].inv_cm_yy=tmpxx*rtcmd;
// 	}
// }
// ////////////////////////////////////////////////////////////////////////
// __global__ void KERNEL_clc_gradient_correction_density001_3D(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part3*P3)
// {
// 	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type!=inout) return;

// 	//Real dw_cx,dw_cy,dw_cz,rhoi;
// 	int_t icell,jcell,kcell;
// 	Real xi,yi,zi;
// 	Real search_range,tmp_h,tmp_A;
// 	Real tmpxx,tmpyy,tmpzz;
// 	Real tmpxy,tmpyz,tmpzx;

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
// 				int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				if(k<0||k>=k_num_cells-1) continue;
// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
// 						Real xj,yj,zj,tdist;
// 						xj=P1[j].x;
// 						yj=P1[j].y;
// 						zj=P1[j].z;

// 						tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
// 						if(tdist>0&&tdist<search_range){
// 							Real tdwij,mj,rhoj,txx,txy,tyy,tzx,tyz,tzz,mtd,rtd;
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
// 							tzx=-mtd*(xi-xj)*(zi-zj);
// 							tzx*=rtd;
// 							tyz=-mtd*(yi-yj)*(zi-zj);
// 							tyz*=rtd;
// 							tzz=-mtd*(zi-zj)*(zi-zj);
// 							tzz*=rtd;
// 							tmpxx+=txx;
// 							tmpxy+=txy;
// 							tmpyy+=tyy;
// 							tmpzx+=tzx;
// 							tmpyz+=tyz;
// 							tmpzz+=tzz;
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

// 	if(abs(tmpcmd)>0){
// 		Real rtcmd=1.0/tmpcmd;
// 		// P3[i].inv_cm_xx=(tmpyy*tmpzz-tmpyz*tmpyz)*rtcmd;
// 		// P3[i].inv_cm_xy=(tmpzx*tmpyz-tmpxy*tmpzz)*rtcmd;
// 		// P3[i].inv_cm_zx=(tmpxy*tmpyz-tmpzx*tmpyy)*rtcmd;
// 		// P3[i].inv_cm_yy=(tmpxx*tmpzz-tmpzx*tmpzx)*rtcmd;
// 		// P3[i].inv_cm_yz=(tmpzx*tmpxy-tmpxx*tmpyz)*rtcmd;
// 		// P3[i].inv_cm_zz=(tmpxx*tmpyy-tmpxy*tmpxy)*rtcmd;
// 	}
// }


__host__ __device__ void apply_gradient_correction_2D(Real Cm[][Correction_Matrix_Size],Real wij, Real dwx,Real dwy,Real* dwcx,Real* dwcy)
{
	if(k_kgc_solve==KGC) {
		*dwcx=Cm[0][0]*dwx + Cm[0][1]*dwy;
		*dwcy=Cm[1][0]*dwx + Cm[1][1]*dwy;
	}else if(k_kgc_solve==FPM) {
		*dwcx=Cm[1][0]*wij + Cm[1][1]*dwx + Cm[1][2]*dwy;
		*dwcy=Cm[2][0]*wij + Cm[2][1]*dwx + Cm[2][2]*dwy;
	}else if(k_kgc_solve==DFPM) {
		*dwcx=Cm[1][0]*wij + Cm[1][1]*dwx + Cm[1][2]*dwy;
		*dwcy=Cm[2][0]*wij + Cm[2][1]*dwx + Cm[2][2]*dwy;
	}
	// else if(k_kgc_solve==KGF) {
		// *dwcx=Cm[1][0]*wij - Cm[1][1]*wij*(xi-xj) - Cm[1][2]*wij*(yi-yj);
		// *dwcy=Cm[2][0]*wij - Cm[2][1]*wij*(xi-xj) - Cm[2][2]*wij*(yi-yj);
	// }
}


__host__ __device__ void apply_gradient_correction_3D(Real Cm[][Correction_Matrix_Size],Real wij,Real dwx,Real dwy,Real dwz,Real* dwcx,Real* dwcy,Real* dwcz)
{
	if(k_kgc_solve==KGC) {
		*dwcx=Cm[0][0]*dwx + Cm[0][1]*dwy + Cm[0][2]*dwz;
		*dwcy=Cm[1][0]*dwx + Cm[1][1]*dwy + Cm[1][2]*dwz;
		*dwcz=Cm[2][0]*dwx + Cm[2][1]*dwy + Cm[2][2]*dwz;
	}else if(k_kgc_solve==FPM) {
		*dwcx=Cm[1][0]*wij + Cm[1][1]*dwx + Cm[1][2]*dwy + Cm[1][3]*dwz;
		*dwcy=Cm[2][0]*wij + Cm[2][1]*dwx + Cm[2][2]*dwy + Cm[2][3]*dwz;
		*dwcz=Cm[3][0]*wij + Cm[3][1]*dwx + Cm[3][2]*dwy + Cm[3][3]*dwz;
	}
	// else if(k_kgc_solve==DFPM) {
	// 	*dwcx=Cm[1][0]*wij + Cm[1][1]*dwx + Cm[1][2]*dwy + Cm[1][3]*dwz;
	// 	*dwcy=Cm[2][0]*wij + Cm[2][1]*dwx + Cm[2][2]*dwy + Cm[2][3]*dwz;
	// 	*dwcz=Cm[3][0]*wij + Cm[3][1]*dwx + Cm[3][2]*dwy + Cm[3][3]*dwz;
	// }else if(k_kgc_solve==KGF) {
	// 	*dwcx=Cm[1][0]*wij - Cm[1][1]*wij*(xi-xj) - Cm[1][2]*wij*(yi-yj) - Cm[1][3]*wij*(zi-zj);
	// 	*dwcy=Cm[2][0]*wij - Cm[2][1]*wij*(xi-xj) - Cm[2][2]*wij*(yi-yj) - Cm[2][3]*wij*(zi-zj);
	// 	*dwcz=Cm[3][0]*wij - Cm[3][1]*wij*(xi-xj) - Cm[3][2]*wij*(yi-yj) - Cm[3][3]*wij*(zi-zj);
	// }
}
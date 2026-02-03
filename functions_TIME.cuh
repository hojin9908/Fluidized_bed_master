

#define half_space	0.002
#define air_vel		0.01008
#define air_time	3.0
#define dl 			0.0000

__host__ __device__ void penetration_box_3D(Real* xi_, Real* yi_, Real* zi_, Real* uxi_, Real* uyi_, Real* uzi_, Real temp_t, int_t temp_count, int_t tmp_idx)
{

	
	Real F_box;
	Real x_b_max,x_b_min,y_b_max,y_b_min,z_b_max,z_b_min;

	Real xi=*xi_;
	Real yi=*yi_;
	Real zi=*zi_;
	Real uxi=*uxi_;
	Real uyi=*uyi_;
	Real uzi=*uzi_;

	x_b_max=0.125;
	x_b_min=-0.125;
	y_b_max=0.025;
	y_b_min=-0.025;
	if(temp_t<air_time) z_b_max=0.44;
	if(temp_t>=air_time) z_b_max=0.44+air_vel*0.5*(temp_t-air_time);
	//z_b_max=0.44+0.5*air_vel*(temp_t-air_time)*(temp_t>=air_time);
	z_b_min=-0.001;

	if((temp_count%(500)==0)&&(tmp_idx==500)) printf("temp_t=%f\n\nz_b_max=%f", temp_t, z_b_max);


	F_box=fmax(fmax(((xi-x_b_max)*(xi-x_b_min)),((yi-y_b_max)*(yi-y_b_min))),((zi-z_b_max)*(zi-z_b_min)));
	if(F_box<=0) return;


	Real cpx,cpy,cpz;
	Real sgn_x,sgn_y,sgn_z,sgn_m;
	Real nx_box,ny_box,nz_box;
	//contact point
	cpx=fmin(x_b_max,fmax(x_b_min,xi));
	cpy=fmin(y_b_max,fmax(y_b_min,yi));
	cpz=fmin(z_b_max,fmax(z_b_min,zi));

	//if(abs(cpx-xi)<1.0e-8) sgn_x=0.0;

	sgn_x=(cpx-xi)/(abs(cpx-xi)+1e-15);
	sgn_y=(cpy-yi)/(abs(cpy-yi)+1e-15);
	sgn_z=(cpz-zi)/(abs(cpy-zi)+1e-15);

	if(abs(cpx-xi)<1.0e-8) sgn_x=0.0;
	if(abs(cpy-yi)<1.0e-8) sgn_y=0.0;
	if(abs(cpz-zi)<1.0e-8) sgn_z=0.0;


	sgn_m=sqrtf(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z);
	nx_box=sgn_x/(sgn_m+1e-15);
	ny_box=sgn_y/(sgn_m+1e-15);
	nz_box=sgn_z/(sgn_m+1e-15);

	*xi_=cpx;
	*yi_=cpy;
	*zi_=cpz;

	//*uxi_=uxi-2*(uxi*nx_box+uyi*ny_box+uzi*nz_box)*nx_box;
	//*uyi_=uyi-2*(uxi*nx_box+uyi*ny_box+uzi*nz_box)*ny_box;
	//*uzi_=uzi-2*(uxi*nx_box+uyi*ny_box+uzi*nz_box)*nz_box;

	*uxi_=0.0;
	*uyi_=0.0;
	*uzi_=0.0;
}


////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_predictor(Real tdt,Real ttime,part1*P1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	int_t p_typei;
	Real tx0,ty0,tz0,txp,typ,tzp;
	Real tux0,tuy0,tuz0,tuxp,tuyp,tuzp;
	Real twx0,twy0,twz0,twxp,twyp,twzp;
	Real tdux_dt0,tduy_dt0,tduz_dt0;
	Real tdwx_dt0,tdwy_dt0,tdwz_dt0;
	Real t_dt;

	int_t buffer_type=P1[i].buffer_type;

	t_dt=tdt;
	p_typei=P1[i].p_type;

	if(p_typei==MOVING){															// 무빙 플레이트 사용시 속도 변경 m->mm 단위로
		tx0=P1[i].x;															// initial x-directional position
		ty0=P1[i].y;															// initial y-directional position
		tz0=P1[i].z;															// initial z-directional position

		P2[i].x0=tx0;
		P2[i].y0=ty0;
		P2[i].z0=tz0;

		tux0=0.0;
		tuy0=0.0;
		tuz0=0.0;
		if ((ttime>=air_time)&&(tz0<0.1)) tuz0=air_vel;
		if ((ttime>=air_time)&&(tz0>0.1)) tuz0=0.5*air_vel;
		


		txp=tx0+tux0*(t_dt*0.5);									// Predict x-directional position (ux0 : velocity of before time step)
		typ=ty0+tuy0*(t_dt*0.5);									// Predict y-directional position (uy0 : velocity of before time step)
		tzp=tz0+tuz0*(t_dt*0.5);									// Predict z-directional position (ux0 : velocity of before time step)

		P1[i].x=txp;															// Update particle data by predicted x-directional position
		P1[i].y=typ;															// Update particle data by predicted y-directional position
		P1[i].z=tzp;															// Update particle data by predicted z-directional position
		P1[i].ux=tux0;														// Update particle data by predicted x-directional velocity
		P1[i].uy=tuy0;														// Update particle data by predicted y-directional velocity
		P1[i].uz=tuz0;														// Update particle data by predicted z-directional velocity
		P2[i].ux0=tux0;														// Update particle data by predicted x-directional velocity
		P2[i].uy0=tuy0;														// Update particle data by predicted y-directional velocity
		P2[i].uz0=tuz0;														// Update particle data by predicted z-directional velocity

	}else{
		tx0=P1[i].x;															// initial x-directional position
		ty0=P1[i].y;															// initial y-directional position
		tz0=P1[i].z;															// initial z-directional position
		tux0=P1[i].ux;															// initial x-directional position //YHS
		tuy0=P1[i].uy;															// initial y-directional position //YHS
		tuz0=P1[i].uz;	
		
		if(p_typei>=0){
			tux0=P1[i].ux;													// initial x-directional velocity
			tuy0=P1[i].uy;													// initial y-directional velocity
			tuz0=P1[i].uz;													// initial z-directional velocity

			twx0=P1[i].wx;													// initial x-directional angular velocity
			twy0=P1[i].wy;													// initial y-directional angular velocity
			twz0=P1[i].wz;													// initial x-directional angular velocity

			tdux_dt0=P3[i].ftotalx*(buffer_type==0);									// initial x-directional acceleration
			tduy_dt0=P3[i].ftotaly*(buffer_type==0);									// initial y-directional acceleration
			tduz_dt0=P3[i].ftotalz*(buffer_type==0);									// initial z-directional acceleration

			tdwx_dt0=P3[i].torqx*(buffer_type==0);									// initial x-directional angular acceleration
			tdwy_dt0=P3[i].torqy*(buffer_type==0);									// initial y-directional angular acceleration
			tdwz_dt0=P3[i].torqz*(buffer_type==0);									// initial z-directional angular acceleration


			txp=tx0+tux0*(t_dt*0.5)*(p_typei>0)*(P1[i].elix);									// Predict x-directional position (ux0 : velocity of before time step)
			typ=ty0+tuy0*(t_dt*0.5)*(p_typei>0)*(P1[i].eliy);									// Predict y-directional position (uy0 : velocity of before time step)
			tzp=tz0+tuz0*(t_dt*0.5)*(p_typei>0)*(P1[i].eliz);									// Predict z-directional position (ux0 : velocity of before time step)

			tuxp=tux0+tdux_dt0*(t_dt*0.5);						// Predict x-directional velocity (dux_dt0 : acceleration of before time step)
			tuyp=tuy0+tduy_dt0*(t_dt*0.5);						// Predict y-directional velocity (duy_dt0 : acceleration of before time step)
			tuzp=tuz0+tduz_dt0*(t_dt*0.5);						// Predict z-directional velocity (duz_dt0 : acceleration of before time step)

			twxp=twx0+tdwx_dt0*(t_dt*0.5);						// Predict x-directional angular velocity (dwx_dt0 : angular acceleration of before time step)
			twyp=twy0+tdwy_dt0*(t_dt*0.5);						// Predict y-directional angular velocity (dwy_dt0 : angular acceleration of before time step)
			twzp=twz0+tdwz_dt0*(t_dt*0.5);						// Predict z-directional angular velocity (dwz_dt0 : angular acceleration of before time step)

		}else{
			txp=tx0;typ=ty0;tzp=tz0;
			tuxp=P1[i].ux;
			tuyp=P1[i].uy;
			tuzp=P1[i].uz;

			twxp=P1[i].wx;
			twyp=P1[i].wy;
			twzp=P1[i].wz;
		}

		P1[i].x=txp;															// Update particle data by predicted x-directional position
		P1[i].y=typ;															// Update particle data by predicted y-directional position
		P1[i].z=tzp;															// Update particle data by predicted z-directional position
		
		P1[i].ux=tuxp;														// Update particle data by predicted x-directional velocity
		P1[i].uy=tuyp;														// Update particle data by predicted y-directional velocity
		P1[i].uz=tuzp;														// Update particle data by predicted z-directional velocity

		P1[i].wx=twxp;														// Update particle data by predicted x-directional angular velocity
		P1[i].wy=twyp;														// Update particle data by predicted y-directional angular velocity
		P1[i].wz=twzp;														// Update particle data by predicted z-directional angular velocity

		P2[i].x0=tx0;															// update x-directional position
		P2[i].y0=ty0;															// update y-directional position
		P2[i].z0=tz0;															// update z-directional position
		
		P2[i].ux0=tux0;														// update x-directional velocity
		P2[i].uy0=tuy0;														// update y-directional velocity
		P2[i].uz0=tuz0;														// update z-directional velocity
	
		P2[i].wx0=twx0;														// update x-directional angular velocity
		P2[i].wy0=twy0;														// update y-directional angular velocity
		P2[i].wz0=twz0;														// update z-directional angular velocity
	
	}

	// // predict density - KERNEL_clc_predictor_continuity(drho_dt0 : time derivatve of density of before time step)
	// if(k_rho_type==Continuity){
	// 	Real trho=P1[i].rho;
	// 	if(ttime<1.0e-10){
	// 		P1[i].vol0 = P1[i].m/trho;
	// 		P1[i].vol = P1[i].m/trho;
	// 	}	
	// 	P2[i].rho0=trho;
	// 	if (p_typei>0)	P1[i].rho=trho+P3[i].drho*(t_dt*0.5);
		


	// }






	if(ttime==0){
		P2[i].rho_ref = P1[i].rho;
	}	
	if((k_rho_type==Continuity)&&(P1[i].p_type>0)){
		Real trho=P1[i].rho;
		if(ttime<1.0e-10)	P1[i].vol0 = P1[i].m/trho;
		P2[i].rho0=trho;
		P1[i].rho=trho+P3[i].drho*(t_dt*0.5);
	}else{
		Real trho=P1[i].rho;
		if(ttime<1.0e-10)	P1[i].vol0 = (P1[i].h/1.6)*(P1[i].h/1.6)*(P1[i].h/1.6);
		P2[i].rho0=trho;
	}










	// // KERNEL_clc_reference_density
	// P2[i].rho_ref=P1[i].m/pow(P1[i].h/1.5,k_dim);
	//----------------------------------------------------
	// KERNEL_clc_predictor_enthalpy - Update particle data by predicted density
	if(k_con_solve==1){
		Real ttempp=P1[i].temp;
		Real ttemp1p=P1[i].temp1;
		Real ttemp2p=P1[i].temp2;
		Real ttemp3p=P1[i].temp3;

		Real tradp=P1[i].rad;

		P2[i].temp0=ttempp;
		P2[i].temp10=ttemp1p;
		P2[i].temp20=ttemp2p;
		P2[i].temp30=ttemp3p;

		P2[i].rad0=tradp;

	//	Real tenthalpyp=P1[i].enthalpy;
	//	P2[i].enthalpy0=tenthalpyp;
		//
		ttempp+=P3[i].dtemp*(t_dt*0.5);
		ttemp1p+=P3[i].dtemp1*(t_dt*0.5);
		ttemp2p+=P3[i].dtemp2*(t_dt*0.5);
		ttemp3p+=P3[i].dtemp3*(t_dt*0.5);

		tradp+=P3[i].drad*(t_dt*0.5);

		P1[i].temp=ttempp;
		P1[i].temp1=ttemp1p;
		P1[i].temp2=ttemp2p;
		P1[i].temp3=ttemp3p;

		P1[i].rad=tradp;
	//	tenthalpyp+=P3[i].denthalpy*(t_dt*0.5);
	//	P1[i].enthalpy=tenthalpyp;
		//P1[i].temp=htoT(tenthalpyp,p_typei);
	}

	// KERNEL_clc_predictor_concn - Predict concentration (dconcn_dt0 : time derivatve of density of before time step)
	// Update particle data by predicted concentration
	if(k_concn_solve==1){
		Real tconcn=P1[i].concn;
		P2[i].concn0=tconcn;
		P1[i].concn=tconcn+P3[i].dconcn*(t_dt*0.5);
	}

}
////////////////////////////////////////////////////////////////////////


__global__ void KERNEL_clc_predictor_1(int_t tcount, Real tdt,Real ttime,part1*P1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method

	int_t p_typei;
	Real tx0,ty0,tz0,txp,typ,tzp;
	Real tux0,tuy0,tuz0,tuxp,tuyp,tuzp;
	Real tdux_dt0,tduy_dt0,tduz_dt0;
	Real t_dt;

	int_t buffer_type=P1[i].buffer_type;

	t_dt=tdt;
	p_typei=P1[i].p_type;

	if(p_typei==MOVING){
		P2[i].x0=P1[i].x;
		P2[i].y0=P1[i].y;
		P2[i].z0=P1[i].z;

		// P1[i].ux=-0.06*PI*sin(ttime*PI);
		P1[i].ux=k_ball_vel;
		P1[i].uy=0;
		P1[i].uz=0;
		P2[i].ux0=k_ball_vel;
		P2[i].uy0=0;
		P2[i].uz0=0;
	}else{
		tx0=P1[i].x;															// initial x-directional position
		ty0=P1[i].y;															// initial y-directional position
		tz0=P1[i].z;															// initial z-directional position
		tux0=P1[i].ux;															// initial x-directional position //YHS
		tuy0=P1[i].uy;															// initial y-directional position //YHS
		tuz0=P1[i].uz;															// initial z-directional position //YHS
		if(p_typei>=0){
			tux0=P1[i].ux;													// initial x-directional velocity
			tuy0=P1[i].uy;													// initial y-directional velocity
			tuz0=P1[i].uz;													// initial z-directional velocity

			tdux_dt0=P3[i].ftotalx*(buffer_type==0);									// initial x-directional acceleration
			tduy_dt0=P3[i].ftotaly*(buffer_type==0);									// initial y-directional acceleration
			tduz_dt0=P3[i].ftotalz*(buffer_type==0);									// initial z-directional acceleration

			txp=tx0+tux0*(t_dt*0.5)*(p_typei>0)*(P1[i].elix);									// Predict x-directional position (ux0 : velocity of before time step)
			typ=ty0+tuy0*(t_dt*0.5)*(p_typei>0)*(P1[i].eliy);									// Predict y-directional position (uy0 : velocity of before time step)
			tzp=tz0+tuz0*(t_dt*0.5)*(p_typei>0)*(P1[i].eliz);									// Predict z-directional position (ux0 : velocity of before time step)

			tuxp=tux0+tdux_dt0*(t_dt*0.5);						// Predict x-directional velocity (dux_dt0 : acceleration of before time step)
			tuyp=tuy0+tduy_dt0*(t_dt*0.5);						// Predict y-directional velocity (duy_dt0 : acceleration of before time step)
			tuzp=tuz0+tduz_dt0*(t_dt*0.5);						// Predict z-directional velocity (duz_dt0 : acceleration of before time step)
		}else{
			txp=tx0;typ=ty0;tzp=tz0;
			tuxp=tux0;tuyp=tuy0;tuzp=tuz0;
			tuxp=P1[i].ux;
			tuyp=P1[i].uy;
			tuzp=P1[i].uz;
		}

		P1[i].x=txp;															// Update particle data by predicted x-directional position
		P1[i].y=typ;															// Update particle data by predicted y-directional position
		P1[i].z=tzp;															// Update particle data by predicted z-directional position
		P1[i].ux=tuxp;														// Update particle data by predicted x-directional velocity
		P1[i].uy=tuyp;														// Update particle data by predicted y-directional velocity
		P1[i].uz=tuzp;														// Update particle data by predicted z-directional velocity

		P2[i].x0=tx0;															// update x-directional position
		P2[i].y0=ty0;															// update y-directional position
		P2[i].z0=tz0;															// update z-directional position
		P2[i].ux0=tux0;														// update x-directional velocity
		P2[i].uy0=tuy0;														// update y-directional velocity
		P2[i].uz0=tuz0;														// update z-directional velocity
	}

	if(ttime==0){
		P2[i].rho_ref = P1[i].rho;
	}	
	if((k_rho_type==Continuity)&&(P1[i].p_type>0)){
		Real trho=P1[i].rho;
		if(tcount==0)	P1[i].vol0 = P1[i].m/trho;
		P2[i].rho0=trho;
		P1[i].rho=trho+P3[i].drho*(t_dt*0.5);
	}else{
		Real trho=P1[i].rho;
		if(tcount==0)	P1[i].vol0 = (P1[i].h/1.6)*(P1[i].h/1.6)*(P1[i].h/1.6);
		P2[i].rho0=trho;
	}

	if((k_con_solve==1)&&(P1[i].p_type>0)){
		// if(enthalpy_eqn){
		// 	Real tenthalpyp=P1[i].enthalpy;
		// 	P2[i].enthalpy0=tenthalpyp;

		// 	tenthalpyp+=P3[i].denthalpy*(t_dt*0.5)*(P1[i].p_type!=-1);
		// 	P1[i].enthalpy=tenthalpyp;
		// 	P1[i].temp=htoT(tenthalpyp,p_typei);
		// 	}else{
		// 	Real ttemp=P1[i].temp;
		// 	P2[i].temp0=ttemp;
		// 	P1[i].temp=ttemp+P3[i].dtemp*(t_dt*0.5)*(P1[i].p_type!=-1);
		// 	}
	}

	if(k_concn_solve==1){
		Real tconcn=P1[i].concn;
		P2[i].concn0=tconcn;
		P1[i].concn=tconcn+P3[i].dconcn*(t_dt*0.5);
	}

}


// Eulerian time integration function
__global__ void KERNEL_clc_euler_update(int_t inout,const Real tdt,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	Real tux,tuy,tuz;													// velocity
	Real twx,twy,twz;													// angular velocity
	Real tx,ty,tz;														// position
	Real dux_dt,duy_dt,duz_dt;								// accleration (time derivative of velocity)
	Real dwx_dt,dwy_dt,dwz_dt;								// angular accleration (time derivative of angular velocity)

	Real t_dt=tdt;

	int_t p_type_i=P1[i].p_type;

	tx=P1[i].x;															// x-directional initial position
	ty=P1[i].y;															// y-directional initial position
	tz=P1[i].z;															// z-directional initial position

	if(p_type_i>0){
		tux=P1[i].ux;													// x-directinoal initial velocity
		tuy=P1[i].uy;													// y-directinoal initial velocity
		tuz=P1[i].uz;													// z-directional initial velocity

		twx=P1[i].wx;													// x-directinoal initial angular velocity
		twy=P1[i].wy;													// y-directinoal initial angular velocity
		twz=P1[i].wz;													// z-directional initial angular velocity

		dux_dt=P3[i].ftotalx;									// x-directional acceleration
		duy_dt=P3[i].ftotaly;									// y-directional acceleration
		duz_dt=P3[i].ftotalz;									// z-directional acceleration

		dwx_dt=P3[i].torqx;									// x-directional angular acceleration
		dwy_dt=P3[i].torqy;									// y-directional angular acceleration
		dwz_dt=P3[i].torqz;									// z-directional angular acceleration

	}else{
		tux=tuy=tuz=0.0;
		dux_dt=duy_dt=duz_dt=0.0;
		twx=twy=twz=0.0;
		dwx_dt=dwy_dt=dwz_dt=0.0;
	}

	tx+=tux*t_dt;															// calculate x-directional position
	ty+=tuy*t_dt;															// calculate y-directional position
	tz+=tuz*t_dt;															// calculate z-directional position

	tux+=dux_dt*t_dt;													// calculate x-directional velocity
	tuy+=duy_dt*t_dt;													// calculate y-directional velocity
	tuz+=duz_dt*t_dt;													// calculate z-directional velocity

	twx+=dwx_dt*t_dt;													// calculate x-directional angular velocity
	twy+=dwy_dt*t_dt;													// calculate y-directional angular velocity
	twz+=dwz_dt*t_dt;													// calculate z-directional angular velocity


	TP1[i].x=tx;															// update x-directional position
	TP1[i].y=ty;															// update y-directional position
	TP1[i].z=tz;															// update z-directional position
	if((tux*tux+tuy*tuy+tuz*tuz)<k_u_limit*k_u_limit){
		TP1[i].ux=tux;													// update x-directional velocity
		TP1[i].uy=tuy;													// update y-directional velocity
		TP1[i].uz=tuz;													// update z-directional velocity
	}

	TP1[i].wx=twx;													// update x-directional angular velocity
	TP1[i].wy=twy;													// update y-directional angular velocity
	TP1[i].wz=twz;													// update z-directional angular velocity
	//KERNEL_clc_precor_update_continuity ---------------------
	if(k_rho_type==Continuity){TP1[i].rho=P2[i].rho0+P3[i].drho*(t_dt*0.5);}
	else TP1[i].rho=P1[i].rho;

	TP1[i].pres=P1[i].pres;
	TP1[i].flt_s=P1[i].flt_s;
	TP1[i].flt_sd=P1[i].flt_sd;

	TP1[i].m=P1[i].m;
	TP1[i].ri=P1[i].ri;
	TP1[i].rad=P1[i].rad;
	TP1[i].dem_idx=P1[i].dem_idx;

	TP1[i].h=P1[i].h;
	TP1[i].temp=P1[i].temp;
	TP1[i].grad_rhox=P1[i].grad_rhox;
	TP1[i].grad_rhoy=P1[i].grad_rhoy;
	TP1[i].grad_rhoz=P1[i].grad_rhoz;

	TP1[i].Fdx_b=P1[i].Fdx_b;
	TP1[i].Fdy_b=P1[i].Fdy_b;
	TP1[i].Fdz_b=P1[i].Fdz_b;

	TP1[i].Fdx_df=P1[i].Fdx_df;
	TP1[i].Fdy_df=P1[i].Fdy_df;
	TP1[i].Fdz_df=P1[i].Fdz_df;

	TP1[i].Fdx_da=P1[i].Fdx_da;
	TP1[i].Fdy_da=P1[i].Fdy_da;
	TP1[i].Fdz_da=P1[i].Fdz_da;

	TP1[i].pgf_x=P1[i].pgf_x;
	TP1[i].pgf_y=P1[i].pgf_y;
	TP1[i].pgf_z=P1[i].pgf_z;

	TP1[i].DEMpor=P1[i].DEMpor;
	TP1[i].DEMvf=P1[i].DEMvf;

	TP1[i].k_turb=P1[i].k_turb;
	TP1[i].e_turb=P1[i].e_turb;
}
////////////////////////////////////////////////////////////////////////
// corrector step for Predictor-Corrector time integration
__global__ void KERNEL_time_update(int_t inout,const Real tdt,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	Real tx0,ty0,tz0,xc,yc,zc;									// position
	Real tux0,tuy0,tuz0,uxc,uyc,uzc;						// velocity
	Real tux,tuy,tuz;										// velocity
	Real twx0,twy0,twz0,wxc,wyc,wzc;						// angular velocity
	Real twx,twy,twz;										// angular velocity
	
	
	Real dux_dt,duy_dt,duz_dt;									// accleration (time derivative of velocity)
	Real dwx_dt,dwy_dt,dwz_dt;									// angular accleration (time derivative of angular velocity)
	
	Real t_dt=tdt;

	int_t p_type_i=P1[i].p_type;

	if(p_type_i==MOVING){
		tx0=P2[i].x0;														// x-directional initial position
		ty0=P2[i].y0;														// x-directional initial position
		tz0=P2[i].z0;														// x-directional initial position
		tux=P1[i].ux;
		tuy=P1[i].uy;
		tuz=P1[i].uz;

		xc=tx0+tux*(t_dt);												// correct x-directional position
		yc=ty0+tuy*(t_dt);												// correct Y-directional position
		zc=tz0+tuz*(t_dt);												// correct Z-directional position

		//TP1[i].x=tx0;															// update x-directional position
		//TP1[i].y=ty0;															// update y-directional position
		//TP1[i].z=tz0;															// update z-directional position
		TP1[i].x=xc;															// update x-directional position
		TP1[i].y=yc;															// update y-directional position
		TP1[i].z=zc;															// update z-directional position
	}else{
		tx0=P2[i].x0;														// x-directional initial position
		ty0=P2[i].y0;														// x-directional initial position
		tz0=P2[i].z0;														// x-directional initial position
		if(p_type_i>0){
			tux0=P2[i].ux0*(p_type_i>0);						// x-directional initial velocity
			tuy0=P2[i].uy0*(p_type_i>0);						// y-directional initial velocity
			tuz0=P2[i].uz0*(p_type_i>0);						// z-directional initial velocity

			twx0=P2[i].wx0*(p_type_i>0);						// x-directional initial angular velocity
			twy0=P2[i].wy0*(p_type_i>0);						// y-directional initial angular velocity
			twz0=P2[i].wz0*(p_type_i>0);						// z-directional initial angular velocity

			dux_dt=P3[i].ftotalx*(p_type_i>0);			// x-directional acceleration
			duy_dt=P3[i].ftotaly*(p_type_i>0);			// y-directional acceleration
			duz_dt=P3[i].ftotalz*(p_type_i>0);			// z-directional acceleration

			dwx_dt=P3[i].torqx*(p_type_i>0);			// x-directional angular acceleration
			dwy_dt=P3[i].torqy*(p_type_i>0);			// y-directional angular acceleration
			dwz_dt=P3[i].torqz*(p_type_i>0);			// z-directional angular acceleration


		}else{
			tux0=tuy0=tuz0=0.0;
			twx0=twy0=twz0=0.0;
			dux_dt=duy_dt=duz_dt=0.0;
			dwx_dt=dwy_dt=dwz_dt=0.0;
		}

		uxc=tux0+dux_dt*(t_dt);										// correct x-directional velocity
		uyc=tuy0+duy_dt*(t_dt);										// correct y-directional velocity
		uzc=tuz0+duz_dt*(t_dt);										// correct z-directional velocity

		wxc=twx0+dwx_dt*(t_dt);										// correct x-directional angular velocity
		wyc=twy0+dwy_dt*(t_dt);										// correct y-directional angular velocity
		wzc=twz0+dwz_dt*(t_dt);										// correct z-directional angular velocity


		if((uxc*uxc+uyc*uyc+uzc*uzc)>=k_u_limit*k_u_limit){
			uxc=tux0;
			uyc=tuy0;
			uzc=tuz0;
		}
		xc=tx0+uxc*(t_dt);												// correct x-directional position
		yc=ty0+uyc*(t_dt);												// correct Y-directional position
		zc=tz0+uzc*(t_dt);												// correct Z-directional position

		if(!k_xsph_solve){
				TP1[i].x=xc;															// update x-directional position
				TP1[i].y=yc;															// update y-directional position
				TP1[i].z=zc;															// update z-directional position
		}

		TP1[i].ux=uxc;														// update x-directional velocity
		TP1[i].uy=uyc;														// update y-directional velocity
		TP1[i].uz=uzc;														// update z-directional velocity
		
		TP1[i].wx=wxc;														// update x-directional angular velocity
		TP1[i].wy=wyc;														// update y-directional angular velocity
		TP1[i].wz=wzc;														// update z-directional angular velocity


	}

	//KERNEL_clc_precor_update_continuity ---------------------
	if(k_rho_type==Continuity) TP1[i].rho=P2[i].rho0+P3[i].drho*t_dt;
	else TP1[i].rho=P1[i].rho;

	//update_properties_enthalpy-------------------------------
	if(k_con_solve==1) TP1[i].enthalpy=P2[i].enthalpy0+P3[i].denthalpy*t_dt;
	else TP1[i].enthalpy=P1[i].enthalpy;

	//update_properties_concn----------------------------------
	if(k_concn_solve==1) TP1[i].concn=P2[i].concn0+P3[i].dconcn*t_dt;
	else TP1[i].concn=P1[i].concn;

	TP1[i].pres=P1[i].pres;
	TP1[i].flt_s=P1[i].flt_s;
	TP1[i].flt_sd=P1[i].flt_sd;

	TP1[i].m=P1[i].m;
	TP1[i].ri=P1[i].ri;
	TP1[i].rad=P1[i].rad;
	TP1[i].dem_idx=P1[i].dem_idx;
	TP1[i].h=P1[i].h;
	TP1[i].temp=P1[i].temp;
	TP1[i].grad_rhox=P1[i].grad_rhox;
	TP1[i].grad_rhoy=P1[i].grad_rhoy;
	TP1[i].grad_rhoz=P1[i].grad_rhoz;

	TP1[i].Fdx_b=P1[i].Fdx_b;
	TP1[i].Fdy_b=P1[i].Fdy_b;
	TP1[i].Fdz_b=P1[i].Fdz_b;

	TP1[i].Fdx_df=P1[i].Fdx_df;
	TP1[i].Fdy_df=P1[i].Fdy_df;
	TP1[i].Fdz_df=P1[i].Fdz_df;

	TP1[i].Fdx_da=P1[i].Fdx_da;
	TP1[i].Fdy_da=P1[i].Fdy_da;
	TP1[i].Fdz_da=P1[i].Fdz_da;


	TP1[i].pgf_x=P1[i].pgf_x;
	TP1[i].pgf_y=P1[i].pgf_y;
	TP1[i].pgf_z=P1[i].pgf_z;

	TP1[i].DEMpor=P1[i].DEMpor;
	TP1[i].DEMvf=P1[i].DEMvf;

	TP1[i].k_turb=P1[i].k_turb;
	TP1[i].e_turb=P1[i].e_turb;
}
////////////////////////////////////////////////////////////////////////
// corrector step for Predictor-Corrector time integration
__global__ void KERNEL_time_update_single(const Real tdt,part1*P1,part1*TP1,part2*P2,part2*TP2,part3*P3,Real t_time,int_t t_count)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	// if(i>=k_num_part2) return;
	// if(P1[i].i_type!=inout) return;

	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real tx0,ty0,tz0,xc,yc,zc;									// position
	Real tux0,tuy0,tuz0,uxc,uyc,uzc;						// velocity
	Real tux,tuy,tuz;										// velocity

	Real twx0,twy0,twz0,wxc,wyc,wzc;						// angular velocity
	Real twx,twy,twz;										// angular velocity

	Real dux_dt,duy_dt,duz_dt;									// accleration (time derivative of velocity)
	Real dwx_dt,dwy_dt,dwz_dt;									// angular accleration (time derivative of angular velocity)


	Real t_dt=tdt;

	int_t buffer_type=P1[i].buffer_type;
	int_t p_type_i=P1[i].p_type;

	if(p_type_i==MOVING){
		tx0=P2[i].x0;														// x-directional initial position
		ty0=P2[i].y0;														// x-directional initial position
		tz0=P2[i].z0;														// x-directional initial position
		tux=P1[i].ux;
		tuy=P1[i].uy;
		tuz=P1[i].uz;

		xc=tx0+tux*(t_dt);												// correct x-directional position
		yc=ty0+tuy*(t_dt);												// correct Y-directional position
		zc=tz0+tuz*(t_dt);												// correct Z-directional position

		TP1[i].x=xc;															// update x-directional position
		TP1[i].y=yc;															// update y-directional position
		TP1[i].z=zc;															// update z-directional position
	}else{
		tx0=P2[i].x0;														// x-directional initial position
		ty0=P2[i].y0;														// x-directional initial position
		tz0=P2[i].z0;														// x-directional initial position
		if(p_type_i>0){
			tux0=P2[i].ux0;						// x-directional initial velocity
			tuy0=P2[i].uy0;						// y-directional initial velocity
			tuz0=P2[i].uz0;						// z-directional initial velocity

			twx0=P2[i].wx0;						// x-directional initial angular velocity
			twy0=P2[i].wy0;						// y-directional initial angular velocity
			twz0=P2[i].wz0;						// z-directional initial angular velocity

			dux_dt=P3[i].ftotalx*(buffer_type==0);			// x-directional acceleration
			duy_dt=P3[i].ftotaly*(buffer_type==0);			// y-directional acceleration
			duz_dt=P3[i].ftotalz*(buffer_type==0);			// z-directional acceleration

			dwx_dt=P3[i].torqx*(buffer_type==0);			// x-directional angular acceleration
			dwy_dt=P3[i].torqy*(buffer_type==0);			// y-directional angular acceleration
			dwz_dt=P3[i].torqz*(buffer_type==0);			// z-directional angular acceleration

		}else{
			tux0=P2[i].ux0;
			tuy0=P2[i].uy0;
			tuz0=P2[i].uz0;

			twx0=P2[i].wx0;
			twy0=P2[i].wy0;
			twz0=P2[i].wz0;

			dux_dt=duy_dt=duz_dt=0.0;
			dwx_dt=dwy_dt=dwz_dt=0.0;
		}

		uxc=tux0+dux_dt*(t_dt);										// correct x-directional velocity
		uyc=tuy0+duy_dt*(t_dt);										// correct y-directional velocity
		uzc=tuz0+duz_dt*(t_dt);										// correct z-directional velocity

		wxc=twx0+dwx_dt*(t_dt);										// correct x-directional angular velocity
		wyc=twy0+dwy_dt*(t_dt);										// correct y-directional angular velocity
		wzc=twz0+dwz_dt*(t_dt);										// correct z-directional angular velocity


		if((uxc*uxc+uyc*uyc+uzc*uzc)>=k_u_limit*k_u_limit){
			uxc=tux0;
			uyc=tuy0;
			uzc=tuz0;
		}
		xc=tx0+uxc*(t_dt)*(p_type_i>0)*(P1[i].elix);												// correct x-directional position
		yc=ty0+uyc*(t_dt)*(p_type_i>0)*(P1[i].eliy);												// correct Y-directional position
		zc=tz0+uzc*(t_dt)*(p_type_i>0)*(P1[i].eliz);												// correct Z-directional position

		if((p_type_i>0)&(p_type_i<=1000)&(p_type_i!=MOVING)){		
			// penetration box
			if((k_penetration_solve==1)) {
				//if(k_dim==2)penetration_box_2D(&xc, &yc, &uxc, &uyc);
				if(k_dim==3)penetration_box_3D(&xc, &yc, &zc, &uxc, &uyc, &uzc, t_time, t_count,i);
			}
		}

		if(!k_xsph_solve){
				TP1[i].x=xc;															// update x-directional position
				TP1[i].y=yc;															// update y-directional position
				TP1[i].z=zc;															// update z-directional position
		}

		TP1[i].ux=uxc;														// update x-directional velocity
		TP1[i].uy=uyc;														// update y-directional velocity
		TP1[i].uz=uzc;														// update z-directional velocity

		TP1[i].wx=wxc;														// update x-directional angular velocity
		TP1[i].wy=wyc;														// update y-directional angular velocity
		TP1[i].wz=wzc;														// update z-directional angular velocity
	}

	//KERNEL_clc_precor_update_continuity ---------------------
	if(k_rho_type==Continuity&&P1[i].p_type>0) TP1[i].rho=P2[i].rho0+P3[i].drho*t_dt;
	else TP1[i].rho=P1[i].rho;

	//update_properties_enthalpy-------------------------------
	//if(k_con_solve==1) TP1[i].enthalpy=P2[i].enthalpy0+P3[i].denthalpy*t_dt;
	//else TP1[i].enthalpy=P1[i].enthalpy;
	if(k_con_solve==1 * p_type_i>0) {
		TP1[i].temp=P2[i].temp0+P3[i].dtemp*t_dt;
		TP1[i].temp1=P2[i].temp10+P3[i].dtemp1*t_dt;
		TP1[i].temp2=P2[i].temp20+P3[i].dtemp2*t_dt;
		TP1[i].temp3=P2[i].temp30+P3[i].dtemp3*t_dt;
		TP1[i].rad=P2[i].rad0+P3[i].drad*t_dt;
		TP1[i].vol_power=P1[i].vol_power;
		
	}
	else {
		
		TP1[i].temp=P1[i].temp;
		TP1[i].temp1=P1[i].temp1;
		TP1[i].temp2=P1[i].temp2;
		TP1[i].temp3=P1[i].temp3;

		TP1[i].rad=P1[i].rad;
	}
	

	//update_properties_concn----------------------------------
	if(k_concn_solve==1) TP1[i].concn=P2[i].concn0+P3[i].dconcn*t_dt;
	else TP1[i].concn=P1[i].concn;

	TP1[i].pres=P1[i].pres;
	TP1[i].pres_ipp=P1[i].pres_ipp;
	TP1[i].flt_s=P1[i].flt_s;
	TP1[i].flt_sd=P1[i].flt_sd;
	TP1[i].flt_sd_2=P1[i].flt_sd_2;
	TP1[i].Q_sd=P1[i].Q_sd;

	TP1[i].m=P1[i].m;
	TP1[i].ri=P1[i].ri;
	//TP1[i].rad=P1[i].rad;
	TP1[i].dem_idx=P1[i].dem_idx;
	TP1[i].h=P1[i].h;
	//TP1[i].temp=P1[i].temp;
	TP1[i].grad_rhox=P1[i].grad_rhox;
	TP1[i].grad_rhoy=P1[i].grad_rhoy;
	TP1[i].grad_rhoz=P1[i].grad_rhoz;

	TP1[i].Fdx_b=P1[i].Fdx_b;
	TP1[i].Fdy_b=P1[i].Fdy_b;
	TP1[i].Fdz_b=P1[i].Fdz_b;

	TP1[i].Fdx_da=P1[i].Fdx_da;
	TP1[i].Fdy_da=P1[i].Fdy_da;
	TP1[i].Fdz_da=P1[i].Fdz_da;

	TP1[i].Fdx_df=P1[i].Fdx_df;
	TP1[i].Fdy_df=P1[i].Fdy_df;
	TP1[i].Fdz_df=P1[i].Fdz_df;

	TP1[i].pgf_x=P1[i].pgf_x;
	TP1[i].pgf_y=P1[i].pgf_y;
	TP1[i].pgf_z=P1[i].pgf_z;

	TP1[i].DEMpor=P1[i].DEMpor;
	TP1[i].DEMvf=P1[i].DEMvf;

	TP1[i].k_turb=P1[i].k_turb;
	TP1[i].e_turb=P1[i].e_turb;

	TP1[i].test1=P1[i].test1;
	TP1[i].test2=P1[i].test2;
	TP1[i].test3=P1[i].test3;
	TP1[i].test4=P1[i].test4;
	TP1[i].test5=P1[i].test5;
	TP1[i].test6=P1[i].test6;
	TP1[i].test7=P1[i].test7;

	TP1[i].elix=P1[i].elix;
	TP1[i].eliy=P1[i].eliy;
	TP1[i].eliz=P1[i].eliz;
	TP1[i].vol=P1[i].vol;
	TP1[i].vol0=P1[i].vol0;

	TP2[i].rho_ref=P2[i].rho_ref;

}

// corrector step for Predictor-Corrector time integration
__global__ void KERNEL_time_update_single_1(const Real tdt,part1*P1,part1*TP1,part2*P2,part2*TP2,part3*P3)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	// if(i>=k_num_part3) return;
	// if(P1[i].i_type!=inout) return;

	if(i>=k_num_part2) return;
	if(P1[i].i_type>i_type_crt) return;
	if(P1[i].p_type>=1000)	return;		// Immersed Boundary Method

	Real tx0,ty0,tz0,xc,yc,zc;									// position
	Real tux0,tuy0,tuz0,uxc,uyc,uzc;						// velocity
	Real tux,tuy,tuz;														// velocity
	Real dux_dt,duy_dt,duz_dt;									// accleration (time derivative of velocity)
	Real t_dt=tdt;

	int_t buffer_type=P1[i].buffer_type;
	int_t p_type_i=P1[i].p_type;

	if(p_type_i==MOVING){
		tx0=P2[i].x0;														// x-directional initial position
		ty0=P2[i].y0;														// x-directional initial position
		tz0=P2[i].z0;														// x-directional initial position
		tux=P2[i].ux0;
		tuy=P2[i].uy0;
		tuz=P2[i].uz0;

		// xc=tx0+tux*(t_dt);												// correct x-directional position
		// yc=ty0+tuy*(t_dt);												// correct Y-directional position
		// zc=tz0+tuz*(t_dt);												// correct Z-directional position

		TP1[i].x=xc;															// update x-directional position
		TP1[i].y=yc;															// update y-directional position
		TP1[i].z=zc;															// update z-directional position
		TP1[i].ux=tux;														// update x-directional velocity
		TP1[i].uy=tuy;														// update y-directional velocity
		TP1[i].uz=tuz;														// update z-directional velocity

	}else{
		tx0=P2[i].x0;														// x-directional initial position
		ty0=P2[i].y0;														// x-directional initial position
		tz0=P2[i].z0;														// x-directional initial position
		if(p_type_i>0){
			tux0=P2[i].ux0;						// x-directional initial velocity
			tuy0=P2[i].uy0;						// y-directional initial velocity
			tuz0=P2[i].uz0;						// z-directional initial velocity

			dux_dt=P3[i].ftotalx*(buffer_type==0);			// x-directional acceleration
			duy_dt=P3[i].ftotaly*(buffer_type==0);			// y-directional acceleration
			duz_dt=P3[i].ftotalz*(buffer_type==0);			// z-directional acceleration
		}else{
			tux0=P2[i].ux0;
			tuy0=P2[i].uy0;
			tuz0=P2[i].uz0;
			dux_dt=duy_dt=duz_dt=0.0;
		}

		uxc=tux0+dux_dt*(t_dt);										// correct x-directional velocity
		uyc=tuy0+duy_dt*(t_dt);										// correct y-directional velocity
		uzc=tuz0+duz_dt*(t_dt);										// correct z-directional velocity

		if((uxc*uxc+uyc*uyc+uzc*uzc)>=k_u_limit*k_u_limit){
			uxc=tux0;
			uyc=tuy0;
			uzc=tuz0;
		}
		xc=tx0+uxc*(t_dt)*(p_type_i>0)*(P1[i].elix);												// correct x-directional position
		yc=ty0+uyc*(t_dt)*(p_type_i>0)*(P1[i].eliy);												// correct Y-directional position
		zc=tz0+uzc*(t_dt)*(p_type_i>0)*(P1[i].eliz);												// correct Z-directional position

		if(!k_xsph_solve){
				TP1[i].x=xc;															// update x-directional position
				TP1[i].y=yc;															// update y-directional position
				TP1[i].z=zc;															// update z-directional position
		}

		TP1[i].ux=uxc;														// update x-directional velocity
		TP1[i].uy=uyc;														// update y-directional velocity
		TP1[i].uz=uzc;														// update z-directional velocity
	}

	//KERNEL_clc_precor_update_continuity ---------------------
	if((k_rho_type==Continuity)&&(P1[i].p_type>0)) TP1[i].rho=P2[i].rho0+P3[i].drho*t_dt;
	else TP1[i].rho=P1[i].rho;

	//update_properties_enthalpy-------------------------------
	if((k_con_solve==1)&&(P1[i].p_type>0)){
		// if(enthalpy_eqn){
		// 	TP1[i].enthalpy=P2[i].enthalpy0+P3[i].denthalpy*t_dt*(P1[i].p_type!=-1);
		// 	TP1[i].temp=P1[i].temp;
		// }else{
		// 	TP1[i].temp=P2[i].temp0+P3[i].dtemp*t_dt*(P1[i].p_type!=-1);
		// }
	}else{
		TP1[i].enthalpy=P1[i].enthalpy;
		TP1[i].temp=P1[i].temp;
	}

	//update_properties_concn----------------------------------
	if(k_concn_solve==1) TP1[i].concn=P2[i].concn0+P3[i].dconcn*t_dt;
	else TP1[i].concn=P1[i].concn;

	// TP1[i].pres=P1[i].pres;
	// TP1[i].flt_s=P1[i].flt_s;
	// TP1[i].m=P1[i].m;
	// TP1[i].ncell=P1[i].ncell;
	// TP1[i].h=P1[i].h;
	// TP1[i].grad_rhox=P1[i].grad_rhox;
	// TP1[i].grad_rhoy=P1[i].grad_rhoy;
	// TP1[i].grad_rhoz=P1[i].grad_rhoz;
	// TP1[i].k_turb=P1[i].k_turb;
	// TP1[i].e_turb=P1[i].e_turb;

	// TP1[i].elix=P1[i].elix;
	// TP1[i].eliy=P1[i].eliy;
	// TP1[i].shiftx=P1[i].shiftx;
	// TP1[i].shifty=P1[i].shifty;
	// TP1[i].vortx=P1[i].vortx;
	// TP1[i].vorty=P1[i].vorty;
	// TP1[i].vortz=P1[i].vortz;
	// TP1[i].vol=P1[i].vol;
	// TP1[i].vol0=P1[i].vol0;
	// TP2[i].rho_ref=P2[i].rho_ref;
	// TP1[i].flt_s=P1[i].flt_s;
	// TP1[i].fcx = P1[i].fcx;
	// TP1[i].fcy = P1[i].fcy;
	// TP1[i].pos=P1[i].pos;
	// TP1[i].concentration=P1[i].concentration;

	// TP1[i].virialx=P1[i].virialx;
	// TP1[i].virialy=P1[i].virialy;
	// TP1[i].virialz=P1[i].virialz;


	TP1[i].pres=P1[i].pres;
	TP1[i].pres_ipp=P1[i].pres_ipp;
	TP1[i].flt_s=P1[i].flt_s;
	TP1[i].flt_sd=P1[i].flt_sd;
	TP1[i].Q_sd=P1[i].Q_sd;

	TP1[i].m=P1[i].m;
	TP1[i].ri=P1[i].ri;
	//TP1[i].rad=P1[i].rad;
	TP1[i].dem_idx=P1[i].dem_idx;
	TP1[i].h=P1[i].h;
	//TP1[i].temp=P1[i].temp;
	TP1[i].grad_rhox=P1[i].grad_rhox;
	TP1[i].grad_rhoy=P1[i].grad_rhoy;
	TP1[i].grad_rhoz=P1[i].grad_rhoz;

	TP1[i].Fdx_b=P1[i].Fdx_b;
	TP1[i].Fdy_b=P1[i].Fdy_b;
	TP1[i].Fdz_b=P1[i].Fdz_b;

	TP1[i].Fdx_da=P1[i].Fdx_da;
	TP1[i].Fdy_da=P1[i].Fdy_da;
	TP1[i].Fdz_da=P1[i].Fdz_da;

	TP1[i].Fdx_df=P1[i].Fdx_df;
	TP1[i].Fdy_df=P1[i].Fdy_df;
	TP1[i].Fdz_df=P1[i].Fdz_df;

	TP1[i].pgf_x=P1[i].pgf_x;
	TP1[i].pgf_y=P1[i].pgf_y;
	TP1[i].pgf_z=P1[i].pgf_z;

	TP1[i].DEMpor=P1[i].DEMpor;
	TP1[i].DEMvf=P1[i].DEMvf;

	TP1[i].k_turb=P1[i].k_turb;
	TP1[i].e_turb=P1[i].e_turb;

	TP1[i].test1=P1[i].test1;
	TP1[i].test2=P1[i].test2;
	TP1[i].test3=P1[i].test3;
	TP1[i].test4=P1[i].test4;
	TP1[i].test5=P1[i].test5;
	TP1[i].test6=P1[i].test6;
	TP1[i].test7=P1[i].test7;

	TP1[i].elix=P1[i].elix;
	TP1[i].eliy=P1[i].eliy;
	TP1[i].eliz=P1[i].eliz;
	TP1[i].vol=P1[i].vol;
	TP1[i].vol0=P1[i].vol0;

	TP2[i].rho_ref=P2[i].rho_ref;


}

__host__ __device__ int clc_idx_insert1(Real min_, Real xc, Real yc, Real zc, Real space_, int Nsx_, int Nsz_)
{
	int Ix, Iy, Iz;
	int idx_1D;
	int buf_size=(k_num_part2-k_num_part)/18;


	Ix = (int)((xc-min_)/space_);
	Iy = (int)((yc-k_y_min)/space_);
	Iz = (int)((zc-k_z_min)/space_);

	idx_1D=(k_num_part2-1)-(Iy+Nsx_*Iz+Nsx_*Nsz_*Ix);

	return idx_1D;
}

__host__ __device__ int clc_idx_insert2(Real min_, Real xc, Real yc, Real zc, Real space_, int Nsx_, int Nsz_)
{
	int Ix, Iy, Iz;
	int idx_1D;
	int buf_size=(k_num_part2-k_num_part)/18;

	Ix = (int)((xc-min_)/space_);
	Iy = (int)((yc-k_y_min)/space_);
	Iz = (int)((zc-k_z_min)/space_);

	idx_1D=(k_num_part2-1)-(2*buf_size+(Iy+Nsx_*Iz+Nsx_*Nsz_*Ix));

	return idx_1D;
}

__host__ __device__ int clc_idx_insert3(Real min_, Real xc, Real yc, Real zc, Real space_, int Nsx_, int Nsz_)
{
	int Ix, Iy, Iz;
	int idx_1D;
	int buf_size=(k_num_part2-k_num_part)/18;

	Ix = (int)((xc-k_x_min)/space_);
	Iy = (int)((yc-min_)/space_);
	Iz = (int)((zc-k_z_min)/space_);

	
	idx_1D=(k_num_part2-1)-(buf_size*4+(Ix+Nsx_*Iz+Nsx_*Nsz_*Iy));

	return idx_1D;
}

__host__ __device__ int clc_idx_insert4(Real min_, Real xc, Real yc, Real zc, Real space_, int Nsx_, int Nsz_)
{
	int Ix, Iy, Iz;
	int idx_1D;
	int buf_size=(k_num_part2-k_num_part)/18;

	Ix = (int)((xc-k_x_min)/space_);
	Iy = (int)((yc-min_)/space_);
	Iz = (int)((zc-k_z_min)/space_);

	idx_1D=(k_num_part2-1)-(buf_size*6+(Ix+Nsx_*Iz+Nsx_*Nsz_*Iy));
	return idx_1D;
}

__host__ __device__ int clc_idx_insert5(Real min_, Real xc, Real yc, Real zc, Real space_, int Nsx_, int Nsz_)
{
	int Ix, Iy, Iz;
	int idx_1D;
	int buf_size=(k_num_part2-k_num_part)/18;

	Ix = (int)((xc-min_)/space_);
	Iy = (int)((yc-k_y_min)/space_);
	Iz = (int)((zc-k_z_min)/space_);

	idx_1D=(k_num_part2-1)-(buf_size*8+(Iy+Nsx_*Iz+Nsx_*Nsz_*Ix));

	return idx_1D;
}

__host__ __device__ int clc_idx_insert6(Real min_, Real xc, Real yc, Real zc, Real space_, int Nsx_, int Nsz_)
{
	int Ix, Iy, Iz;
	int idx_1D;
	int buf_size=(k_num_part2-k_num_part)/18;

	Ix = (int)((xc-min_)/space_);
	Iy = (int)((yc-k_y_min)/space_);
	Iz = (int)((zc-k_z_min)/space_);

	idx_1D=(k_num_part2-1)-(buf_size*8+(Iy+Nsx_*Iz+Nsx_*Nsz_*Ix));

	return idx_1D;
}


__host__ __device__ int clc_idx_insert7(Real min_, Real xc, Real yc, Real zc, Real space_, int Nsx_, int Nsz_)
{
	int Ix, Iy, Iz;
	int idx_1D;
	int buf_size=(k_num_part2-k_num_part)/18;

	Ix = (int)((xc+0.333*(L1-L2)-k_x_min)/space_);
	Iy = (int)((yc-min_)/space_);
	Iz = (int)((zc-k_z_min)/space_);

	idx_1D=(k_num_part2-1)-(buf_size*8+(Ix+Nsx_*Iz+Nsx_*Nsz_*Iy));
	return idx_1D;
}

__host__ __device__ int clc_idx_insert8(Real min_, Real xc, Real yc, Real zc, Real space_, int Nsx_, int Nsz_)
{
	int Ix, Iy, Iz;
	int idx_1D;
	int buf_size=(k_num_part2-k_num_part)/18;

	Ix = (int)((xc-0.333*(L1-L2)-k_x_min)/space_);
	Iy = (int)((yc-min_)/space_);
	Iz = (int)((zc-k_z_min)/space_);

	idx_1D=(k_num_part2-1)-(buf_size*8+(Ix+Nsx_*Iz+Nsx_*Nsz_*Iy));
	return idx_1D;
}
////////////////////////////////////////////////////////////////////////
// open_boundary
__global__ void KERNEL_open_boundary(const Real tdt,part1*P1,part1*TP1,part2*P2,part3*P3,Real space_,int Nsx_, int Nsz_)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type>2) return;
	if(P1[i].p_type<1) return;

	Real tx0,ty0,tz0,xc,yc,zc;									// position
	Real tux0,tuy0,tuz0,uxc,uyc,uzc;						// velocity
	Real t_dt=tdt;
	int_t idx_insert;

	int_t p_type=TP1[i].p_type;
	int_t i_type=TP1[i].i_type;
	int_t buffer_type=TP1[i].buffer_type;

	xc=TP1[i].x;												// correct x-directional position
	yc=TP1[i].y;												// correct Y-directional position
	zc=TP1[i].z;												// correct Z-directional position

	uxc=TP1[i].ux;
	uyc=TP1[i].uy;

	// if ((buffer_type==0)&&(i_type==1))
	// {
	// 	if (zc>L3)
	// 	{
	// 		TP1[i].i_type=2;
	// 		TP1[i].buffer_type=2;
	// 	}
	// }

	// if ((buffer_type==2)&&(i_type==2)){
	// 	if (zc>L4) {
	// 		TP1[i].i_type=3;
	// 		TP1[i].buffer_type=0;
	// 	}
	// }

	if (i_type==2) {
		TP1[i].i_type=3;
	}


	if ((xc<(L2))&&(abs(yc)<=L1)&&(uxc<=0)&&(i_type==1)&&(p_type>1000)) {
		TP1[i].x=xc+(L1-L2);
	}

	if ((xc>(L1))&&(abs(yc)<=L1)&&(uxc>=0)&&(i_type==1)&&(p_type>1000)) {
		TP1[i].x=xc-(L1-L2);
	}

	if ((yc>(L1))&&(abs(xc)<=L1)&&(uyc>=0)&&(i_type==1)&&(p_type>1000)) {
		TP1[i].y=yc-(L1-L2);
	}

	if ((yc<(L2))&&(abs(xc)<=L1)&&(uyc<=0)&&(i_type==1)&&(p_type>1000)) {
		TP1[i].y=yc+(L1-L2);
	}

	// if ((xc<=(L2))&&(abs(yc)<L1)&&(i_type==1)&&(p_type>1000)) {
	// 	TP1[i].x=xc+(L1-L2);
	// }

	// if ((xc>=(L1))&&(abs(yc)<L1)&&(i_type==1)&&(p_type>1000)) {
	// 	TP1[i].x=xc-(L1-L2);
	// }

	// if ((yc>=(L1))&&(abs(xc)<L1)&&(i_type==1)&&(p_type>1000)) {
	// 	TP1[i].y=yc-(L1-L2);
	// }

	// if ((yc<=(L2))&&(abs(xc)<L1)&&(i_type==1)&&(p_type>1000)) {
	// 	TP1[i].y=yc+(L1-L2);
	// }
	if ((xc<(L2))&&(yc>L1)&&(i_type==1)&&(p_type>1000)) {
		TP1[i].x=xc+(L1-L2);
		TP1[i].y=yc-(L1-L2);
	}

	if ((xc>(L1))&&(yc<L2)&&(i_type==1)&&(p_type>1000)) {
		TP1[i].x=xc-(L1-L2);
		TP1[i].y=yc+(L1-L2);
	}

	if ((xc<(L2))&&(yc<L2)&&(i_type==1)&&(p_type>1000)) {
		TP1[i].x=xc+(L1-L2);
		TP1[i].y=yc+(L1-L2);
	}


	if ((xc>(L1))&&(yc>L1)&&(i_type==1)&&(p_type>1000)) {
		TP1[i].x=xc-(L1-L2);
		TP1[i].y=yc-(L1-L2);
	}

	// if ((xc>(L1+DP))&&(i_type==1)&&(p_type>1000)){
	// 	TP1[i].i_type=3;
	// }

	// if ((xc<(L2-DP))&&(i_type==1)&&(p_type>1000)){
	// 	TP1[i].i_type=3;
	// }

	// if ((yc>(L1+DP))&&(i_type==1)&&(p_type>1000)){
	// 	TP1[i].i_type=3;
	// }

	// if ((yc<(L2-DP))&&(i_type==1)&&(p_type>1000)){
	// 	TP1[i].i_type=3;
	// }



	// inlet
	if ((xc<=(L2+2.4*DP))&&(xc>=(L2))&&(i_type==1)&&(p_type>1000)) {
		//if (xc>L2) {

			Real t_min=L2;


			P1[i]=TP1[i];



			idx_insert=clc_idx_insert1(t_min, xc,yc,zc,space_,Nsx_,Nsz_);

			TP1[idx_insert]=P1[i];


			TP1[idx_insert].x=xc+(L1-L2-dl);
			TP1[idx_insert].y=yc;
			TP1[idx_insert].z=zc;

			TP1[idx_insert].i_type=2;
			//TP1[i].i_type=2;
			//TP1[i].buffer_type=0;

		//}./
	}

	if ((xc>=(L1-2.4*DP))&&(xc<=(L1))&&(i_type==1)&&(p_type>1000)) {
		//if (xc>L2) {

			Real t_min=L1-2.4*DP;
			
			P1[i]=TP1[i];



			idx_insert=clc_idx_insert2(t_min, xc,yc,zc,space_,Nsx_,Nsz_);

			TP1[idx_insert]=P1[i];

			TP1[idx_insert].x=xc-(L1-L2-dl);
			TP1[idx_insert].y=yc;
			TP1[idx_insert].z=zc;

			TP1[idx_insert].i_type=2;
			//TP1[i].i_type=2;
			//TP1[i].buffer_type=0;

		//}
	}

	if ((yc<=(L2+2.4*DP))&&(yc>=(L2))&&(i_type==1)&&(p_type>1000)) {
		//if (xc>L2) {

			Real t_min=L2;
			
			P1[i]=TP1[i];



			idx_insert=clc_idx_insert3(t_min, xc,yc,zc,space_,Nsx_,Nsz_);

			TP1[idx_insert]=P1[i];

			TP1[idx_insert].x=xc;
			TP1[idx_insert].y=yc+(L1-L2-dl);
			TP1[idx_insert].z=zc;

			TP1[idx_insert].i_type=2;
			//TP1[i].i_type=2;
			//TP1[i].buffer_type=0;

		//}
	}

	if ((yc>=(L1-2.4*DP))&&(yc<=(L1))&&(i_type==1)&&(p_type>1000)) {
		//if (xc>L2) {


			Real t_min=L1-2.4*DP;


			P1[i]=TP1[i];



			idx_insert=clc_idx_insert4(t_min, xc,yc,zc,space_,Nsx_,Nsz_);

			TP1[idx_insert]=P1[i];

			TP1[idx_insert].x=xc;
			TP1[idx_insert].y=yc-(L1-L2-dl);
			TP1[idx_insert].z=zc;

			TP1[idx_insert].i_type=2;
			//TP1[i].i_type=2;
			//TP1[i].buffer_type=0;

		//}
	}

	if ((yc>=(L1-2.4*DP))&&(yc<=(L1))&&(xc<=(L2+2.4*DP))&&(xc>=(L2))&&(i_type==1)&&(p_type>1000)) {
		//if (xc>L2) {


			Real t_min=L2;

			P1[i]=TP1[i];



			idx_insert=clc_idx_insert5(t_min, xc,yc,zc,space_,Nsx_,Nsz_);

			TP1[idx_insert]=P1[i];

			TP1[idx_insert].x=xc+(L1-L2-dl);
			TP1[idx_insert].y=yc-(L1-L2-dl);
			TP1[idx_insert].z=zc;

			TP1[idx_insert].i_type=2;
			//TP1[i].i_type=2;
			//TP1[i].buffer_type=0;

		//}
	}

	
	if ((yc<=(L2+2.4*DP))&&(yc>=(L2))&&(xc>=(L1-2.4*DP))&&(xc<=(L1))&&(i_type==1)&&(p_type>1000)) {
		//if (xc>L2) {

			Real t_min=L1-2.4*DP;

			P1[i]=TP1[i];



			idx_insert=clc_idx_insert6(t_min, xc,yc,zc,space_,Nsx_,Nsz_);

			TP1[idx_insert]=P1[i];

			TP1[idx_insert].x=xc-(L1-L2-dl);
			TP1[idx_insert].y=yc+(L1-L2-dl);
			TP1[idx_insert].z=zc;

			TP1[idx_insert].i_type=2;
			//TP1[i].i_type=2;
			//TP1[i].buffer_type=0;

		//}
	}
	
	if ((yc<=(L2+2.4*DP))&&(yc>=(L2))&&(xc<=(L2+2.4*DP))&&(xc>=(L2))&&(i_type==1)&&(p_type>1000)) {
		//if (xc>L2) {

			Real t_min=L2;

			P1[i]=TP1[i];



			idx_insert=clc_idx_insert7(t_min, xc,yc,zc,space_,Nsx_,Nsz_);

			TP1[idx_insert]=P1[i];

			TP1[idx_insert].x=xc+(L1-L2-dl);
			TP1[idx_insert].y=yc+(L1-L2-dl);
			TP1[idx_insert].z=zc;

			TP1[idx_insert].i_type=2;
			//TP1[i].i_type=2;
			//TP1[i].buffer_type=0;

		//}
	}

	if ((yc>=(L1-2.4*DP))&&(yc<=(L1))&&(xc>=(L1-2.4*DP))&&(xc<=(L1))&&(i_type==1)&&(p_type>1000)) {
		//if (xc>L2) {

			Real t_min=L1-2.4*DP;

			P1[i]=TP1[i];



			idx_insert=clc_idx_insert8(t_min, xc,yc,zc,space_,Nsx_,Nsz_);

			TP1[idx_insert]=P1[i];

			TP1[idx_insert].x=xc-(L1-L2-dl);
			TP1[idx_insert].y=yc-(L1-L2-dl);
			TP1[idx_insert].z=zc;

			TP1[idx_insert].i_type=2;
			//TP1[i].i_type=2;
			//TP1[i].buffer_type=0;

		//}
	}
	//
	// if (zc>L3)
	// {
	// 	TP1[i].buffer_type=2;
	// 	TP1[i].ux=0;
	// 	TP1[i].uy=0;
	// }

	// if (zc>L3)
	// {
	// 	TP1[i].i_type=3;
	// }

}
////////////////////////////////////////////////////////////////////////
// corrector step for Predictor-Corrector time integration
__global__ void KERNEL_xsph2D(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part1*TP1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;

	int_t icell,jcell;
	int_t ptypei;
	Real xi,yi;
	Real uxi,uyi;
	Real tux,tuy;																	// velocity
	Real tx0,ty0,xc,yc;														// position ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
	Real flt_si;
	Real tmpx,tmpy;
	Real search_range,tmp_h,tmp_A;
	Real t_dt=tdt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;						// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	uxi=TP1[i].ux;
	uyi=TP1[i].uy;
	tx0=P2[i].x0;																// x-directional initial position
	ty0=P2[i].y0;																// x-directional initial position
	flt_si=P1[i].flt_s;
	tux=uxi;
	tuy=uyi;

	Real x_boundary=0.1*cos(PI*ttime)+3.79;

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
							Real twij,uxj,uyj,mj,rhoj;
							twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
							//p_type_j=p_type_[j];
							uxj=TP1[j].ux;
							uyj=TP1[j].uy;
							mj=P1[j].m;
							rhoj=P1[j].rho;
	
							tmpx+=k_c_xsph*mj/rhoj*(-uxi+uxj)*twij/flt_si;
							tmpy+=k_c_xsph*mj/rhoj*(-uyi+uyj)*twij/flt_si;
						}


					}

					
				}
			}
		}
	}

	if(ptypei==MOVING){
		xc=tx0+tux*(t_dt);			// correct x-directional position
		yc=ty0+tuy*(t_dt);			// correct Y-directional position
	}else{
		xc=tx0+(tux+tmpx)*(t_dt)*(ptypei>0);			// correct x-directional position
		yc=ty0+(tuy+tmpy)*(t_dt)*(ptypei>0);			// correct Y-directional position
	}
	TP1[i].x=xc;				// update x-directional position
	TP1[i].y=yc;				// update y-directional position
}
////////////////////////////////////////////////////////////////////////
// corrector step for Predictor-Corrector time integration
__global__ void KERNEL_xsph3D(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;
	if((P1[i].p_type==0)||(P1[i].p_type==9)) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	Real tempi;
	Real tux,tuy,tuz,ttemp;																	// velocity
	Real tx0,ty0,tz0,xc,yc,zc;												// position ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
	//Real flt_si;
	Real tmpx,tmpy,tmpz,tmptemp;
	Real search_range,tmp_h,tmp_A,tmp_flt;
	Real t_dt=tdt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;								// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	tempi=P1[i].temp;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	tx0=P2[i].x0;					// x-directional initial position
	ty0=P2[i].y0;					// x-directional initial position
	tz0=P2[i].z0;					// x-directional initial position
	//flt_si=P1[i].flt_s;
	tux=uxi;
	tuy=uyi;
	tuz=uzi;
	ttemp=tempi;

	//Real x_boundary=0.1*cos(PI*ttime)+3.79;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=tmptemp=0.0;
	tmp_flt=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						
						if((P1[j].p_type==1)||(P1[j].p_type==3)){

							Real xj,yj,zj,tdist;
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist<search_range){
								Real twij,uxj,uyj,uzj,tempj,mj,rhoj;
								twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								//p_type_j=p_type_[j];
								uxj=P1[j].ux;
								uyj=P1[j].uy;
								uzj=P1[j].uz;
								mj=P1[j].m;
								rhoj=P1[j].rho;
								tempj=P1[j].temp;
	
								tmpx+=k_c_xsph*mj/rhoj*(-uxi+uxj)*twij;
								tmpy+=k_c_xsph*mj/rhoj*(-uyi+uyj)*twij;
								tmpz+=k_c_xsph*mj/rhoj*(-uzi+uzj)*twij;

								tmptemp+=0.01*k_c_xsph*mj/rhoj*(tempj-tempi)*twij;

								tmp_flt+=mj/rhoj*twij;
							}



						}
						
						
					}
				}
			}
		}
	}

	xc=tx0+(tux+tmpx/tmp_flt)*(t_dt)*(ptypei>0);			// correct x-directional position
	yc=ty0+(tuy+tmpy/tmp_flt)*(t_dt)*(ptypei>0);			// correct Y-directional position
	zc=tz0+(tuz+tmpz/tmp_flt)*(t_dt)*(ptypei>0);			// correct Z-directional position

	P1[i].XSPH_ux=(tux+tmpx/tmp_flt)*(ptypei>0);				// update x-directional position
	P1[i].XSPH_uy=(tuy+tmpy/tmp_flt)*(ptypei>0);				// update y-directional position
	P1[i].XSPH_uz=(tuz+tmpz/tmp_flt)*(ptypei>0);				// update z-directional position

	P1[i].XSPH_temp=(ttemp+tmptemp/tmp_flt)*(ptypei>0);	
}



__global__ void KERNEL_xsph3D_sph(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;
	if((P1[i].p_type==0)||(P1[i].p_type==9)) return;

	int_t ptypei;
	int_t icell,jcell,kcell;
	Real xi,yi,zi;
	Real uxi,uyi,uzi;
	Real tempi;
	Real tux,tuy,tuz,ttemp;																	// velocity
	Real tx0,ty0,tz0,xc,yc,zc;												// position ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
	//Real flt_si;
	Real tmpx,tmpy,tmpz,tmptemp;
	Real search_range,tmp_h,tmp_A,tmp_flt;
	Real t_dt=tdt;

	tmp_h=P1[i].h;
	tmp_A=calc_tmpA(tmp_h);
	search_range=k_search_kappa*tmp_h;								// search range

	ptypei=P1[i].p_type;

	xi=P1[i].x;
	yi=P1[i].y;
	zi=P1[i].z;
	tempi=P1[i].temp;
	uxi=P1[i].ux;
	uyi=P1[i].uy;
	uzi=P1[i].uz;
	tx0=P2[i].x0;					// x-directional initial position
	ty0=P2[i].y0;					// x-directional initial position
	tz0=P2[i].z0;					// x-directional initial position
	//flt_si=P1[i].flt_s;
	tux=uxi;
	tuy=uyi;
	tuz=uzi;
	ttemp=tempi;

	//Real x_boundary=0.1*cos(PI*ttime)+3.79;

	// calculate I,J,K in cell
	if((k_x_max==k_x_min)){icell=0;}
	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
	if((k_y_max==k_y_min)){jcell=0;}
	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
	if((k_z_max==k_z_min)){kcell=0;}
	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
	// out-of-range handling
	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

	tmpx=tmpy=tmpz=tmptemp=0.0;
	tmp_flt=0.0;
	for(int_t z=-1;z<=1;z++){
		for(int_t y=-1;y<=1;y++){
			for(int_t x=-1;x<=1;x++){
				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
				if(g_str[k]!=cu_memset){
					int_t fend=g_end[k];
					for(int_t j=g_str[k];j<fend;j++){
						
						if((P1[j].p_type==1)||(P1[j].p_type==3)){

							Real xj,yj,zj,tdist;
							xj=P1[j].x;
							yj=P1[j].y;
							zj=P1[j].z;
	
							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
							if(tdist<search_range){
								Real twij,uxj,uyj,uzj,tempj,mj,rhoj;
								twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
								//p_type_j=p_type_[j];
								uxj=P1[j].ux;
								uyj=P1[j].uy;
								uzj=P1[j].uz;
								mj=P1[j].m;
								rhoj=P1[j].rho;
								tempj=P1[j].temp;
	
								tmpx+=k_c_xsph*mj/rhoj*(-uxi+uxj)*twij;
								tmpy+=k_c_xsph*mj/rhoj*(-uyi+uyj)*twij;
								tmpz+=k_c_xsph*mj/rhoj*(-uzi+uzj)*twij;

								tmptemp+=0.01*k_c_xsph*mj/rhoj*(tempj-tempi)*twij;

								tmp_flt+=mj/rhoj*twij;
							}



						}
						
						
					}
				}
			}
		}
	}

	xc=tx0+(tux+tmpx/tmp_flt)*(t_dt)*(ptypei>0);			// correct x-directional position
	yc=ty0+(tuy+tmpy/tmp_flt)*(t_dt)*(ptypei>0);			// correct Y-directional position
	zc=tz0+(tuz+tmpz/tmp_flt)*(t_dt)*(ptypei>0);			// correct Z-directional position

	P1[i].XSPH_ux=(tux+tmpx/tmp_flt)*(ptypei>0);				// update x-directional position
	P1[i].XSPH_uy=(tuy+tmpy/tmp_flt)*(ptypei>0);				// update y-directional position
	P1[i].XSPH_uz=(tuz+tmpz/tmp_flt)*(ptypei>0);				// update z-directional position

	P1[i].XSPH_temp=(ttemp+tmptemp/tmp_flt)*(ptypei>0);	
}
// __global__ void KERNEL_xsph3D_sph(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part2*P2)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2_sph) return;
// 	if(P1[i].i_type!=inout) return;
// 	if(P1[i].p_type>1000) return;
// 	if((P1[i].p_type==0)||(P1[i].p_type==9)) return;

// 	int_t ptypei;
// 	int_t icell,jcell,kcell;
// 	Real xi,yi,zi;
// 	Real uxi,uyi,uzi;
// 	Real tempi;
// 	Real tux,tuy,tuz,ttemp;																	// velocity
// 	Real tx0,ty0,tz0,xc,yc,zc;												// position ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
// 	//Real flt_si;
// 	Real tmpx,tmpy,tmpz,tmptemp;
// 	Real search_range,tmp_h,tmp_A,tmp_flt;
// 	Real t_dt=tdt;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	search_range=k_search_kappa*tmp_h;								// search range

// 	ptypei=P1[i].p_type;

// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;
// 	tempi=P1[i].temp;
// 	uxi=P1[i].ux;
// 	uyi=P1[i].uy;
// 	uzi=P1[i].uz;
// 	tx0=P2[i].x0;					// x-directional initial position
// 	ty0=P2[i].y0;					// x-directional initial position
// 	tz0=P2[i].z0;					// x-directional initial position
// 	//flt_si=P1[i].flt_s;
// 	tux=uxi;
// 	tuy=uyi;
// 	tuz=uzi;
// 	ttemp=tempi;

// 	//Real x_boundary=0.1*cos(PI*ttime)+3.79;

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

// 	tmpx=tmpy=tmpz=tmptemp=0.0;
// 	tmp_flt=0.0;
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
						
// 						if((P1[j].p_type==1)||(P1[j].p_type==3)){

// 							Real xj,yj,zj,tdist;
// 							xj=P1[j].x;
// 							yj=P1[j].y;
// 							zj=P1[j].z;
	
// 							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
// 							if(tdist<search_range){
// 								Real twij,uxj,uyj,uzj,tempj,mj,rhoj;
// 								twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
// 								//p_type_j=p_type_[j];
// 								uxj=P1[j].ux;
// 								uyj=P1[j].uy;
// 								uzj=P1[j].uz;
// 								mj=P1[j].m;
// 								rhoj=P1[j].rho;
// 								tempj=P1[j].temp;
	
// 								tmpx+=k_c_xsph*mj/rhoj*(-uxi+uxj)*twij;
// 								tmpy+=k_c_xsph*mj/rhoj*(-uyi+uyj)*twij;
// 								tmpz+=k_c_xsph*mj/rhoj*(-uzi+uzj)*twij;

// 								tmptemp+=0.01*k_c_xsph*mj/rhoj*(tempj-tempi)*twij;

// 								tmp_flt+=mj/rhoj*twij;
// 							}



// 						}
						
						
// 					}
// 				}
// 			}
// 		}
// 	}

// 	xc=tx0+(tux+tmpx/tmp_flt)*(t_dt)*(ptypei>0);			// correct x-directional position
// 	yc=ty0+(tuy+tmpy/tmp_flt)*(t_dt)*(ptypei>0);			// correct Y-directional position
// 	zc=tz0+(tuz+tmpz/tmp_flt)*(t_dt)*(ptypei>0);			// correct Z-directional position

// 	P1[i].XSPH_ux=(tux+tmpx/tmp_flt)*(ptypei>0);				// update x-directional position
// 	P1[i].XSPH_uy=(tuy+tmpy/tmp_flt)*(ptypei>0);				// update y-directional position
// 	P1[i].XSPH_uz=(tuz+tmpz/tmp_flt)*(ptypei>0);				// update z-directional position

// 	P1[i].XSPH_temp=(ttemp+tmptemp/tmp_flt)*(ptypei>0);	
// }



__global__ void KERNEL_xsph3D_calc(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;
	if((P1[i].p_type==0)||(P1[i].p_type==9)) return;

	P1[i].ux=P1[i].XSPH_ux;				// update x-directional position
	P1[i].uy=P1[i].XSPH_uy;				// update y-directional position
	P1[i].uz=P1[i].XSPH_uz;
	P1[i].temp=P1[i].XSPH_temp;
}


__global__ void KERNEL_xsph3D_calc_sph(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part2*P2)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_sph) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type>1000) return;
	if((P1[i].p_type==0)||(P1[i].p_type==9)) return;

	P1[i].ux=P1[i].XSPH_ux;				// update x-directional position
	P1[i].uy=P1[i].XSPH_uy;				// update y-directional position
	P1[i].uz=P1[i].XSPH_uz;
	P1[i].temp=P1[i].XSPH_temp;
}
// __global__ void KERNEL_xsph3D_calc_sph(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part2*P2)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2_sph) return;
// 	if(P1[i].i_type!=inout) return;
// 	if(P1[i].p_type>1000) return;
// 	if((P1[i].p_type==0)||(P1[i].p_type==9)) return;

// 	P1[i].ux=P1[i].XSPH_ux;				// update x-directional position
// 	P1[i].uy=P1[i].XSPH_uy;				// update y-directional position
// 	P1[i].uz=P1[i].XSPH_uz;
// 	P1[i].temp=P1[i].XSPH_temp;
// }

// __global__ void KERNEL_xsph3D(int_t inout,int_t*g_str,int_t*g_end,Real tdt,Real ttime,part1*P1,part2*P2)
// {
// 	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
// 	if(i>=k_num_part2) return;
// 	if(P1[i].i_type!=inout) return;
// 	if(P1[i].p_type>1000) return;
// 	if((P1[i].p_type==0)||(P1[i].p_type==9)) return;

// 	int_t ptypei;
// 	int_t icell,jcell,kcell;
// 	Real xi,yi,zi;
// 	Real uxi,uyi,uzi;
// 	Real tempi;
// 	Real tux,tuy,tuz,ttemp;																	// velocity
// 	Real tx0,ty0,tz0,xc,yc,zc;												// position ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
// 	//Real flt_si;
// 	Real tmpx,tmpy,tmpz,tmptemp;
// 	Real search_range,tmp_h,tmp_A,tmp_flt;
// 	Real t_dt=tdt;

// 	tmp_h=P1[i].h;
// 	tmp_A=calc_tmpA(tmp_h);
// 	search_range=k_search_kappa*tmp_h;								// search range

// 	ptypei=P1[i].p_type;

// 	xi=P1[i].x;
// 	yi=P1[i].y;
// 	zi=P1[i].z;
// 	tempi=P1[i].temp;
// 	uxi=P1[i].ux;
// 	uyi=P1[i].uy;
// 	uzi=P1[i].uz;
// 	tx0=P2[i].x0;					// x-directional initial position
// 	ty0=P2[i].y0;					// x-directional initial position
// 	tz0=P2[i].z0;					// x-directional initial position
// 	//flt_si=P1[i].flt_s;
// 	tux=uxi;
// 	tuy=uyi;
// 	tuz=uzi;
// 	ttemp=tempi;

// 	//Real x_boundary=0.1*cos(PI*ttime)+3.79;

// 	// calculate I,J,K in cell
// 	if((k_x_max==k_x_min)){icell=0;}
// 	else{icell=min(floor((xi-k_x_min)/(k_x_max-k_x_min)*k_NI),k_NI-1);}
// 	if((k_y_max==k_y_min)){jcell=0;}
// 	else{jcell=min(floor((yi-k_y_min)/(k_y_max-k_y_min)*k_NJ),k_NJ-1);}
// 	if((k_z_max==k_z_min)){kcell=0;}
// 	else{kcell=min(floor((zi-k_z_min)/(k_z_max-k_z_min)*k_NK),k_NK-1);}
// 	// out-of-range handling
// 	if(icell<0) icell=0;	if(jcell<0) jcell=0;	if(kcell<0) kcell=0;

// 	tmpx=tmpy=tmpz=tmptemp=0.0;
// 	tmp_flt=0.0;
// 	for(int_t z=-1;z<=1;z++){
// 		for(int_t y=-1;y<=1;y++){
// 			for(int_t x=-1;x<=1;x++){
// 				// int_t k=(icell+x)+k_NI*(jcell+y)+k_NI*k_NJ*(kcell+z);
// 				int_t k=idx_cell(icell+x,jcell+y,kcell+z);

// 				if(((icell+x)<0)||((icell+x)>(k_NI-1))||((jcell+y)<0)||((jcell+y)>(k_NJ-1))||((kcell+z)<0)||((kcell+z)>(k_NK-1))) continue;
// 				if(g_str[k]!=cu_memset){
// 					int_t fend=g_end[k];
// 					for(int_t j=g_str[k];j<fend;j++){
						
// 						if((P1[j].p_type==1)||(P1[j].p_type==3)){

// 							Real xj,yj,zj,tdist;
// 							xj=P1[j].x;
// 							yj=P1[j].y;
// 							zj=P1[j].z;
	
// 							tdist=sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj));
// 							if(tdist<search_range){
// 								Real twij,uxj,uyj,uzj,tempj,mj,rhoj;
// 								twij=calc_kernel_wij(tmp_A,tmp_h,tdist);
// 								//p_type_j=p_type_[j];
// 								uxj=P1[j].ux;
// 								uyj=P1[j].uy;
// 								uzj=P1[j].uz;
// 								mj=P1[j].m;
// 								rhoj=P1[j].rho;
// 								tempj=P1[j].temp;
	
// 								tmpx+=k_c_xsph*mj/rhoj*(-uxi+uxj)*twij;
// 								tmpy+=k_c_xsph*mj/rhoj*(-uyi+uyj)*twij;
// 								tmpz+=k_c_xsph*mj/rhoj*(-uzi+uzj)*twij;

// 								tmptemp+=0.01*k_c_xsph*mj/rhoj*(tempj-tempi)*twij;

// 								tmp_flt+=mj/rhoj*twij;
// 							}



// 						}
						
						
// 					}
// 				}
// 			}
// 		}
// 	}

// 	xc=tx0+(tux+tmpx/tmp_flt)*(t_dt)*(ptypei>0);			// correct x-directional position
// 	yc=ty0+(tuy+tmpy/tmp_flt)*(t_dt)*(ptypei>0);			// correct Y-directional position
// 	zc=tz0+(tuz+tmpz/tmp_flt)*(t_dt)*(ptypei>0);			// correct Z-directional position

// 	P1[i].ux=(tux+tmpx/tmp_flt)*(ptypei>0);				// update x-directional position
// 	P1[i].uy=(tuy+tmpy/tmp_flt)*(ptypei>0);				// update y-directional position
// 	P1[i].uz=(tuz+tmpz/tmp_flt)*(ptypei>0);				// update z-directional position

// 	P1[i].temp=(ttemp+tmptemp/tmp_flt)*(ptypei>0);	
// }
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_update_density(int_t inout,part1*P1,part1*TP1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;

	Real tmp_r=2*TP1[i].rho-P2[i].rho0;
	// update density
	TP1[i].rho=tmp_r;
}
////////////////////////////////////////////////////////////////////////
// void update_properties(int_t inout,int_t*g_str,int_t*g_end,part1*P1,part1*TP1,part2*P2,part3*P3)
// {
// 	dim3 b,t;
// 	t.x=128;
// 	b.x=(num_part-1)/t.x+1;
//
// 	if(time_type==Euler){
// 		// Eulerian time integration function
// 		// KERNEL_clc_euler_update include KERNEL_clc_precor_update_continuity
// 		KERNEL_clc_euler_update<<<b,t>>>(inout,dt,P1,TP1,P2,P3);
// 		cudaDeviceSynchronize();
// 		if(rho_type==Continuity){
// 			if(count%freq_mass_sum==0){
// 				if(dim==2) KERNEL_clc_mass_sum2D<<<b,t>>>(g_str,g_end,P1,TP1,P2);
// 				if(dim==3) KERNEL_clc_mass_sum3D<<<b,t>>>(g_str,g_end,P1,TP1,P2);
// 				cudaDeviceSynchronize();
// 			}
// 		}
// 	}
//
// 	if(time_type==Pre_Cor){
// 		// Predictor-Corrector time integration function
// 		if(xsph_solve==1){
// 			// KERNEL_clc_precor_update_vel include KERNEL_clc_precor_update_continuity,
// 			// update_properties_enthalpy and update_properties_concn
// 			// KERNEL_clc_precor_update_vel<<<b,t>>>(inout,dt,P1,TP1,P2,P3);
// 			// cudaDeviceSynchronize();
// 			// // if(dim==2) KERNEL_clc_precor_update_xsph2D<<<b,t>>>(inout,g_str,g_end,dt,time,P1,TP1,P2);
// 			// // if(dim==3) KERNEL_clc_precor_update_xsph3D<<<b,t>>>(inout,g_str,g_end,dt,time,P1,TP1,P2);
// 			// cudaDeviceSynchronize();
// 		}else{
			// KERNEL_clc_precor_update include KERNEL_clc_precor_update_continuity,
			// update_properties_enthalpy and update_properties_concn
// 			KERNEL_clc_precor_update<<<b,t>>>(inout,dt,P1,TP1,P2,P3);
// 			cudaDeviceSynchronize();
// 		}
// 		if(rho_type==Continuity){
// 			if(count%freq_mass_sum==0){
// 				if(dim==2) KERNEL_clc_density_renormalization_norm2D<<<b,t>>>(g_str,g_end,P1,TP1,P2);
// 				if(dim==3) KERNEL_clc_density_renormalization_norm3D<<<b,t>>>(g_str,g_end,P1,TP1,P2);
// 				cudaDeviceSynchronize();
// 			}
// 			KERNEL_clc_update_density<<<b,t>>>(inout,P1,TP1,P2);
// 			cudaDeviceSynchronize();
// 		}
// 	}
// }

__global__ void KERNEL_clc_predictor_movingptl (Real tdt ,Real ttime ,part1 *P1 ,part2 *P2 ,part3 *P3 )
{
    int_t i =threadIdx .x +blockIdx .x *blockDim .x ;
    if (i >=k_num_part2) return ;
    if (P1 [i].i_type ==3 ) return ;
    int_t p_typei;
    Real tx0,ty0,tz0,txp,typ,tzp;
    Real tux0,tuy0,tuz0,tuxp,tuyp,tuzp;
    Real tdux_dt0,tduy_dt0,tduz_dt0;
    Real t_dt;
    int_t buffer_type =P1 [i].buffer_type ;
    t_dt =tdt;
	p_typei = P1 [i].p_type ;
	tx0 = P1 [i].x;
	ty0 = P1 [i].y;
    tz0 = P1 [i].z;															// z 위치에 따라 적용 입자 고르기 위해 zi 먼저 불러옴
    if ((p_typei ==2) && (tz0 >= 0.0) && (tx0*tx0 + ty0*ty0 <= 18) ){      //노즐안에 유체에 적용        // 무빙 플레이트 사용시 속도 변경 m->mm 단위로
        //tx0 =P1 [i].x ;                                                     // initial x-directional position
        //ty0 =P1 [i].y ;                                                     // initial y-directional position
        //tz0 =P1 [i].z ;    위에서 먼저 불러와서 주석처리                      // initial z-directional position
        P2 [i].x0 =tx0;
        P2 [i].y0 =ty0;
        P2 [i].z0 =tz0;
        tux0 =0.0 ;
        tuy0 =0.0 ;
        tuz0 =-2120.0 ;
        txp =tx0 +tux0 *(t_dt *0.5 );                                    // Predict x-directional position (ux0 : velocity of before time step)
        typ =ty0 +tuy0 *(t_dt *0.5 );                                    // Predict y-directional position (uy0 : velocity of before time step)
        tzp =tz0 +tuz0 *(t_dt *0.5 );                                    // Predict z-directional position (ux0 : velocity of before time step)
        P1 [i].x =txp;                                                            // Update particle data by predicted x-directional position
        P1 [i].y =typ;                                                            // Update particle data by predicted y-directional position
        P1 [i].z =tzp;                                                            // Update particle data by predicted z-directional position
        P1 [i].ux =tux0;                                                      // Update particle data by predicted x-directional velocity
        P1 [i].uy =tuy0;                                                      // Update particle data by predicted y-directional velocity
        P1 [i].uz =tuz0;                                                      // Update particle data by predicted z-directional velocity
        P2 [i].ux0 =tux0;                                                     // Update particle data by predicted x-directional velocity
        P2 [i].uy0 =tuy0;                                                     // Update particle data by predicted y-directional velocity
        P2 [i].uz0 =tuz0;                                                     // Update particle data by predicted z-directional velocity
    }else {
        tx0 =P1 [i].x ;                                                            // initial x-directional position
        ty0 =P1 [i].y ;                                                            // initial y-directional position
        tz0 =P1 [i].z ;                                                            // initial z-directional position
        if (p_typei >0 ){
            tux0 =P1 [i].ux ;                                                  // initial x-directional velocity
            tuy0 =P1 [i].uy ;                                                  // initial y-directional velocity
            tuz0 =P1 [i].uz ;                                                  // initial z-directional velocity
            tdux_dt0 =P3 [i].ftotalx *(buffer_type ==0 );                                    // initial x-directional acceleration
            tduy_dt0 =P3 [i].ftotaly *(buffer_type ==0 );                                    // initial y-directional acceleration
            tduz_dt0 =P3 [i].ftotalz *(buffer_type ==0 );                                    // initial z-directional acceleration
            txp =tx0 +tux0 *(t_dt *0.5 );                                    // Predict x-directional position (ux0 : velocity of before time step)
            typ =ty0 +tuy0 *(t_dt *0.5 );                                    // Predict y-directional position (uy0 : velocity of before time step)
            tzp =tz0 +tuz0 *(t_dt *0.5 );                                    // Predict z-directional position (ux0 : velocity of before time step)
            tuxp =tux0 +tdux_dt0 *(t_dt *0.5 );                      // Predict x-directional velocity (dux_dt0 : acceleration of before time step)
            tuyp =tuy0 +tduy_dt0 *(t_dt *0.5 );                      // Predict y-directional velocity (duy_dt0 : acceleration of before time step)
            tuzp =tuz0 +tduz_dt0 *(t_dt *0.5 );                      // Predict z-directional velocity (duz_dt0 : acceleration of before time step)
        }else {
            txp =tx0;typ =ty0;tzp =tz0;
            tuxp =P1 [i].ux ;
            tuyp =P1 [i].uy ;
            tuzp =P1 [i].uz ;
        }
        P1 [i].x =txp;                                                            // Update particle data by predicted x-directional position
        P1 [i].y =typ;                                                            // Update particle data by predicted y-directional position
        P1 [i].z =tzp;                                                            // Update particle data by predicted z-directional position
        P1 [i].ux =tuxp;                                                      // Update particle data by predicted x-directional velocity
        P1 [i].uy =tuyp;                                                      // Update particle data by predicted y-directional velocity
        P1 [i].uz =tuzp;                                                      // Update particle data by predicted z-directional velocity
        P2 [i].x0 =tx0;                                                           // update x-directional position
        P2 [i].y0 =ty0;                                                           // update y-directional position
        P2 [i].z0 =tz0;                                                           // update z-directional position
        P2 [i].ux0 =tux0;                                                     // update x-directional velocity
        P2 [i].uy0 =tuy0;                                                     // update y-directional velocity
        P2 [i].uz0 =tuz0;                                                     // update z-directional velocity
    }
    // predict density - KERNEL_clc_predictor_continuity(drho_dt0 : time derivatve of density of before time step)
    if (k_rho_type ==Continuity){
        Real trho =P1 [i].rho ;
        P2 [i].rho0 =trho;
        P1 [i].rho =trho +P3 [i].drho *(t_dt *0.5 );
    }
    // // KERNEL_clc_reference_density
    // P2[i].rho_ref=P1[i].m/pow(P1[i].h/1.5,k_dim);
    //----------------------------------------------------
    // KERNEL_clc_predictor_enthalpy - Update particle data by predicted density
    if (k_con_solve ==1 ){
        Real tenthalpyp =P1 [i].enthalpy ;
        P2 [i].enthalpy0 =tenthalpyp;
        //
        tenthalpyp +=P3 [i].denthalpy *(t_dt *0.5 );
        P1 [i].enthalpy =tenthalpyp;
        P1 [i].temp =htoT (tenthalpyp,p_typei);
    }
    // KERNEL_clc_predictor_concn - Predict concentration (dconcn_dt0 : time derivatve of density of before time step)
    // Update particle data by predicted concentration
    if (k_concn_solve ==1 ){
        Real tconcn =P1 [i].concn ;
        P2 [i].concn0 =tconcn;
        P1 [i].concn =tconcn +P3 [i].dconcn *(t_dt *0.5 );
    }
}

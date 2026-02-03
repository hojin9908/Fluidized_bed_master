#define rest_n 		0.9
#define rest_s 		-0.9	

#define mu_s 		0.10
#define mu_rs		0.2



#define xy_cylinder		0.0
#define zy_cylinder 	0.2

#define ry_cylinder	 	0.2


__global__ void KERNEL_treat_DEM_cylinder_y(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;

	
	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_cylinder;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_cylinder_mag;
	Real nx_cylinder,ny_cylinder,nz_cylinder;
	Real sx_cylinder,sy_cylinder,sz_cylinder;
	Real swx_cylinder,swy_cylinder,swz_cylinder;

	Real x_center=xy_cylinder;
	Real z_center=zy_cylinder;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_cylinder=(x-x_center)*(x-x_center)+(z-z_center)*(z-z_center)-(ry_cylinder-rad)*(ry_cylinder-rad);

	if(F_cylinder<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+(ry_cylinder-rad)*(x-x_center)/sqrt((x-x_center)*(x-x_center)+(z-z_center)*(z-z_center));
	cpy=y;
	cpz=z_center+(ry_cylinder-rad)*(z-z_center)/sqrt((x-x_center)*(x-x_center)+(z-z_center)*(z-z_center));



	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	nx_cylinder=-(cpx-x_center)/(ry_cylinder-rad);
	ny_cylinder=0.0;
	nz_cylinder=-(cpz-z_center)/(ry_cylinder-rad);
	
	n_cylinder_mag=sqrt(nx_cylinder*nx_cylinder+ny_cylinder*ny_cylinder+nz_cylinder*nz_cylinder);


	if(n_cylinder_mag<0.5) return;		// actually do not contact the cylinder wall yet...

	un_mag=ux*nx_cylinder+uy*ny_cylinder+uz*nz_cylinder;

	ucx=ux-rad*(wy*nz_cylinder-wz*ny_cylinder);
	ucy=uy-rad*(wz*nx_cylinder-wx*nz_cylinder);
	ucz=uz-rad*(wx*ny_cylinder-wy*nx_cylinder);

	ucx_w=-rad*(wy*nz_cylinder-wz*ny_cylinder);
	ucy_w=-rad*(wz*nx_cylinder-wx*nz_cylinder);
	ucz_w=-rad*(wx*ny_cylinder-wy*nx_cylinder);

	ucn_mag=ucx*nx_cylinder+ucy*ny_cylinder+ucz*nz_cylinder;
	uc_wn_mag=ucx_w*nx_cylinder+ucy_w*ny_cylinder+ucz_w*nz_cylinder;

	ucs_x=ucx-ucn_mag*nx_cylinder;
	ucs_y=ucy-ucn_mag*ny_cylinder;
	ucs_z=ucz-ucn_mag*nz_cylinder;

	uc_ws_x=ucx_w-uc_wn_mag*nx_cylinder;
	uc_ws_y=ucy_w-uc_wn_mag*ny_cylinder;
	uc_ws_z=ucz_w-uc_wn_mag*nz_cylinder;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_cylinder=ucs_x/(ucs_mag+1e-20);
	sy_cylinder=ucs_y/(ucs_mag+1e-20);
	sz_cylinder=ucs_z/(ucs_mag+1e-20);

	swx_cylinder=uc_ws_x/(uc_ws_mag+1e-20);
	swy_cylinder=uc_ws_y/(uc_ws_mag+1e-20);
	swz_cylinder=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt)) ) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		//ffx=-mu_s*m*Gravitational_CONST*nz_cylinder*sx_cylinder;
		//ffy=-mu_s*m*Gravitational_CONST*nz_cylinder*sy_cylinder;
		//ffz=-mu_s*m*Gravitational_CONST*nz_cylinder*sz_cylinder;

		//frfx=-mu_rs*m*Gravitational_CONST*nz_cylinder*swx_cylinder;
		//frfy=-mu_rs*m*Gravitational_CONST*nz_cylinder*swy_cylinder;
		//frfz=-mu_rs*m*Gravitational_CONST*nz_cylinder*swz_cylinder;

		ffx=mu_s*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*sx_cylinder;
		ffy=mu_s*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*sy_cylinder;
		ffz=mu_s*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*sz_cylinder;

		frfx=mu_rs*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*swx_cylinder;
		frfy=mu_rs*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*swy_cylinder;
		frfz=mu_rs*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*swz_cylinder;

		if ((ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_cylinder*(ffz+frfz)-nz_cylinder*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_cylinder*(ffx+frfx)-nx_cylinder*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_cylinder*(ffy+frfy)-ny_cylinder*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_cylinder+uy_c*ny_cylinder+uz_c*nz_cylinder;

		TP1[i].ux=ux_c-u_cn_mag*nx_cylinder;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_cylinder;
		TP1[i].uz=uz_c-u_cn_mag*nz_cylinder;

		TP1[i].wx=wx-(rad/ri)*(ny_cylinder*(ffz+frfz)-nz_cylinder*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_cylinder*(ffx+frfx)-nx_cylinder*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_cylinder*(ffy+frfy)-ny_cylinder*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_cylinder;
		Jny=-m*(1+rest_n)*ucn_mag*ny_cylinder;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_cylinder;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_cylinder;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_cylinder;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_cylinder;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_cylinder*Jz-nz_cylinder*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_cylinder*Jx-nx_cylinder*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_cylinder*Jy-ny_cylinder*Jx);


	}


}


#define xz_cylinder		0.0
#define yz_cylinder 	0.0

#define rz_cylinder	 	0.0125

	
__global__ void KERNEL_treat_DEM_cylinder_z(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;

	
	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_cylinder;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_cylinder_mag;
	Real nx_cylinder,ny_cylinder,nz_cylinder;
	Real sx_cylinder,sy_cylinder,sz_cylinder;
	Real swx_cylinder,swy_cylinder,swz_cylinder;

	Real x_center=xz_cylinder;
	Real y_center=yz_cylinder;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_cylinder=(x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)-(rz_cylinder-rad)*(rz_cylinder-rad);

	if(F_cylinder<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+(rz_cylinder-rad)*(x-x_center)/sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center));
	cpy=y_center+(rz_cylinder-rad)*(y-y_center)/sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center));
	cpz=z;


	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	nx_cylinder=-(cpx-x_center)/(rz_cylinder-rad);
	ny_cylinder=-(cpy-y_center)/(rz_cylinder-rad);
	nz_cylinder=0.0;

	n_cylinder_mag=sqrt(nx_cylinder*nx_cylinder+ny_cylinder*ny_cylinder+nz_cylinder*nz_cylinder);


	if(n_cylinder_mag<0.5) return;		// actually do not contact the cylinder wall yet...

	un_mag=ux*nx_cylinder+uy*ny_cylinder+uz*nz_cylinder;

	ucx=ux-rad*(wy*nz_cylinder-wz*ny_cylinder);
	ucy=uy-rad*(wz*nx_cylinder-wx*nz_cylinder);
	ucz=uz-rad*(wx*ny_cylinder-wy*nx_cylinder);

	ucx_w=-rad*(wy*nz_cylinder-wz*ny_cylinder);
	ucy_w=-rad*(wz*nx_cylinder-wx*nz_cylinder);
	ucz_w=-rad*(wx*ny_cylinder-wy*nx_cylinder);

	ucn_mag=ucx*nx_cylinder+ucy*ny_cylinder+ucz*nz_cylinder;
	uc_wn_mag=ucx_w*nx_cylinder+ucy_w*ny_cylinder+ucz_w*nz_cylinder;

	ucs_x=ucx-ucn_mag*nx_cylinder;
	ucs_y=ucy-ucn_mag*ny_cylinder;
	ucs_z=ucz-ucn_mag*nz_cylinder;

	uc_ws_x=ucx_w-uc_wn_mag*nx_cylinder;
	uc_ws_y=ucy_w-uc_wn_mag*ny_cylinder;
	uc_ws_z=ucz_w-uc_wn_mag*nz_cylinder;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_cylinder=ucs_x/(ucs_mag+1e-20);
	sy_cylinder=ucs_y/(ucs_mag+1e-20);
	sz_cylinder=ucs_z/(ucs_mag+1e-20);

	swx_cylinder=uc_ws_x/(uc_ws_mag+1e-20);
	swy_cylinder=uc_ws_y/(uc_ws_mag+1e-20);
	swz_cylinder=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*sx_cylinder;
		ffy=mu_s*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*sy_cylinder;
		ffz=mu_s*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*sz_cylinder;

		frfx=mu_rs*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*swx_cylinder;
		frfy=mu_rs*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*swy_cylinder;
		frfz=mu_rs*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*swz_cylinder;

		if ((ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_cylinder*(ffz+frfz)-nz_cylinder*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_cylinder*(ffx+frfx)-nx_cylinder*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_cylinder*(ffy+frfy)-ny_cylinder*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_cylinder+uy_c*ny_cylinder+uz_c*nz_cylinder;

		TP1[i].ux=ux_c-u_cn_mag*nx_cylinder;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_cylinder;
		TP1[i].uz=uz_c-u_cn_mag*nz_cylinder;

		TP1[i].wx=wx-(rad/ri)*(ny_cylinder*(ffz+frfz)-nz_cylinder*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_cylinder*(ffx+frfx)-nx_cylinder*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_cylinder*(ffy+frfy)-ny_cylinder*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_cylinder;
		Jny=-m*(1+rest_n)*ucn_mag*ny_cylinder;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_cylinder;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_cylinder;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_cylinder;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_cylinder;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_cylinder*Jz-nz_cylinder*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_cylinder*Jx-nx_cylinder*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_cylinder*Jy-ny_cylinder*Jx);


	}


}


	
__global__ void KERNEL_treat_DEM_cylinder_z_dem(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;

	
	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_cylinder;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_cylinder_mag;
	Real nx_cylinder,ny_cylinder,nz_cylinder;
	Real sx_cylinder,sy_cylinder,sz_cylinder;
	Real swx_cylinder,swy_cylinder,swz_cylinder;

	Real x_center=xz_cylinder;
	Real y_center=yz_cylinder;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_cylinder=(x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)-(rz_cylinder-rad)*(rz_cylinder-rad);

	if(F_cylinder<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+(rz_cylinder-rad)*(x-x_center)/sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center));
	cpy=y_center+(rz_cylinder-rad)*(y-y_center)/sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center));
	cpz=z;


	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	nx_cylinder=-(cpx-x_center)/(rz_cylinder-rad);
	ny_cylinder=-(cpy-y_center)/(rz_cylinder-rad);
	nz_cylinder=0.0;

	n_cylinder_mag=sqrt(nx_cylinder*nx_cylinder+ny_cylinder*ny_cylinder+nz_cylinder*nz_cylinder);


	if(n_cylinder_mag<0.5) return;		// actually do not contact the cylinder wall yet...

	un_mag=ux*nx_cylinder+uy*ny_cylinder+uz*nz_cylinder;

	ucx=ux-rad*(wy*nz_cylinder-wz*ny_cylinder);
	ucy=uy-rad*(wz*nx_cylinder-wx*nz_cylinder);
	ucz=uz-rad*(wx*ny_cylinder-wy*nx_cylinder);

	ucx_w=-rad*(wy*nz_cylinder-wz*ny_cylinder);
	ucy_w=-rad*(wz*nx_cylinder-wx*nz_cylinder);
	ucz_w=-rad*(wx*ny_cylinder-wy*nx_cylinder);

	ucn_mag=ucx*nx_cylinder+ucy*ny_cylinder+ucz*nz_cylinder;
	uc_wn_mag=ucx_w*nx_cylinder+ucy_w*ny_cylinder+ucz_w*nz_cylinder;

	ucs_x=ucx-ucn_mag*nx_cylinder;
	ucs_y=ucy-ucn_mag*ny_cylinder;
	ucs_z=ucz-ucn_mag*nz_cylinder;

	uc_ws_x=ucx_w-uc_wn_mag*nx_cylinder;
	uc_ws_y=ucy_w-uc_wn_mag*ny_cylinder;
	uc_ws_z=ucz_w-uc_wn_mag*nz_cylinder;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_cylinder=ucs_x/(ucs_mag+1e-20);
	sy_cylinder=ucs_y/(ucs_mag+1e-20);
	sz_cylinder=ucs_z/(ucs_mag+1e-20);

	swx_cylinder=uc_ws_x/(uc_ws_mag+1e-20);
	swy_cylinder=uc_ws_y/(uc_ws_mag+1e-20);
	swz_cylinder=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*sx_cylinder;
		ffy=mu_s*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*sy_cylinder;
		ffz=mu_s*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*sz_cylinder;

		frfx=mu_rs*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*swx_cylinder;
		frfy=mu_rs*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*swy_cylinder;
		frfz=mu_rs*m*(ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)*swz_cylinder;

		if ((ftotalx*nx_cylinder+ftotaly*ny_cylinder+ftotalz*nz_cylinder)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_cylinder*(ffz+frfz)-nz_cylinder*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_cylinder*(ffx+frfx)-nx_cylinder*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_cylinder*(ffy+frfy)-ny_cylinder*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_cylinder+uy_c*ny_cylinder+uz_c*nz_cylinder;

		TP1[i].ux=ux_c-u_cn_mag*nx_cylinder;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_cylinder;
		TP1[i].uz=uz_c-u_cn_mag*nz_cylinder;

		TP1[i].wx=wx-(rad/ri)*(ny_cylinder*(ffz+frfz)-nz_cylinder*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_cylinder*(ffx+frfx)-nx_cylinder*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_cylinder*(ffy+frfy)-ny_cylinder*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_cylinder;
		Jny=-m*(1+rest_n)*ucn_mag*ny_cylinder;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_cylinder;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_cylinder;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_cylinder;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_cylinder;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_cylinder*Jz-nz_cylinder*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_cylinder*Jx-nx_cylinder*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_cylinder*Jy-ny_cylinder*Jx);


	}


}
////////////////////////////////////////////////////////////////////////



#define xz_cone		0.0
#define yz_cone 	0.0
#define zz_cone		0.0

#define r_cone	 	1.5
#define h_cone		1.5

__global__ void KERNEL_treat_DEM_cone_z(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;

	

	
	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_cone;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_cone_mag;
	Real nx_cone,ny_cone,nz_cone;
	Real sx_cone,sy_cone,sz_cone;
	Real swx_cone,swy_cone,swz_cone;

	Real x_center=xz_cone;
	Real y_center=yz_cone;
	Real z_center=zz_cone;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	if(z>=0) return;

	F_cone=(x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)-(1.5-sqrt(2.0)*rad+z)*(1.5-sqrt(2.0)*rad+z);

	if(F_cone<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x;
	cpy=y;
	cpz=sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center))-1.5+sqrt(2.0)*rad;


	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	nx_cone=-(cpx-x_center)/(sqrt(2.0)*sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)));
	ny_cone=-(cpy-y_center)/(sqrt(2.0)*sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)));
	nz_cone=sqrt((cpx-x_center)*(cpx-x_center)+(cpy-y_center)*(cpy-y_center))/(sqrt(2.0)*sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)));

	n_cone_mag=sqrt(nx_cone*nx_cone+ny_cone*ny_cone+nz_cone*nz_cone);


	//if(n_cone_mag<0.5) return;		// actually do not contact the cone wall yet...

	un_mag=ux*nx_cone+uy*ny_cone+uz*nz_cone;

	ucx=ux-rad*(wy*nz_cone-wz*ny_cone);
	ucy=uy-rad*(wz*nx_cone-wx*nz_cone);
	ucz=uz-rad*(wx*ny_cone-wy*nx_cone);

	ucx_w=-rad*(wy*nz_cone-wz*ny_cone);
	ucy_w=-rad*(wz*nx_cone-wx*nz_cone);
	ucz_w=-rad*(wx*ny_cone-wy*nx_cone);

	ucn_mag=ucx*nx_cone+ucy*ny_cone+ucz*nz_cone;
	uc_wn_mag=ucx_w*nx_cone+ucy_w*ny_cone+ucz_w*nz_cone;

	ucs_x=ucx-ucn_mag*nx_cone;
	ucs_y=ucy-ucn_mag*ny_cone;
	ucs_z=ucz-ucn_mag*nz_cone;

	uc_ws_x=ucx_w-uc_wn_mag*nx_cone;
	uc_ws_y=ucy_w-uc_wn_mag*ny_cone;
	uc_ws_z=ucz_w-uc_wn_mag*nz_cone;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_cone=ucs_x/(ucs_mag+1e-20);
	sy_cone=ucs_y/(ucs_mag+1e-20);
	sz_cone=ucs_z/(ucs_mag+1e-20);

	swx_cone=uc_ws_x/(uc_ws_mag+1e-20);
	swy_cone=uc_ws_y/(uc_ws_mag+1e-20);
	swz_cone=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*sx_cone;
		ffy=mu_s*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*sy_cone;
		ffz=mu_s*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*sz_cone;

		frfx=mu_rs*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*swx_cone;
		frfy=mu_rs*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*swy_cone;
		frfz=mu_rs*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*swz_cone;

		if ((ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_cone*(ffz+frfz)-nz_cone*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_cone*(ffx+frfx)-nx_cone*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_cone*(ffy+frfy)-ny_cone*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_cone+uy_c*ny_cone+uz_c*nz_cone;

		TP1[i].ux=ux_c-u_cn_mag*nx_cone;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_cone;
		TP1[i].uz=uz_c-u_cn_mag*nz_cone;

		TP1[i].wx=wx-(rad/ri)*(ny_cone*(ffz+frfz)-nz_cone*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_cone*(ffx+frfx)-nx_cone*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_cone*(ffy+frfy)-ny_cone*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_cone;
		Jny=-m*(1+rest_n)*ucn_mag*ny_cone;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_cone;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_cone;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_cone;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_cone;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_cone*Jz-nz_cone*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_cone*Jx-nx_cone*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_cone*Jy-ny_cone*Jx);


	}


}
//////////////////////////////



#define xx_box		0.0
#define yx_box 		0.0
#define zx_box		0.0

#define Lxx_box 	0.4
#define Lyx_box  	10.0
#define Lzx_box 	10.0


__global__ void KERNEL_treat_DEM_box_x(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xx_box;
	Real y_center=yx_box;
	Real z_center=zx_box;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxx_box-rad)),(abs(y-y_center)-(0.5*Lyx_box-rad))),(abs(z-z_center)-(0.5*Lzx_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxx_box-rad),fmax(-(0.5*Lxx_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyx_box-rad),fmax(-(0.5*Lyx_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzx_box-rad),fmax(-(0.5*Lzx_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}




#define xy_box		0.0
#define yy_box 		0.0
#define zy_box		0.0

#define Lxy_box 	10.0
#define Lyy_box  	0.04
#define Lzy_box 	10.0


__global__ void KERNEL_treat_DEM_box_y(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xy_box;
	Real y_center=yy_box;
	Real z_center=zy_box;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxy_box-rad)),(abs(y-y_center)-(0.5*Lyy_box-rad))),(abs(z-z_center)-(0.5*Lzy_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxy_box-rad),fmax(-(0.5*Lxy_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyy_box-rad),fmax(-(0.5*Lyy_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzy_box-rad),fmax(-(0.5*Lzy_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt)) ) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}




#define xz_box		0.0
#define yz_box 		0.0
#define zz_box		-0.00176647182

#define Lxz_box 	1.15
#define Lyz_box  	1.015
#define Lzz_box 	0.03353294364


__global__ void KERNEL_treat_DEM_box_z(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)		// 아래쪽 경계면
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xz_box;
	Real y_center=yz_box;
	Real z_center=zz_box;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxz_box-rad)),(abs(y-y_center)-(0.5*Lyz_box-rad))),(abs(z-z_center)-(0.5*Lzz_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxz_box-rad),fmax(-(0.5*Lxz_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyz_box-rad),fmax(-(0.5*Lyz_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzz_box-rad),fmax(-(0.5*Lzz_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}



__global__ void KERNEL_treat_DEM_box_z_dem(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)		// 아래쪽 경계면
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xz_box;
	Real y_center=yz_box;
	Real z_center=zz_box;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxz_box-rad)),(abs(y-y_center)-(0.5*Lyz_box-rad))),(abs(z-z_center)-(0.5*Lzz_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxz_box-rad),fmax(-(0.5*Lxz_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyz_box-rad),fmax(-(0.5*Lyz_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzz_box-rad),fmax(-(0.5*Lzz_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}
//////////////////////////////////////////////


#define xza_box		0.0
#define yza_box 		0.0
#define zza_box		4.0

#define Lxza_box 	5.0
#define Lyza_box  	5.0
#define Lzza_box 	16.0


__global__ void KERNEL_treat_DEM_box_za(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)		//size 딱 맞는 box_z
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xza_box;
	Real y_center=yza_box;
	Real z_center=zza_box-0.0016*(ttime);


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxza_box-rad)),(abs(y-y_center)-(0.5*Lyza_box-rad))),(abs(z-z_center)-(0.5*Lzza_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxza_box-rad),fmax(-(0.5*Lxza_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyza_box-rad),fmax(-(0.5*Lyza_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzza_box-rad),fmax(-(0.5*Lzza_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz-0.00001;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}


#define xz25l_box		0.0
#define yz25l_box 		0.0
#define zz25l_box		0.218

#define Lxz25_box 	10.0
#define Lyz25_box  	10.0
#define Lzz25_box 	6.0


__global__ void KERNEL_treat_DEM_box_z25l(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xz25l_box;
	Real y_center=yz25l_box;
	//Real z_center=zz25l_box+0.008*(ttime-2.4)*(ttime>=2.4);
	Real z_center=zz25l_box-0.0016*(ttime);
	//Real z_center=zz25l_box


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=(z-z_center)-0.46631*x;

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=(z+0.00001-z_center)*2.14451;
	cpy=y;
	cpz=z;

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	// Real sgn_x, sgn_y, sgn_z;

	// sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	// sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	// sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	// if (abs(cpy-y)<1e-12) sgn_y=0.0;
	// if (abs(cpy-x)<1e-12) sgn_x=0.0;
	// if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=0.42262;
	ny_box=0.0;
	nz_box=-0.90630;

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}




#define xz25r_box		0.0
#define yz25r_box 		0.0
#define zz25r_box		0.218

#define Lxz25_box 	10.0
#define Lyz25_box  	10.0
#define Lzz25_box 	6.0


__global__ void KERNEL_treat_DEM_box_z25r(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xz25r_box;
	Real y_center=yz25r_box;
	//Real z_center=zz25r_box+0.008*(ttime-2.4)*(ttime>=2.4);
	Real z_center=zz25r_box-0.0016*(ttime);
	//Real z_center=zz25r_box


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=(z-z_center)+0.46631*x;

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=-(z+0.00001-z_center)*2.14451;
	cpy=y;
	cpz=z;

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	// Real sgn_x, sgn_y, sgn_z;

	// sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	// sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	// sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	// if (abs(cpy-y)<1e-12) sgn_y=0.0;
	// if (abs(cpy-x)<1e-12) sgn_x=0.0;
	// if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=-0.42262;
	ny_box=0.0;
	nz_box=-0.90630;

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}




///////////////////////////////////////////////


#define xz5_box		0.0
#define yz5_box 		0.0
#define zz5_box		-0.03824

#define Lxz5_box 	10.0
#define Lyz5_box  	10.0
#define Lzz5_box 	0.34244		//0.16842+0.003-0.0002


__global__ void KERNEL_treat_DEM_box_z5(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xz5_box;
	Real y_center=yz5_box;
	Real z_center=zz5_box+0.01*(ttime-0.1)*(ttime>=0.1);


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxz5_box-rad)),(abs(y-y_center)-(0.5*Lyz5_box-rad))),(abs(z-z_center)-(0.5*Lzz5_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxz5_box-rad),fmax(-(0.5*Lxz5_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyz5_box-rad),fmax(-(0.5*Lyz5_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzz5_box-rad),fmax(-(0.5*Lzz5_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}




#define xx1_box		-1.5
#define yx1_box 		0.0
#define zx1_box		0.0

#define Lxx1_box 	1.0
#define Lyx1_box  	10.0
#define Lzx1_box 	10.0


__global__ void KERNEL_treat_DEM_box_x1(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xx1_box;
	Real y_center=yx1_box;
	Real z_center=zx1_box;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxx1_box-rad)),(abs(y-y_center)-(0.5*Lyx1_box-rad))),(abs(z-z_center)-(0.5*Lzx1_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxx1_box-rad),fmax(-(0.5*Lxx1_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyx1_box-rad),fmax(-(0.5*Lyx1_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzx1_box-rad),fmax(-(0.5*Lzx1_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}


#define xxl_box		5.0
#define yxl_box 	0.0
#define zxl_box		0.0

#define Lxxl_box 	10.15
#define Lyxl_box  	10.0
#define Lzxl_box 	10.0


__global__ void KERNEL_treat_DEM_box_xl(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	Real x0,y0,z0;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ux0,uy0,uz0;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real wx0,wy0,wz0;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xxl_box;
	Real y_center=yxl_box;
	Real z_center=zxl_box;


	x0=TP1[i].x;
	y0=TP1[i].y;
	z0=TP1[i].z;

	x=x0*0.5-z0*0.8660254;
	z=x0*0.8660254+z0*0.5;
	y=y0;

	ux0=TP1[i].ux;
	uy0=TP1[i].uy;
	uz0=TP1[i].uz;

	ux=ux0*0.5-uz0*0.8660254;
	uz=ux0*0.8660254+uz0*0.5;
	uy=uy0;

	wx0=TP1[i].wx;
	wy0=TP1[i].wy;
	wz0=TP1[i].wz;
	
	wx=wx0*0.5-wz0*0.8660254;
	wz=wx0*0.8660254+wz0*0.5;
	wy=wy0;

	Real ftotalx,ftotaly,ftotalz;
	Real ftotalx0,ftotaly0,ftotalz0;
	Real torqx,torqy,torqz;
	Real torqx0,torqy0,torqz0;
	

	ftotalx0=P3[i].ftotalx;
	ftotaly0=P3[i].ftotaly;
	ftotalz0=P3[i].ftotalz;

	ftotalx=ftotalx0*0.5-ftotalz0*0.8660254;
	ftotalz=ftotalx0*0.8660254+ftotalz0*0.5;
	ftotaly=ftotaly0;

	torqx0=P3[i].torqx;
	torqy0=P3[i].torqy;
	torqz0=P3[i].torqz;

	torqx=torqx0*0.5-torqz0*0.8660254;
	torqz=torqx0*0.8660254+torqz0*0.5;
	torqy=torqy0;


	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxxl_box-rad)),(abs(y-y_center)-(0.5*Lyxl_box-rad))),(abs(z-z_center)-(0.5*Lzxl_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxxl_box-rad),fmax(-(0.5*Lxxl_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyxl_box-rad),fmax(-(0.5*Lyxl_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzxl_box-rad),fmax(-(0.5*Lzxl_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx*0.5+cpz*0.8660254;
	TP1[i].z=cpz*0.5-cpx*0.8660254;
	TP1[i].y=cpy;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt)) ) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx*0.5+ftotalz*0.8660254;
		P3[i].ftotalz=ftotalz*0.5-ftotalx*0.8660254;
		P3[i].ftotaly=ftotaly;

		P3[i].torqx=torqx*0.5+torqz*0.8660254;
		P3[i].torqz=torqz*0.5-torqx*0.8660254;
		P3[i].torqy=torqy;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=(ux_c-u_cn_mag*nx_box)*0.5+(uz_c-u_cn_mag*nz_box)*0.8660254;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uz=(uz_c-u_cn_mag*nz_box)*0.5-(ux_c-u_cn_mag*nx_box)*0.8660254;
		TP1[i].uy=uy_c-u_cn_mag*ny_box;

		TP1[i].wx=(wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt)*0.5+(wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt)*0.8660254;
		TP1[i].wz=(wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt)*0.5-(wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt)*0.8660254;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=(ux+(1/m)*Jx)*0.5+(uz+(1/m)*Jz)*0.8660254;
		TP1[i].uz=(uz+(1/m)*Jz)*0.5-(ux+(1/m)*Jx)*0.8660254;
		TP1[i].uy=uy+(1/m)*Jy;

		// angular velocity KERNEL_update
		TP1[i].wx=(wx-(rad/ri)*(ny_box*Jz-nz_box*Jy))*0.5+(wz-(rad/ri)*(nx_box*Jy-ny_box*Jx))*0.8660254;
		TP1[i].wz=(wz-(rad/ri)*(nx_box*Jy-ny_box*Jx))*0.5-(wx-(rad/ri)*(ny_box*Jz-nz_box*Jy))*0.8660254;
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);


	}





}




#define xxr_box		-5.0
#define yxr_box 	0.0
#define zxr_box		0.0

#define Lxxr_box 	10.15
#define Lyxr_box  	10.0
#define Lzxr_box 	10.0


__global__ void KERNEL_treat_DEM_box_xr(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;


	Real x,y,z;
	Real x0,y0,z0;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ux0,uy0,uz0;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real wx0,wy0,wz0;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xxr_box;
	Real y_center=yxr_box;
	Real z_center=zxr_box;


	x0=TP1[i].x;
	y0=TP1[i].y;
	z0=TP1[i].z;

	x=x0*0.5+z0*0.8660254;
	z=-x0*0.8660254+z0*0.5;
	y=y0;

	ux0=TP1[i].ux;
	uy0=TP1[i].uy;
	uz0=TP1[i].uz;

	ux=ux0*0.5+uz0*0.8660254;
	uz=-ux0*0.8660254+uz0*0.5;
	uy=uy0;

	wx0=TP1[i].wx;
	wy0=TP1[i].wy;
	wz0=TP1[i].wz;
	
	wx=wx0*0.5+wz0*0.8660254;
	wz=-wx0*0.8660254+wz0*0.5;
	wy=wy0;

	Real ftotalx,ftotaly,ftotalz;
	Real ftotalx0,ftotaly0,ftotalz0;
	Real torqx,torqy,torqz;
	Real torqx0,torqy0,torqz0;
	

	ftotalx0=P3[i].ftotalx;
	ftotaly0=P3[i].ftotaly;
	ftotalz0=P3[i].ftotalz;

	ftotalx=ftotalx0*0.5+ftotalz0*0.8660254;
	ftotalz=-ftotalx0*0.8660254+ftotalz0*0.5;
	ftotaly=ftotaly0;

	torqx0=P3[i].torqx;
	torqy0=P3[i].torqy;
	torqz0=P3[i].torqz;

	torqx=torqx0*0.5+torqz0*0.8660254;
	torqz=-torqx0*0.8660254+torqz0*0.5;
	torqy=torqy0;


	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	F_box=fmax(fmax((abs(x-x_center)-(0.5*Lxxr_box-rad)),(abs(y-y_center)-(0.5*Lyxr_box-rad))),(abs(z-z_center)-(0.5*Lzxr_box-rad)));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxxr_box-rad),fmax(-(0.5*Lxxr_box-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyxr_box-rad),fmax(-(0.5*Lyxr_box-rad), y-y_center));
	cpz=z_center+fmin((0.5*Lzxr_box-rad),fmax(-(0.5*Lzxr_box-rad), z-z_center));

	

	//position update
	TP1[i].x=cpx*0.5-cpz*0.8660254;
	TP1[i].z=cpz*0.5+cpx*0.8660254;
	TP1[i].y=cpy;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	//if (abs(ux)<1e-8) ux=0.0;
	//if (abs(uy)<1e-8) uy=0.0;
	//if (abs(uz)<1e-8) uz=0.0;
	//sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	//normal vector
	//nx_box=ux/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//ny_box=uy/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);
	//nz_box=uz/(sqrt(ux*ux+uy*uy+uz*uz)+1e-20);

	nx_box=sgn_x/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	ny_box=sgn_y/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);
	nz_box=sgn_z/(sqrt(sgn_x*sgn_x+sgn_y*sgn_y+sgn_z*sgn_z)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx*0.5-ftotalz*0.8660254;
		P3[i].ftotalz=ftotalz*0.5+ftotalx*0.8660254;
		P3[i].ftotaly=ftotaly;

		P3[i].torqx=torqx*0.5-torqz*0.8660254;
		P3[i].torqz=torqz*0.5+torqx*0.8660254;
		P3[i].torqy=torqy;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=(ux_c-u_cn_mag*nx_box)*0.5-(uz_c-u_cn_mag*nz_box)*0.8660254;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uz=(uz_c-u_cn_mag*nz_box)*0.5+(ux_c-u_cn_mag*nx_box)*0.8660254;
		TP1[i].uy=uy_c-u_cn_mag*ny_box;

		TP1[i].wx=(wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt)*0.5-(wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt)*0.8660254;
		TP1[i].wz=(wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt)*0.5+(wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt)*0.8660254;
		TP1[i].wy=(wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt);


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=(ux+(1/m)*Jx)*0.5-(uz+(1/m)*Jz)*0.8660254;
		TP1[i].uz=(uz+(1/m)*Jz)*0.5+(ux+(1/m)*Jx)*0.8660254;
		TP1[i].uy=uy+(1/m)*Jy;

		// angular velocity KERNEL_update
		TP1[i].wx=(wx-(rad/ri)*(ny_box*Jz-nz_box*Jy))*0.5-(wz-(rad/ri)*(nx_box*Jy-ny_box*Jx))*0.8660254;
		TP1[i].wz=(wz-(rad/ri)*(nx_box*Jy-ny_box*Jx))*0.5+(wx-(rad/ri)*(ny_box*Jz-nz_box*Jy))*0.8660254;
		TP1[i].wy=(wy-(rad/ri)*(nz_box*Jx-nx_box*Jz));


	}





}




#define xzp_box		0.0
#define yzp_box 	0.0
#define zzp_box		-0.053

#define Lxzp1_box 	0.15
#define Lxzp2_box 	0.015
#define Lyzp_box  	0.015
#define Lzzp_box 	0.106



__global__ void KERNEL_treat_DEM_pyramid_z(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)		// 아래쪽 경계면
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;

	Real h_gamma = abs(Lxzp1_box-Lxzp2_box)/2;
	Real L_gamma = sqrt(h_gamma*h_gamma+Lzzp_box*Lzzp_box+1e-20);
	Real cos_gamma	= Lzzp_box/L_gamma;
	Real sin_gamma 	= h_gamma/L_gamma;
	Real tan_gamma	= h_gamma/Lzzp_box;

	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_box;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_box_mag;
	Real nx_box,ny_box,nz_box;
	Real sx_box,sy_box,sz_box;
	Real swx_box,swy_box,swz_box;

	Real x_center=xzp_box;
	Real y_center=yzp_box;
	Real z_center=zzp_box;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	Real Lxzp_star = Lxzp2_box+2*(Lzzp_box+z)*tan_gamma;

	F_box=fmax(abs(x-x_center)-(0.5*Lxzp_star-rad),abs(y-y_center)-(0.5*Lyzp_box-rad));

	if(F_box<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x_center+fmin((0.5*Lxzp_star-rad),fmax(-(0.5*Lxzp_star-rad), x-x_center));
	cpy=y_center+fmin((0.5*Lyzp_box-rad),fmax(-(0.5*Lyzp_box-rad), y-y_center));
	cpz=z;

	

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	// sgn_x, sgn_y, sgn_z
	//Real sgn_x, sgn_y, sgn_z;
	Real sgn_x, sgn_y, sgn_z;

	sgn_x=(cpx-x)/(abs(cpx-x)+1e-20);
	sgn_y=(cpy-y)/(abs(cpy-y)+1e-20);
	sgn_z=(cpz-z)/(abs(cpz-z)+1e-20);

	if (abs(cpy-y)<1e-12) sgn_y=0.0;
	if (abs(cpy-x)<1e-12) sgn_x=0.0;
	if (abs(cpz-z)<1e-12) sgn_z=0.0;

	Real npxx, npxy, npxz;
	Real npyx, npyy, npyz;
	Real Npx, Npy, Npz;

	npxx = sgn_x * cos_gamma;
	npxy = 0;
	npxz = -sgn_x * sin_gamma;

	npyx = 0;
	npyy = sgn_y;
	npyz = 0;

	Npx = npxx + npyx;
	Npy = npxy + npyy;
	Npz = npxz + npyz;

	nx_box=Npx/(sqrt(Npx*Npx+Npy*Npy+Npz*Npz)+1e-20);
	ny_box=Npy/(sqrt(Npx*Npx+Npy*Npy+Npz*Npz)+1e-20);
	nz_box=Npz/(sqrt(Npx*Npx+Npy*Npy+Npz*Npz)+1e-20);

	n_box_mag=sqrt(nx_box*nx_box+ny_box*ny_box+nz_box*nz_box);


	if(n_box_mag<0.5) return;		// actually do not contact the wall yet...

	un_mag=ux*nx_box+uy*ny_box+uz*nz_box;

	ucx=ux-rad*(wy*nz_box-wz*ny_box);
	ucy=uy-rad*(wz*nx_box-wx*nz_box);
	ucz=uz-rad*(wx*ny_box-wy*nx_box);

	ucx_w=-rad*(wy*nz_box-wz*ny_box);
	ucy_w=-rad*(wz*nx_box-wx*nz_box);
	ucz_w=-rad*(wx*ny_box-wy*nx_box);

	ucn_mag=ucx*nx_box+ucy*ny_box+ucz*nz_box;
	uc_wn_mag=ucx_w*nx_box+ucy_w*ny_box+ucz_w*nz_box;

	ucs_x=ucx-ucn_mag*nx_box;
	ucs_y=ucy-ucn_mag*ny_box;
	ucs_z=ucz-ucn_mag*nz_box;

	uc_ws_x=ucx_w-uc_wn_mag*nx_box;
	uc_ws_y=ucy_w-uc_wn_mag*ny_box;
	uc_ws_z=ucz_w-uc_wn_mag*nz_box;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_box=ucs_x/(ucs_mag+1e-20);
	sy_box=ucs_y/(ucs_mag+1e-20);
	sz_box=ucs_z/(ucs_mag+1e-20);

	swx_box=uc_ws_x/(uc_ws_mag+1e-20);
	swy_box=uc_ws_y/(uc_ws_mag+1e-20);
	swz_box=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sx_box;
		ffy=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sy_box;
		ffz=mu_s*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*sz_box;

		frfx=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swx_box;
		frfy=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swy_box;
		frfz=mu_rs*m*(ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)*swz_box;

		if ((ftotalx*nx_box+ftotaly*ny_box+ftotalz*nz_box)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_box+uy_c*ny_box+uz_c*nz_box;

		TP1[i].ux=ux_c-u_cn_mag*nx_box;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_box;
		TP1[i].uz=uz_c-u_cn_mag*nz_box;

		TP1[i].wx=wx-(rad/ri)*(ny_box*(ffz+frfz)-nz_box*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_box*(ffx+frfx)-nx_box*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_box*(ffy+frfy)-ny_box*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_box;
		Jny=-m*(1+rest_n)*ucn_mag*ny_box;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_box;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_box;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_box;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_box;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_box*Jz-nz_box*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_box*Jx-nx_box*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_box*Jy-ny_box*Jx);


	}





}
//////////////////////////////////////////////


#define xzt_cone	0.0
#define yzt_cone 	0.0
#define zzt_cone	0.0

#define r1_cone	 	0.0125
#define r2_cone		0.0018
#define h_cone		0.01853294364

__global__ void KERNEL_treat_DEM_truncated_cone_z(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;

	

	
	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_cone;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_cone_mag;
	Real nx_cone,ny_cone,nz_cone;
	Real sx_cone,sy_cone,sz_cone;
	Real swx_cone,swy_cone,swz_cone;

	Real x_center=xz_cone;
	Real y_center=yz_cone;
	Real z_center=zz_cone;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	if(z<=-h_cone+rad) return;

	Real sin_trunc = h_cone/sqrt((r1_cone-r2_cone)*(r1_cone-r2_cone)+h_cone*h_cone);
	Real cos_trunc = abs((r1_cone-r2_cone))/sqrt((r1_cone-r2_cone)*(r1_cone-r2_cone)+h_cone*h_cone);
	Real tan_trunc = sin_trunc/cos_trunc;

	F_cone=(x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)-(r2_cone+(h_cone+z)/tan_trunc-rad/sin_trunc)*(r2_cone+(h_cone+z)/tan_trunc-rad/sin_trunc);

	if(F_cone<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x;
	cpy=y;
	//cpz=tan_trunc*sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center))-h_cone+rad/cos_trunc;
	cpz=tan_trunc*(sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center))+rad/sin_trunc-r2_cone)-h_cone;

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	nx_cone=-sin_trunc*(cpx-x_center)/(sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)));
	ny_cone=-sin_trunc*(cpy-y_center)/(sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)));
	nz_cone=cos_trunc*1;

	n_cone_mag=sqrt(nx_cone*nx_cone+ny_cone*ny_cone+nz_cone*nz_cone);


	//if(n_cone_mag<0.5) return;		// actually do not contact the cone wall yet...

	un_mag=ux*nx_cone+uy*ny_cone+uz*nz_cone;

	ucx=ux-rad*(wy*nz_cone-wz*ny_cone);
	ucy=uy-rad*(wz*nx_cone-wx*nz_cone);
	ucz=uz-rad*(wx*ny_cone-wy*nx_cone);

	ucx_w=-rad*(wy*nz_cone-wz*ny_cone);
	ucy_w=-rad*(wz*nx_cone-wx*nz_cone);
	ucz_w=-rad*(wx*ny_cone-wy*nx_cone);

	ucn_mag=ucx*nx_cone+ucy*ny_cone+ucz*nz_cone;
	uc_wn_mag=ucx_w*nx_cone+ucy_w*ny_cone+ucz_w*nz_cone;

	ucs_x=ucx-ucn_mag*nx_cone;
	ucs_y=ucy-ucn_mag*ny_cone;
	ucs_z=ucz-ucn_mag*nz_cone;

	uc_ws_x=ucx_w-uc_wn_mag*nx_cone;
	uc_ws_y=ucy_w-uc_wn_mag*ny_cone;
	uc_ws_z=ucz_w-uc_wn_mag*nz_cone;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_cone=ucs_x/(ucs_mag+1e-20);
	sy_cone=ucs_y/(ucs_mag+1e-20);
	sz_cone=ucs_z/(ucs_mag+1e-20);

	swx_cone=uc_ws_x/(uc_ws_mag+1e-20);
	swy_cone=uc_ws_y/(uc_ws_mag+1e-20);
	swz_cone=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*sx_cone;
		ffy=mu_s*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*sy_cone;
		ffz=mu_s*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*sz_cone;

		frfx=mu_rs*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*swx_cone;
		frfy=mu_rs*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*swy_cone;
		frfz=mu_rs*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*swz_cone;

		if ((ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_cone*(ffz+frfz)-nz_cone*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_cone*(ffx+frfx)-nx_cone*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_cone*(ffy+frfy)-ny_cone*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_cone+uy_c*ny_cone+uz_c*nz_cone;

		TP1[i].ux=ux_c-u_cn_mag*nx_cone;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_cone;
		TP1[i].uz=uz_c-u_cn_mag*nz_cone;

		TP1[i].wx=wx-(rad/ri)*(ny_cone*(ffz+frfz)-nz_cone*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_cone*(ffx+frfx)-nx_cone*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_cone*(ffy+frfy)-ny_cone*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_cone;
		Jny=-m*(1+rest_n)*ucn_mag*ny_cone;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_cone;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_cone;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_cone;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_cone;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_cone*Jz-nz_cone*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_cone*Jx-nx_cone*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_cone*Jy-ny_cone*Jx);


	}


}



__global__ void KERNEL_treat_DEM_truncated_cone_z_dem(int_t inout,part1*P1,part1*TP1,part2*P2,part3*P3,Real ttime)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2_dem) return;
	if(P1[i].i_type!=inout) return;
	if(P1[i].p_type<=1000) return;

	

	
	Real x,y,z;
	//Real x,y,z;
	Real rad;
	//Real ux,uy,uz;
	Real ux,uy,uz;
	Real ucx,ucy,ucz; 		// contact velocity
	Real ucx_w,ucy_w,ucz_w;	// contact velocity component due to the rolling of the DEM particles
	Real ucs_x,ucs_y,ucs_z;	// contact velocity in shear direction
	Real uc_ws_x,uc_ws_y,uc_ws_z;	// uc_w in shear direction (maybe same with ucx_w, ucy_w, ucz_w)
	Real ucn_mag,ucs_mag;	// magnitude of contact velocity vector in surface normal & shear direction
	Real uc_wn_mag,uc_ws_mag;	
	Real un_mag;			// magnitude of initial veloicty vector in surface normal direction 
	Real wx,wy,wz;
	Real m,ri;

	Real F_cone;

	//Real cpx,cpy,cpz;
	Real cpx,cpy,cpz;
	Real dth;

	Real Jx,Jy,Jz;			// linear momentum
	Real Jnx,Jny,Jnz;		// linear momentum in normal direction
	Real Jsx,Jsy,Jsz;		// linear momentum in shear direction
	
	Real n_cone_mag;
	Real nx_cone,ny_cone,nz_cone;
	Real sx_cone,sy_cone,sz_cone;
	Real swx_cone,swy_cone,swz_cone;

	Real x_center=xz_cone;
	Real y_center=yz_cone;
	Real z_center=zz_cone;


	x=TP1[i].x;
	y=TP1[i].y;
	z=TP1[i].z;

	ux=TP1[i].ux;
	uy=TP1[i].uy;
	uz=TP1[i].uz;

	wx=TP1[i].wx;
	wy=TP1[i].wy;
	wz=TP1[i].wz;

	Real ftotalx,ftotaly,ftotalz;
	Real torqx,torqy,torqz;
	

	ftotalx=P3[i].ftotalx;
	ftotaly=P3[i].ftotaly;
	ftotalz=P3[i].ftotalz;

	torqx=P3[i].torqx;
	torqy=P3[i].torqy;
	torqz=P3[i].torqz;



	m=TP1[i].m;
	ri=TP1[i].ri;

	rad=TP1[i].rad;

	if(z<=-h_cone+rad) return;

	Real sin_trunc = h_cone/sqrt((r1_cone-r2_cone)*(r1_cone-r2_cone)+h_cone*h_cone);
	Real cos_trunc = abs((r1_cone-r2_cone))/sqrt((r1_cone-r2_cone)*(r1_cone-r2_cone)+h_cone*h_cone);
	Real tan_trunc = sin_trunc/cos_trunc;

	F_cone=(x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)-(r2_cone+(h_cone+z)/tan_trunc-rad/sin_trunc)*(r2_cone+(h_cone+z)/tan_trunc-rad/sin_trunc);

	if(F_cone<=0) return;		// leave if the particle is inside.

	//contact point
	cpx=x;
	cpy=y;
	//cpz=tan_trunc*sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center))-h_cone+rad/cos_trunc;
	cpz=tan_trunc*(sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center))+rad/sin_trunc-r2_cone)-h_cone;

	//position update
	TP1[i].x=cpx;
	TP1[i].y=cpy;
	TP1[i].z=cpz;
	

	nx_cone=-sin_trunc*(cpx-x_center)/(sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)));
	ny_cone=-sin_trunc*(cpy-y_center)/(sqrt((x-x_center)*(x-x_center)+(y-y_center)*(y-y_center)));
	nz_cone=cos_trunc*1;

	n_cone_mag=sqrt(nx_cone*nx_cone+ny_cone*ny_cone+nz_cone*nz_cone);


	//if(n_cone_mag<0.5) return;		// actually do not contact the cone wall yet...

	un_mag=ux*nx_cone+uy*ny_cone+uz*nz_cone;

	ucx=ux-rad*(wy*nz_cone-wz*ny_cone);
	ucy=uy-rad*(wz*nx_cone-wx*nz_cone);
	ucz=uz-rad*(wx*ny_cone-wy*nx_cone);

	ucx_w=-rad*(wy*nz_cone-wz*ny_cone);
	ucy_w=-rad*(wz*nx_cone-wx*nz_cone);
	ucz_w=-rad*(wx*ny_cone-wy*nx_cone);

	ucn_mag=ucx*nx_cone+ucy*ny_cone+ucz*nz_cone;
	uc_wn_mag=ucx_w*nx_cone+ucy_w*ny_cone+ucz_w*nz_cone;

	ucs_x=ucx-ucn_mag*nx_cone;
	ucs_y=ucy-ucn_mag*ny_cone;
	ucs_z=ucz-ucn_mag*nz_cone;

	uc_ws_x=ucx_w-uc_wn_mag*nx_cone;
	uc_ws_y=ucy_w-uc_wn_mag*ny_cone;
	uc_ws_z=ucz_w-uc_wn_mag*nz_cone;

	if (abs(ucs_x)<1e-10) ucs_x=0.0;
	if (abs(ucs_y)<1e-10) ucs_y=0.0;
	if (abs(ucs_z)<1e-10) ucs_z=0.0;

	if (abs(uc_ws_x)<1e-10) uc_ws_x=0.0;
	if (abs(uc_ws_y)<1e-10) uc_ws_y=0.0;
	if (abs(uc_ws_z)<1e-10) uc_ws_z=0.0;

	ucs_mag=sqrt(ucs_x*ucs_x+ucs_y*ucs_y+ucs_z*ucs_z);
	uc_ws_mag=sqrt(uc_ws_x*uc_ws_x+uc_ws_y*uc_ws_y+uc_ws_z*uc_ws_z);

	sx_cone=ucs_x/(ucs_mag+1e-20);
	sy_cone=ucs_y/(ucs_mag+1e-20);
	sz_cone=ucs_z/(ucs_mag+1e-20);

	swx_cone=uc_ws_x/(uc_ws_mag+1e-20);
	swy_cone=uc_ws_y/(uc_ws_mag+1e-20);
	swz_cone=uc_ws_z/(uc_ws_mag+1e-20);

	


	if ((abs(un_mag)<(50*k_dt))) {		// rolling on the surface

		Real ffx,ffy,ffz;				// friction force vector in surface (shear direction)
		Real frfx,frfy,frfz;			// rolling friction in surface (rolling direction in surface)
		Real ux_c,uy_c,uz_c;			// updated velocity
		Real u_cn_mag;					// magnitude of normal direction updated velocity					

		ffx=mu_s*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*sx_cone;
		ffy=mu_s*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*sy_cone;
		ffz=mu_s*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*sz_cone;

		frfx=mu_rs*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*swx_cone;
		frfy=mu_rs*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*swy_cone;
		frfz=mu_rs*m*(ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)*swz_cone;

		if ((ftotalx*nx_cone+ftotaly*ny_cone+ftotalz*nz_cone)>0.0)
		{
			ffx=0.0;
			ffy=0.0;
			ffz=0.0;

			frfx=0.0;
			frfy=0.0;
			frfz=0.0;
		}

		ftotalx+=(ffx)/m;
		ftotaly+=(ffy)/m;
		ftotalz+=(ffz)/m;

		torqx+=-(rad/ri)*(ny_cone*(ffz+frfz)-nz_cone*(ffy+frfy));
		torqy+=-(rad/ri)*(nz_cone*(ffx+frfx)-nx_cone*(ffz+frfz));
		torqz+=-(rad/ri)*(nx_cone*(ffy+frfy)-ny_cone*(ffx+frfx));

		P3[i].ftotalx=ftotalx;
		P3[i].ftotaly=ftotaly;
		P3[i].ftotalz=ftotalz;

		P3[i].torqx=torqx;
		P3[i].torqy=torqy;
		P3[i].torqz=torqz;

		ux_c=ux+((ffx)/m)*k_dt;
		uy_c=uy+((ffy)/m)*k_dt;
		uz_c=uz+((ffz)/m)*k_dt;

		u_cn_mag=ux_c*nx_cone+uy_c*ny_cone+uz_c*nz_cone;

		TP1[i].ux=ux_c-u_cn_mag*nx_cone;					// normal direction updated velocity = 0   &   shear direction velocity : based on rolling&sliding dynamics
		TP1[i].uy=uy_c-u_cn_mag*ny_cone;
		TP1[i].uz=uz_c-u_cn_mag*nz_cone;

		TP1[i].wx=wx-(rad/ri)*(ny_cone*(ffz+frfz)-nz_cone*(ffy+frfy))*k_dt;
		TP1[i].wy=wy-(rad/ri)*(nz_cone*(ffx+frfx)-nx_cone*(ffz+frfz))*k_dt;
		TP1[i].wz=wz-(rad/ri)*(nx_cone*(ffy+frfy)-ny_cone*(ffx+frfx))*k_dt;


	}
	else{

		Jnx=-m*(1+rest_n)*ucn_mag*nx_cone;
		Jny=-m*(1+rest_n)*ucn_mag*ny_cone;
		Jnz=-m*(1+rest_n)*ucn_mag*nz_cone;

		Jsx=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sx_cone;
		Jsy=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sy_cone;
		Jsz=-1/((1/m)+(rad*rad/ri))*(1+rest_s)*ucs_mag*sz_cone;

		Jx=Jnx+Jsx;
		Jy=Jny+Jsy;
		Jz=Jnz+Jsz;

		// velocity KERNEL_update
		TP1[i].ux=ux+(1/m)*Jx;
		TP1[i].uy=uy+(1/m)*Jy;
		TP1[i].uz=uz+(1/m)*Jz;

		// angular velocity KERNEL_update
		TP1[i].wx=wx-(rad/ri)*(ny_cone*Jz-nz_cone*Jy);
		TP1[i].wy=wy-(rad/ri)*(nz_cone*Jx-nx_cone*Jz);
		TP1[i].wz=wz-(rad/ri)*(nx_cone*Jy-ny_cone*Jx);


	}


}
